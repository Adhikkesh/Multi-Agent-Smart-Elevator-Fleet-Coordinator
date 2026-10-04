# PHASE 4 — The LiftZero brain: set-Transformer, imitation learning, DAgger, ONNX, and the learned bidder inside the real simulator

> **Prerequisites (do not start before):** Phase 3 is merged into `v2`
> (`src/elevator_mas/learning/` exists with `schema.py, features.py, recorder.py, dataset.py,
> twin/, env.py, evaluate.py`, and the CLI has `elevator learn …`). The user will tell you
> the base commit. Phase 2 (UI) is *not* required.
> The names below follow the Phase 3 prompt §9 (public API). **If the merged Phase 3 code differs,
> the code wins** — adapt, and list every adaptation in your final report.

Paste this whole file into Antigravity (Planning mode, strongest model).

---

## 0. Context and goal

Project: a multi-agent smart-elevator fleet (Mesa 3, `src/elevator_mas/`). Each car answers a
Contract-Net call-for-proposals (CFP) with a *bid* — today the marginal cost computed by an A*
search (`ElevatorAgent.marginal_cost`). The dispatcher awards the call to the lowest bid.
That classical bidder is the **teacher**.

**LiftZero** replaces the *bid computation inside each car* with a small neural network. This
phase delivers the first learned bidder, trained **only by imitation** of the teacher
(behaviour cloning + DAgger), and plugs it into the real simulator as a new dispatch strategy
`liftzero_bc`. Phase 5 will then improve it with reinforcement learning, Phase 6 will add
look-ahead search and the UI.

You will deliver:

1. the **model** (PyTorch, ~0.27 M parameters, permutation-equivariant set-Transformer),
2. the **imitation training pipeline** (losses, augmentation, schedules, 3 seeds, logging),
3. **offline evaluation** (agreement, regret, hard-decision accuracy, scaling split),
4. **ONNX export + onnxruntime inference** (CPU, no torch at demo time) with parity tests,
5. the **integration**: a learned bidder in the cars + a `liftzero_bc` strategy + shadow-teacher mode,
6. **closed-loop evaluation** in the real simulator and **two DAgger rounds**,
7. model card, docs, tests.

**Hard acceptance targets** (§12): ≥ 85 % teacher agreement on non-trivial validation decisions;
closed-loop average wait within ±5 % of the teacher in the four regimes (or an honest analysis of
why not), zero safety-invariant violations, ≤ 2 ms median inference for 8 cars.

---

## 1. Working agreement

* Branch `phase-4-imitation` from the updated `v2`. Small commits. **Do not push; do not touch `main`.**
* Python 3.12 / `uv`. Existing suite must stay green; `ruff check` and `ruff format --check` clean (line length 100).
* Dependencies:
  * **Training only** (new group, used together with the Phase 3 `learn` group — `uv sync --group learn --group lift-train`): `[dependency-groups] lift-train = ["torch>=2.4", "onnx>=1.16", "matplotlib", "pyyaml"]`. Use the **CPU** PyTorch wheel by default; if CUDA/MPS is available use it automatically (`device="auto"`), but nothing may *require* a GPU. Training must finish in ≤ 45 min on 8 CPU cores for the default preset.
  * **Runtime (base dependencies)**: add `onnxruntime>=1.18` (CPU) to the main `dependencies`, so a fresh `uv sync` can run the learned bidder **without torch**. Runtime code must import neither `torch` nor `onnx`.
* All randomness seeded; every training run writes a `run.json` (git sha, config, seed, data hash, torch version, wall time).
* Write a plan first. Keep all ML code in `src/elevator_mas/learning/lift/` (new package), the engine
  integration minimal and behaviour-neutral for the existing strategies (proof required, §7.3).

---

## 2. The model — `learning/lift/model.py`

### 2.1 Input/Output (the contract with Phases 5–6)

Input = the Phase 3 `Encoded` batch: `call [B,Kc]`, `cars [B,N,Kcar]`, `glob [B,Kg]`,
`mask [B,N]` (True = car exists), `eligible [B,N]`. `N ≤ MAX_CARS` (pad/truncate to the batch max).

```python
class LiftZeroNet(nn.Module):
    def forward(self, call, cars, glob, mask, eligible=None) -> NetOut

@dataclass
class NetOut:
    score:   Tensor  # [B,N]   predicted "cost" of giving the call to car i (LOWER = better)
    aux:     Tensor  # [B,N,4] predicted bid breakdown (wait, ride, crowding, energy)  (training aid, also shown in UI)
    value:   Tensor  # [B]     state value for the decision (trained in Phase 5; untrained here -> keep the head, init small, exclude from BC loss)
    attn:    Tensor  # [B,N]   attention of the call token over car tokens, last layer, head-averaged (explainability)
    logit_temp: Tensor # scalar  learnable log-temperature, see §2.3
```

### 2.2 Architecture (defaults; all in a `ModelConfig` dataclass loaded from YAML)

* Token embeddings: `call_mlp: Kc→d`, `car_mlp: Kcar→d`, `glob_mlp: Kg→d` (2-layer MLPs with GELU, LayerNorm).
  Add a learned **type embedding** (3 types: global, call, car) to the tokens. **No positional
  encoding and no car-id embedding** — the net must be permutation-*equivariant* over cars
  (outputs permute with the inputs). This is what lets one network serve fleets of any size.
* Sequence = `[glob, call, car_1 … car_N]`; padded cars are masked out of attention via the key
  padding mask.
* `L = 4` pre-LN Transformer encoder layers, `d = 64`, `heads = 4`, FFN `4·d = 256`, GELU,
  dropout 0.05 (train only). Use `torch.nn.MultiheadAttention(batch_first=True, need_weights=True)` for the last layer so `attn` is available; others may use fused attention.
* Heads (on car tokens after the encoder): `score_head: d→d→1`, `aux_head: d→d→4`;
  `value_head: concat(glob_token, call_token, masked-mean(car tokens)) → d → 1`.
* **Parameter budget 0.2–0.4 M** (print and assert in a test; target ≈ 0.27 M).
* Must be exportable to ONNX (opset 17) with dynamic axes on batch and `N`; avoid ops that
  don't export (no `torch.where` on Python scalars inside the graph, no data-dependent control flow, no
  `nn.MultiheadAttention` quirks that break dynamic masks — verify by actually exporting).

### 2.3 Policy semantics

Deterministic decision: `argmin_i score_i` over `mask & eligible` (ties → lowest car id).
Stochastic policy used by Phase 5: `π(i) = softmax_i(−score_i · exp(−logit_temp))` over eligible cars.
In this phase `logit_temp` is initialised to 0 and trained with the listwise loss (§3.1) so the
softmax is calibrated against the teacher.

**Eligibility is never learned.** Ineligible cars (out of service, fire mode, full) are masked in the
loss and in action selection, and in the real system they still answer `REFUSE` through the existing
classical logic (§7). The network prices only cars that can actually take the call.

### 2.4 Required properties (tested)

* Equivariance: permuting the car axis (with mask/eligible) permutes `score/aux/attn` identically (atol 1e-5).
* Padding invariance: adding masked padding cars does not change the scores of real cars (atol 1e-5).
* Size invariance sanity: scores for the same real cars are finite for N = 1, 2, 8, 16, 32.
* Parameter count in range; `forward` is deterministic in eval mode; no NaN under extreme-but-valid inputs.

---

## 3. Imitation training — `learning/lift/train_bc.py`

### 3.1 Targets and losses

Teacher fields from the Phase 3 shards: `teacher_cost[M,N]` (1e6 where refused/padded), `winner[M]`,
`teacher_wait/ride/crowding/energy[M,N]`, `eligible`, `mask`.

Work on the **eligible** set `E` of each decision; skip decisions with `|E| < 2` for the ranking
losses (they are trivial) but keep them in a small proportion (≤ 5 %) as sanity examples.

Let `y_i = log1p(teacher_cost_i)` (costs span orders of magnitude) and `ŷ_i = score_i`.

1. **Listwise cross-entropy** (primary): `L_list = −log softmax_{i∈E}(−ŷ_i · exp(−logit_temp))[winner]`.
2. **Cost regression** (calibration of the scores, makes them usable by MCTS/UI): `L_reg = Huber(ŷ_i − y_i, δ=1)` over `E`, mean over cars.
   Because the cost scale differs between decisions, also add a per-decision-centred term
   `L_rank = Huber((ŷ_i − mean_E ŷ) − (y_i − mean_E y), δ=0.5)`.
3. **Auxiliary breakdown**: `L_aux = Huber(aux_{i,k} − log1p(teacher_k_i))` for k ∈ {wait, ride, crowding, energy}, over `E`.
4. **Pairwise hinge** on the winner vs. the runner-up with margin 0.1·scale to sharpen hard decisions:
   `L_pair = max(0, m − (ŷ_runner_up − ŷ_winner))`, weight × 2 on **hard** decisions (§5 hard = margin < 10 % of best cost).

Total: `L = 1.0·L_list + 0.5·L_reg + 0.5·L_rank + 0.2·L_aux + 0.3·L_pair` (weights in config).

### 3.2 Augmentation (each batch, on the fly, in torch)

* **Random car permutation** (always).
* **Car dropout**: with p = 0.3 per decision, drop 1–3 *non-winner* cars (mask them out) as long as ≥ 2 cars remain
  — teaches variable fleet sizes. Dropping the winner is forbidden (labels would be wrong).
* **Feature noise**: Gaussian σ = 0.01 on continuous features only (never on one-hots, masks, booleans).
* Optional (config flag, off by default): **mirror augmentation** (flip floors: `floor→F−1−floor`, UP↔DOWN) is *not* valid
  (lobby is at the bottom). Do not implement; mention in the docs why.

### 3.3 Optimisation

AdamW (lr 1e-3, wd 1e-2, betas 0.9/0.99), linear warm-up 500 steps then cosine to 1e-5, batch 512,
gradient clip 1.0, up to 30 epochs with early stopping on validation *non-trivial agreement* (patience 4),
`torch.compile` off by default (portability), mixed precision off on CPU. DataLoader: use `ExpertDataset.iter_batches`
(memory-mapped, shuffled per epoch with seeded RNG); no Python per-sample loops.

Presets (YAML in `configs/learning/bc*.yaml`): `smoke` (≈ 3 min, 50 k decisions, 3 epochs — for CI/tests), `default` (full train
split, ≤ 45 min on 8 cores), `large` (d=128, for the ablation).

Train **3 seeds** (0,1,2). Select the checkpoint with the best *validation non-trivial agreement*; report mean ± std across seeds; the shipped model is the best seed.

### 3.4 Logging

CSV per epoch (`train_loss` components, `val_agree_nontrivial`, `val_agree_hard`, `val_regret`, lr, time) in
`runs/<id>/metrics.csv`, plus `curves.png` (matplotlib). No TensorBoard dependency. `runs/` is git-ignored except the *final* run summary copied to `docs/lift/`.

---

## 4. Offline evaluation — `learning/lift/eval_offline.py`

On the **val**, **test** and **test_large** (held-out 33–40 floors × 7–8 cars) splits report, per split and per teacher (`cnp_astar`, `full`) and per traffic pattern:

* **Top-1 agreement** with the teacher winner — overall, on non-trivial decisions (|E| ≥ 2), and on **hard** decisions.
* **Regret**: `teacher_cost[chosen] − teacher_cost[winner]` (mean, median, p95, and *relative* regret). This matters more than agreement — choosing the 2nd-best car at a near-tie is harmless.
* **Rank correlation** (Kendall τ between `score` and `teacher_cost` over E).
* **Calibration**: mean absolute error of `expm1(score)` vs teacher cost (bucketed by teacher cost decile).
* **Baselines on the same decisions** (from Phase 3): nearest-by-distance, lowest-ETA, `cost_greedy`.
* Confusion analysis: which situations does it get wrong? Cluster the errors by (pattern, load level, door state, call_on_route) and print the top-10 failure modes with counts. Include 5 concrete mistaken decisions rendered as text ("car 2 at floor 7 ↑ load 6/10, plan ends 12; call 9↑…").

Output: `docs/lift/offline_eval.md` (tables) + `reports/lift/offline_*.png` (agreement vs fleet size; regret CDF; calibration plot).

**Gate:** non-trivial val agreement ≥ 85 % (mean over 3 seeds). If you cannot reach it, do not move the goalpost: diagnose (data quantity? features missing? teacher noisy at ties? — measure the teacher's own *self-consistency* by re-labelling a sample with a different seed, which bounds the achievable accuracy), fix what is fixable, and report the residual honestly.

---

## 5. ONNX export and runtime — `learning/lift/export.py`, `learning/lift/runtime.py`

* `elevator lift export --ckpt runs/.../best.pt --out models/liftzero_bc_v1.onnx` writes the ONNX graph (inputs `call, cars, glob, mask, eligible`; outputs `score, aux, value, attn`; dynamic axes `batch`, `n_cars`) **and** `models/liftzero_bc_v1.json` — the **model card**: name, version, feature version + schema hash, parameter count, training data hash/splits, metrics (offline + closed-loop once available), git sha, date, intended use, limitations.
* `LiftRuntime(path)` (in `runtime.py`, numpy + onnxruntime only): `score(encoded) -> NetOutNp`; validates `FEATURE_VERSION`/schema hash against the model card and **refuses to load a mismatching model** with a clear error. Thread-safe, single `InferenceSession` with `intra_op_num_threads=1` (the sim is not GPU-bound and 1 thread is fastest for tiny graphs), sessions created lazily, graph optimisation level ALL.
* **Parity test**: torch vs ONNX max-abs difference < 1e-4 on 1 000 random real decisions and on edge shapes (N=1, N=32).
* **Latency benchmark** (`elevator lift bench-infer`): median / p95 / p99 ms for a single decision at N=2, 4, 8, 16, 32 — **Acceptance: median ≤ 2 ms at N=8, ≤ 5 ms at N=32** on the user's machine; report the CPU model.
* **Commit the ONNX file and card** (they're small); `.gitignore` must not exclude `models/`.

---

## 6. Strategy and cars: how the learned bidder lives inside the multi-agent system

**Design rule (Phase 1):** agents never call each other's methods; cars answer CFPs from their inbox using the public status board. LiftZero respects that: *each car runs the (shared) network on the public view and bids with its own output*.

### 6.1 Strategy registry change (additive)

Extend `DispatchStrategy` with `bidder: str = "classical"` (`"classical"` or `"learned"`) and `model_path: str | None = None`. Register:

```python
DispatchStrategy(name="liftzero_bc", label="LiftZero (imitation)", assignment="cnp", routing="astar",
                 reassignment="simulated_annealing", parking_policy="hill_climb", adapts_weights=True,
                 bidder="learned", description="Contract Net where each car's bid is computed by the LiftZero network "
                 "(trained by imitating the A* bidder). Everything else equals the 'full' strategy.")
```

Keeping everything else identical to `full` makes `liftzero_bc` vs `full` a clean, single-variable comparison (**the bidder is the only thing that differs**). Also register `liftzero_bc_cnp` (same as `cnp_astar` except the bidder: no SA / parking / adaptive weights) so the effect can be isolated from the extras.

### 6.2 Where the code goes

In `ElevatorAgent.compute_bid()`: if `strategy.bidder == "learned"` and the car is *eligible* (not refused by `_refusal()`), delegate to `self.model.learned_bidder.bid(self, call, urgency, waiting)`; otherwise unchanged. `LearnedBidder` (new, `learning/lift/bidder.py`):

* Builds the `DecisionContext` **from the public board only**: `model.board.cars()` (CarStatus), `model.board.policy`, the CFP's `{call, urgency, waiting}`, static building config. It must not read any private attribute of any car, passenger or floor object (a test enforces this by passing it a board-only stub).
* **Inference mode** `per_car_inference` (config; default `"shared"`): `"shared"` = one forward pass per CFP, cached by `conversation_id`, every car reads *its own* score (the weights are shared; mathematically identical to each car running the net itself because the net sees only public data); `"per_car"` = each car runs its own forward pass (faithful to "decentralised execution", ~N× slower). A test proves both give identical bids.
* Returns `Bid(car_id, total=expm1(score) clipped to [0, 1e5], wait/ride/crowding/energy = expm1(aux), eta=<from CarStatus plan_end_eta + distance heuristic, used only for display>, refused=False)`.
* **Shadow teacher**: when `model.config.shadow_teacher` is true, the car *also* computes its classical bid (not sent as a bid — it is sent in a debug field `shadow` of the PROPOSE content and stored on the `DecisionEvent` as `shadow_bids`). Cost: classical planning per CFP; used for (a) DAgger labelling, (b) UI agreement display. Default off.
* **Fallback**: if the model cannot be loaded, the strategy is reported as unavailable (`/api/meta` marks `available: false, reason`), *not* silently replaced by the classical bidder.

### 6.3 Decision trace

`explain_award` already writes a one-line reason. For learned bids append: `"[LiftZero] net scores: car 2=3.1, car 1=7.9; teacher agrees"` when shadow is on, else without the teacher part.

### 6.4 Decision hook additions (additive)

`DecisionEvent` gains optional `shadow_bids: list[Bid] | None` and `bidder: str`. Phase 3's behaviour-neutrality test must still pass; add a second neutrality test: strategies `nearest_car, collective, cnp_astar, full` produce identical metrics before/after this phase's engine changes (golden file of `Metrics.as_dict()` for 4 strategies × 3 scenarios × 2 seeds, generated **before** you touch the engine and committed as `tests/golden/strategy_metrics.json`).

---

## 7. Closed-loop evaluation and DAgger

### 7.1 Closed-loop protocol

`elevator lift eval-sim --strategy liftzero_bc --regimes all --seeds val|test --n 100` runs the **real** Mesa simulator (900 ticks) for each of the four Phase 3 regimes (`up_peak, down_peak, two_way, interfloor`) plus the 9 scenario YAMLs, for strategies `{nearest_car, collective, cnp_astar, full, liftzero_bc, liftzero_bc_cnp}`, using `evaluate()` from Phase 3 (paired seeds, bootstrap CIs). Multiprocessing across seeds. Report: avg wait, p95, max wait, long-wait %, throughput, energy, `compute_ms_per_tick`, safety violations (must be 0 — `model.violations()` is checked every tick), fraction of decisions where LiftZero's winner ≠ shadow teacher's winner.

### 7.2 DAgger (required, 2 rounds)

Behaviour cloning suffers from compounding error: the learned policy visits states the teacher never did. Fix:

For round r = 1, 2: roll out `liftzero_bc` (current model) in the real simulator over **N_r = 1 500 fresh training-split runs** drawn from the training distribution with `shadow_teacher=True`; record every decision with the **teacher's** costs/winner (labels) but the **learner's** state distribution; mix with a β-schedule (β_1 = 0.5, β_2 = 0.25: with probability β the *executed* award is the teacher's, otherwise the learner's, to keep exploration safe); aggregate into the dataset (new shards tagged `source="dagger_r{r}"`, same splits rule: **by run seed, never leaking val/test**); retrain from the previous checkpoint for ≤ 10 epochs with a lower LR (3e-4). Report per round: offline agreement on the *DAgger-state* validation set (states induced by the learner) **and** closed-loop metrics. Ship the best round by **validation closed-loop avg wait**, not by offline agreement.

### 7.3 Neutrality & safety

All existing tests stay green, golden metrics unchanged for the 4 classical strategies, and a new test runs `liftzero_bc` on **all 9 scenarios × 3 seeds** asserting `model.violations() == []` every tick, every passenger delivered after a drain, no REFUSE-bypass (an ineligible car never wins), and no exceptions on `car_fault`, `fire_alarm`, `rush` events.

---

## 8. CLI (new `lift` command group on the existing Typer app)

```
elevator lift train-bc      --config configs/learning/bc.yaml [--preset smoke|default|large] [--seed N] [--out runs/]
elevator lift eval-offline  --ckpt PATH [--split val|test|test_large]
elevator lift export        --ckpt PATH --out models/liftzero_bc_v1.onnx
elevator lift bench-infer   --model models/liftzero_bc_v1.onnx
elevator lift eval-sim      --strategy liftzero_bc [--regimes all] [--seeds val] [--n 100] [--workers N]
elevator lift dagger        --round 1 --ckpt PATH [--runs 1500] [--beta 0.5] [--workers N]
elevator lift card          --model models/liftzero_bc_v1.onnx          # prints the model card
```

Every command has `--help` with examples and a one-line description. A `scripts/reproduce_phase4.sh` runs the **smoke** pipeline end to end (record 20 k decisions → train smoke → export → closed-loop on 5 seeds) in ≤ 15 minutes.

---

## 9. Testing (graded)

New tests under `tests/lift/` (≥ 60), marked `slow` where > 10 s; the model/trainer tests `importorskip("torch")`, the runtime tests need only onnxruntime:

* Model: equivariance, padding invariance, size sweep, parameter count, determinism, export-ability, gradient flow to every parameter (no dead heads except `value`).
* Losses: hand-computed examples for each term; masked/ineligible entries contribute exactly 0; hard-decision weighting; trivial-decision filtering.
* Augmentation: car dropout never drops the winner; labels remain valid after permutation; noise not applied to one-hots.
* Overfit sanity: the smoke preset overfits 512 decisions to ≥ 99 % agreement in < 60 s (proves the pipeline can learn).
* Training determinism: same seed ⇒ identical loss for the first 20 steps.
* Offline metrics: regret/agreement/Kendall τ on synthetic cases with known answers.
* ONNX: parity, dynamic shapes, schema-hash mismatch refuses to load, missing file ⇒ strategy `available:false`.
* Bidder: board-only stub test (no private attribute access — use a stub whose attribute access raises), `shared` vs `per_car` equality, refused cars still answer REFUSE classically, shadow teacher fields present only when enabled.
* Strategy: registry entries, `/api/meta` exposes `bidder` and availability (additive, don't break the existing API tests), WebSocket frames remain strict JSON.
* Integration: the 9-scenario safety matrix of §7.3; golden-metrics regression for the 4 classical strategies.
* DAgger: aggregated dataset has `source` tags, no val/test seed leaks into train, β-mixing statistics (fraction teacher-executed ≈ β ± 0.05 over 2 000 decisions).

Coverage of `elevator_mas.learning.lift` ≥ 85 %.

---

## 10. Documentation

* `docs/lift/PHASE4_IMITATION.md`: architecture figure (draw it: tokens → encoder → heads), why a set-Transformer (variable fleet, permutation symmetry, scalability), loss derivations, data & splits, results tables (offline + closed-loop, mean ± CI), DAgger rounds table, error analysis, the self-consistency bound on achievable accuracy, limitations (e.g. routing is still classical A*, eligibility rules classical, trained on one building family), reproduction commands.
* Model card JSON (§5) and `docs/lift/MODEL_CARD.md` rendered from it.
* Update `docs/DESIGN.md` (new §: "Learned bidding": where the network sits in the PEAS/agent-type picture — a *learning agent* with performance element = the network, critic = teacher/imitation loss, learning element = training pipeline, problem generator = domain randomisation; answer "which AIMA agent type is this?" explicitly), `docs/TESTING.md` (new test layers + counts), `README.md` (quick start for `liftzero_bc`).
* Do **not** touch the React UI in this phase (Phase 6 does); only make sure `/api/meta` lists the new strategies.

---

## 11. Pitfalls to avoid (learned the hard way on this codebase)

1. `Bid.total` is `inf` for refused cars; JSON must stay strict (`null`). Never put `inf` in `.npz` labels — use the `1e6` sentinel + masks.
2. Calls are auctioned **one at a time**, so later rounds in the same tick see earlier awards on the board (`assigned_calls` changes). The board snapshot used for the features must be the one at *announce* time of that round — Phase 3's `DecisionEvent` provides it; the live bidder reads the board when the CFP arrives (same moment). Test equality of the two paths.
3. The teacher adds an urgency discount (`escalation_bonus`) via `urgency`; the call token has `call_urgency` so the network can learn it. Don't re-implement the discount in Python.
4. Do not let the learned model decide *eligibility*; a full car must still answer `REFUSE reason="full"`.
5. Teacher ties are broken by lowest car id: your argmin tie-break must match, or agreement is under-reported.
6. A model trained on 15×4 buildings must be sanity-checked on 40×8 (`test_large`) — features are normalised by floors/capacity, but check `plan_end_eta`'s 240 s clip on tall buildings.
7. Don't evaluate on training seeds. Splits are by **run seed**; the closed-loop `val` seeds are 8000–8049 and `test` 8500–8599 (Phase 3).

---

## 12. Acceptance checklist (evidence in the final report)

- [ ] Existing + new tests green (`uv run pytest`, with and without the `lift-train` group where applicable); ruff clean; coverage ≥ 85 % for `learning.lift`.
- [ ] Model: 0.2–0.4 M params; equivariance & padding invariance proven by tests.
- [ ] Offline: **≥ 85 % non-trivial val agreement** (mean of 3 seeds) — or documented diagnosis with the measured teacher self-consistency bound; hard-decision accuracy, regret and `test_large` numbers reported.
- [ ] ONNX committed with model card; torch↔ONNX parity < 1e-4; median latency ≤ 2 ms (N=8), ≤ 5 ms (N=32).
- [ ] `liftzero_bc` and `liftzero_bc_cnp` registered; the learned bidder reads only public data (stub test); `shared`≡`per_car`.
- [ ] Closed-loop on the real sim, 100 paired test seeds per regime: report avg wait/p95/long-wait/throughput/energy vs `full` and `cnp_astar` with CIs. **Target:** avg wait within ±5 % of the teacher in all four regimes, p95 within ±10 %. If a regime misses, explain with evidence.
- [ ] Two DAgger rounds done and compared; the shipped model is the best by validation closed-loop wait.
- [ ] Safety matrix (9 scenarios × 3 seeds) zero violations; golden metrics for the four classical strategies unchanged.
- [ ] Docs + model card + README/DESIGN/TESTING updated; `scripts/reproduce_phase4.sh` runs.
- [ ] Branch `phase-4-imitation`, logical commits, not pushed.

## 13. Final report format

Per-module summary; training curves and tables (3 seeds); offline and closed-loop result tables with CIs;
DAgger table; latency table; test counts/coverage; deviations from this prompt and why; known limitations;
the exact verification commands, ending with:

```
git checkout phase-4-imitation
uv sync --group learn --group lift-train
uv run pytest
uv run elevator lift card --model models/liftzero_bc_v1.onnx
uv run elevator lift bench-infer --model models/liftzero_bc_v1.onnx
uv run elevator lift eval-sim --strategy liftzero_bc --regimes all --seeds val --n 20
```
