# PHASE 5 — Cooperative reinforcement learning: PPO on the twin, KL-anchored to the teacher, validated on the real simulator

> **Prerequisites:** Phase 3 (twin, env, regimes, evaluation harness) and Phase 4 (LiftZero
> network, ONNX runtime, `liftzero_bc` strategy, closed-loop evaluation, model card) are merged
> into `v2`. The user will give you the base commit. Names follow the Phase 3/4 prompts; **where
> the merged code differs, the code wins** — adapt and list adaptations in the final report.

Paste this whole file into Antigravity (Planning mode, strongest model).

---

## 0. Context and goal

The imitation model (`liftzero_bc`) can at best *match* the classical A* bidder. Reinforcement
learning can do something imitation cannot: **optimise the true objective** (passenger waiting
time, fairness, energy) directly, and so discover assignments the heuristic teacher misses —
for example in **down-peak**, where the current best strategy (`full`) is known to *lose* to the
simple `collective` baseline.

You will train the same `LiftZeroNet` with **cooperative PPO** on the fast twin (Phase 3), warm-
started from the Phase 4 checkpoint and **anchored to it by a KL penalty** so it improves
without forgetting how to be sane. You will then validate on the **real** simulator — because
the twin is only a stand-in — and, if a sim-to-real gap shows, close it with a short real-sim
fine-tune. The result is the shipped model `liftzero_ppo`.

The honest research question: **Does RL beat the teacher in at least 2 of the 4 regimes
(`up_peak, down_peak, two_way, interfloor`) on the real simulator, with statistically
significant paired improvements and without hurting fairness (p95 / long waits)?**
A clean *no* with a good analysis is an acceptable outcome; a *yes* obtained by cheating
(training on test seeds, hiding starvation, evaluating only on the twin) is not.

---

## 1. Working agreement

* Branch `phase-5-ppo` from the updated `v2`. Small commits. **Do not push; do not touch `main`.**
* `uv`, Python 3.12, ruff clean (line length 100). Existing + Phase 3/4 tests must stay green.
* Training deps use the existing `lift-train` group (torch, onnx). No new heavy dependencies
  (no stable-baselines3, no RLlib, no Ray): **implement PPO yourself** (~400 lines) — the point of
  the project is understanding and explaining the algorithm in the viva. Use `numpy` + `torch`.
* Seeds everywhere; a training run writes `run.json` (git sha, config, seeds, wall time, torch version).
* Never read `val` (8000–8049) or `test` (8500–8599) seeds during training. Model selection uses `val`
  only; the **test split is touched exactly once**, for the final table (a test enforces that the
  training code cannot request test seeds: `assert_no_test_seed()` guard in the env factory).
* Budget presets: `smoke` (≈ 5 min, 0.5 M decisions), `default` (≈ 2–3 h on 8 CPU cores, 30 M decisions),
  `long` (overnight, 100 M). Training must be **resumable** from checkpoints (optimizer + RNG + step).

---

## 2. Formulation (write this precisely in the docs — it is a viva question)

### 2.1 Semi-Markov decision process

One agent **decision** = one hall call to assign. Between decisions an irregular number of ticks
`Δt ≥ 0` elapse (several calls can fall in one tick: `Δt = 0`). The environment is the Phase 3
`VectorDecisionEnv` (observation = `Encoded`, action = car index, masked).

* **State** (what the policy sees): the Phase 3 feature tokens (public fleet status, call, global).
* **Action**: choose car `a ∈ E` (eligible). Under the policy `π(a|s) = softmax_{i∈E}(−score_i · exp(−logit_temp))`.
* **Reward**: the Phase 3 team reward (identical for all cars → *cooperative*, fully shared):
  `r = −(Σ_waiting Δt + 0.5·Σ_riding Δt)/100 − 0.02·floors_moved − 0.5·new_threshold_crossings`.
* **Discount for a SMDP**: `γ_eff = γ^(Δt/τ)` with `γ = 0.99`, `τ = 5` ticks — time-aware, so a decision followed by a
  long gap is discounted more. GAE uses `γ_eff` per step: `δ_t = r_t + γ_eff,t · V(s_{t+1}) − V(s_t)`; `A_t = Σ (γ_eff λ)…` with `λ = 0.95`.
* **Horizon**: fixed-length episodes (600–1800 ticks, the training distribution); `truncated=True` at the end →
  **bootstrap** the final value (never treat the horizon as terminal).

### 2.2 Why this is multi-agent (and honest about it)

Centralised training with decentralised execution (CTDE):
* **Decentralised execution**: each car computes its own score from the public board with the *shared* network.
* **Shared team reward** ⇒ cooperative game; no car is rewarded for stealing calls.
* **Asymmetric centralised critic** (the CTDE part that is more than rhetoric): the value head additionally
  receives **privileged information the actors never see** — per-floor waiting counts, the age of the oldest
  waiter per floor/direction, and the number of people riding to each floor. The twin exposes this as
  `info["privileged"]` (add to the Phase 3 env, additive; vector of fixed length `KPRIV = 2*MAX_FLOORS + 2*MAX_FLOORS + MAX_FLOORS` with `MAX_FLOORS = 48`, normalised, zero-padded). The actor
  never receives it (test: gradient of `score` w.r.t. `privileged` is exactly zero because it is not an input; and ONNX export contains no such input).
  Implementation: `ValueHeadPriv(concat(pooled_tokens, MLP(privileged)) → 1)` used **only in training**; the exported actor (and the
  Phase 4 `value` head used by MCTS in Phase 6) is the non-privileged head — so train **two** value heads: `V_priv` (critic for GAE, advantage
  estimation) and `V_pub` (distilled from `V_priv` by MSE, weight 0.5; this is what MCTS will use because it needs only public information).

### 2.3 Algorithm: PPO with a teacher anchor

For each update:
1. Collect `T = 128` decisions from each of `B = 64` parallel envs (8 192 samples) with the current stochastic policy.
2. Compute GAE advantages with `V_priv`; normalise advantages per mini-batch.
3. For `K = 4` epochs, shuffle, mini-batches of 2 048 decisions, optimise
   ```
   L = −L_clip(ε=0.2)  +  c_v · Huber(V_priv − R)  +  c_d · MSE(V_pub − sg(V_priv))
       −  c_e · H[π]   +  β · KL(π_BC ‖ π)
   ```
   with `c_v = 0.5`, `c_d = 0.5`, entropy `c_e: 0.01 → 0.001` (linear), and the **anchor** `β: 0.2 → 0.0` (linear over the first 60 % of training).
   `π_BC` is the frozen Phase 4 policy (same architecture, loaded once). The KL is computed over eligible cars per decision.
4. Adam (lr 3e-4 → 3e-5 linear), grad-clip 0.5, **value-function clipping off**, **early-stop the epoch loop** if mean approx-KL(old‖new) > 0.03.
5. Log every update: mean episodic return, policy loss, value loss, entropy, approx-KL, clip-fraction, explained variance, `β`, SPS (samples/s),
   and **the true evaluation metrics of the training episodes** (avg wait, p95, long-wait %) — the shaped reward is never used to judge success.

### 2.4 Initialisation & parametrisation details that matter

* Warm start: load Phase 4 weights (score head, encoder). `logit_temp` initialised from Phase 4's calibrated value. The new `V_priv` head and the `V_pub` distillation head start small (std 0.01 final layer).
* **Observation normalisation** is already built into the feature encoder — do not add running-mean normalisers (they would break the ONNX contract).
* **Reward scaling**: keep the fixed `/100` scale; log return scale; if the value loss is dominated by outliers, clip per-step reward to [−20, 0].
* **Action masking** via `−inf` logits on ineligible cars; decisions with `|E| = 1` are *skipped for the loss* (zero gradient signal) but still advance the environment and count for GAE.
* **Exploration**: the stochastic policy with temperature suffices; additionally with ε-teacher mixing at the start: with probability `ε_t` (0.2 → 0 over 20 % of training) execute the Phase 3 `cost_greedy` action — **but store the policy's own log-prob for the executed action only when it matches**; to avoid off-policy bias, simply *do not* include teacher-executed steps in the PPO loss (still use them for value targets). Document this choice.

---

## 3. Domain randomisation and curriculum

* Training distribution = `configs/learning/regimes.yaml: training:` (Phase 3): floors 6–40, cars 2–8, capacity 6–16, `seconds_per_floor` 1–3, 1–4 phases from the 5 patterns, rate = `u·cars·0.04`, `u ~ U(0.6, 1.8)`, faults/fire with probability 0.25.
* **Curriculum** (config): stage 1 (first 20 %): `u ∈ [0.6, 1.2]`, no disturbances; stage 2: full range; stage 3 (last 20 %): oversample the *hard* patterns (`down_peak`, `two_way`) 2×. Document that this is a scheduled, not adaptive, curriculum.
* Evaluation regimes (four fixed) are **never** sampled during training with their exact parameters + seed range; the training sampler must avoid seed collisions (guard test).

---

## 4. Implementation layout

```
src/elevator_mas/learning/lift/rl/
  ppo.py          # PPOTrainer: rollout buffer, GAE (SMDP), losses, update, logging, checkpoint/resume
  buffer.py       # RolloutBuffer (numpy, preallocated), minibatch iterator
  policy.py       # TorchPolicy wrapper: act(obs)->(action, logp, value), masked categorical, deterministic mode
  anchors.py      # frozen BC policy + masked KL
  curriculum.py
  eval_loop.py    # periodic twin evaluation on VAL seeds across the 4 regimes + checkpoint selection
  real_finetune.py# real-simulator rollout collector + short PPO fine-tune (stage C)
configs/learning/ppo_smoke.yaml, ppo_default.yaml, ppo_long.yaml
tests/lift/rl/...
```

CLI additions (group `lift`):

```
elevator lift train-ppo   --config configs/learning/ppo_default.yaml [--init runs/.../bc_best.pt] [--resume runs/.../last.pt] [--seed N]
elevator lift eval-twin   --ckpt PATH --split val|test --n 50
elevator lift finetune-real --ckpt PATH --decisions 1000000 --workers N
elevator lift export      --ckpt PATH --out models/liftzero_ppo_v1.onnx        # reuses Phase 4 export (+ value_pub output)
elevator lift eval-sim    --strategy liftzero_ppo --regimes all --seeds test --n 100
```

The exported ONNX gains the output `value` (= `V_pub`) already present in the Phase 4 contract; add
`strategy "liftzero_ppo"` (same as `liftzero_bc` but `model_path="models/liftzero_ppo_v1.onnx"`) and `liftzero_ppo_cnp`.

---

## 5. Training stages

**Stage A — PPO on the twin (the main run).** Presets above. Evaluate every 1 M decisions on the twin with the VAL seeds
(8000–8049, all four regimes, deterministic `argmin` policy), keep the top-3 checkpoints by *mean paired improvement over the `cost_greedy` baseline* with a **p95 guard**: a checkpoint is
ineligible if p95 or long-wait % is worse than the teacher-mimic by > 10 % in any regime. Train **3 seeds**; report mean ± std of the learning curves.

**Stage B — Real-simulator validation.** For the best checkpoint of each seed, export to ONNX and evaluate in the **real** simulator on VAL seeds (the Phase 4 `eval-sim` path). Pick the best seed by *real* validation performance (not twin).
Measure the **sim-to-real gap** per regime: `twin_improvement − real_improvement`. A scatter plot "twin vs real improvement per checkpoint" goes in the report.

**Stage C — Real-sim fine-tune (run only if the gap is material: real improvement < 50 % of twin improvement in ≥ 2 regimes, or any real regression).**
A rollout collector `collect_rollout_real(policy, n_runs)` runs the real `ElevatorModel` with the learned bidder and a **forced-winner** hook (the policy's action decides which car's bid is lowest: chosen car bid = 0, other eligible cars bid `1e3`; ineligible cars still REFUSE) while the `DecisionEvent` hook records `(obs, action, logp, value, reward)` — reward computed from the model's own per-passenger records between decisions with the same `RewardConfig` as the twin. Then run the same PPO update (β fixed at 0.05, lr 1e-4, 1–2 M decisions, 8 workers; this takes ~hours: run it as a background job with checkpoints). Compare before/after on VAL. Keep the fine-tuned model only if real VAL avg wait improves by ≥ 2 % and fairness guards hold.

**Stage D — Final test.** Evaluate **once** on TEST seeds (8500–8599) in the real simulator: strategies `{nearest_car, collective, cnp_astar, full, liftzero_bc, liftzero_ppo}` × the four regimes × 100 paired seeds, plus the 9 scenarios × 20 seeds and the scale regimes `small_6x2`, `large_40x8`.

---

## 6. What "success" means, and the statistics (be rigorous)

For every regime and metric (avg wait, p95, max wait, long-wait %, throughput, energy) compute the **paired** difference
`liftzero_ppo − full` (and `− cnp_astar`, `− collective`) over the 100 seeds: mean, 95 % bootstrap CI (10 000 resamples), Wilcoxon signed-rank
p-value, win-rate, Cohen's d_z. Apply **Holm–Bonferroni** across the 4 regimes for the avg-wait claim.

* **Primary claim** (pre-registered here): *`liftzero_ppo` has lower average wait than `full` with Holm-adjusted p < 0.05 in ≥ 2 of the 4 regimes, with p95 and long-wait % not worse by more than 10 %.*
* **Secondary**: beats `collective` in `down_peak` (the known weak spot); matches or beats `full` on throughput; energy within +10 %.
* Report the outcome whichever way it goes. If the primary claim fails: present the table, then analyse (learning curves plateau? sim-to-real gap? reward misalignment — did avg wait fall while p95 rose? Are improvements concentrated in some seeds/patterns?). A good negative result earns more credit than a doctored positive one.
* **Reward-hacking checks (required section in the docs):** look for policies that improve the shaped reward but not the metrics: (a) starvation (max wait ↑), (b) refusing/ignoring far floors, (c) exploiting the twin's simplifications (e.g. twin-specific door timing) — test by evaluating on the real sim, (d) over-concentrating calls on one car (fairness across cars: Gini of per-car served passengers, report it), (e) energy blow-up.

---

## 7. Analysis and explainability deliverables

* **Learning curves** (return, true avg wait per regime, p95, KL to BC, entropy) with 3-seed bands.
* **What did RL change?** Compare the PPO policy's decisions with the teacher's on 20 000 held-out real-sim decisions: overall disagreement rate; disagreement by pattern, load, door state, `call_on_route`; for the top-5 disagreement situations render a text vignette and the counterfactual outcome (re-run that decision from a saved twin state with each alternative and report the 60-tick cost) — "RL sends the *nearly empty car already heading the other way* instead of the *closest car with a long plan*".
* **Bid-landscape plots**: score of a car as a function of (distance to call, load, plan_end_eta) for BC vs PPO vs teacher (marginals with the rest fixed at typical values).
* **Feature-group occlusion** importance (zero out position / load / doors / plan / call-relation / policy-pattern groups on the actor input and measure Δ avg-wait in the twin) per regime — shows what the policy relies on and whether it uses the pattern token.
* **Training-cost table**: decisions, wall time, CPU model, samples/s.

---

## 8. Testing (graded)

New tests `tests/lift/rl/` (≥ 45), torch tests under `importorskip`:

* GAE: hand-computed 5-step example with irregular `Δt` (SMDP discounting) and a truncation bootstrap; equals a slow reference implementation on random data.
* PPO loss: clipping behaviour at ratio > 1+ε and < 1−ε with sign-correct advantages; masked categorical (ineligible mass = 0, `|E|=1` → entropy 0, skipped in loss); KL anchor (zero when policies equal; positive otherwise; correct direction `KL(π_BC‖π)`).
* Buffer: shapes, preallocation, truncated vs terminated handling.
* **Learning sanity (fast, deterministic):** on a tiny 2-car bandit-like twin config the PPO smoke run must prefer the obviously better car > 90 % of the time within 60 s; on the smoke preset the policy's true avg wait on the twin must be ≤ the random-eligible baseline's.
* Privileged-info isolation: actor output independent of `privileged`; exported ONNX has no such input.
* No-test-seed guard: training factory raises when asked for seeds ≥ 8000.
* Checkpoint resume reproduces the same next-update loss (bitwise on CPU) as an uninterrupted run.
* Selection logic: p95 guard rejects a checkpoint that improves avg wait but worsens p95 by 15 %.
* Forced-winner hook: chosen car wins every auction in the real sim; ineligible cars still REFUSE; metrics of an all-forced-teacher run equal the classical run (proves the hook is faithful).
* Real rollout collector: reward matches an independent recomputation from passenger records on a short run.
* Stats helpers: bootstrap CI coverage ≥ 90 % in 100 synthetic trials; Holm correction on known p-values; Wilcoxon against `scipy.stats.wilcoxon`.
* Regression: all earlier tests green; golden strategy metrics unchanged.

---

## 9. Documentation

* `docs/lift/PHASE5_PPO.md`: the SMDP formulation, CTDE with the asymmetric critic, the exact loss, hyper-parameters (table), curriculum, stages A–D, the results (twin and real), statistics, reward-hacking audit, fairness (per-car Gini), analysis of what RL learned, honest limitations (twin routing = LOOK vs real A*, public-information-only actor, finite training distribution), reproduction commands, compute budget.
* Update `docs/lift/MODEL_CARD.md` and the ONNX card JSON for `liftzero_ppo_v1`.
* Update `docs/TESTING.md` (counts) and `README.md`. Add `scripts/reproduce_phase5.sh` (smoke pipeline ≤ 20 min: record small → BC smoke → PPO smoke → real eval 5 seeds).
* Figures to `reports/lift/` (PNG at 200 dpi, consistent palette, readable in dark and light slides).

---

## 10. Pitfalls

1. **Truncation ≠ termination** — bootstrap `V(s_T)`; getting this wrong biases values near the horizon.
2. Several decisions in the same tick have `Δt = 0` ⇒ `γ_eff = 1`; the reward between them is 0 — fine, but don't divide by zero in any rate computation.
3. Vector envs with different fleet sizes: padded cars must be masked in the policy, the critic's pooling and the KL.
4. The `/100` reward scale and `RewardConfig` are part of the model card; changing them silently invalidates comparisons.
5. Advantage normalisation per mini-batch when `|batch|` small can blow up — use `(A − mean)/(std + 1e-8)` and skip normalisation for batches < 64.
6. Don't select checkpoints on the twin alone — the twin can be exploited; the real VAL run is the arbiter.
7. Don't forget deterministic evaluation (`argmin`) vs stochastic training behaviour; report both for the final model.
8. The real simulator is ~100× slower than the twin; Stage C must be parallel and checkpointed, and the user may stop it at any time: always keep `best.pt` current.

---

## 11. Acceptance checklist

- [ ] Own PPO implementation, tested (§8), CTDE asymmetric critic isolated from the actor; ONNX unchanged contract.
- [ ] 3 training seeds finished (default preset or documented smaller budget with reason); curves with bands.
- [ ] Twin and **real** VAL evaluation; sim-to-real gap analysed; Stage C executed or its skip justified by the stated criterion.
- [ ] Final TEST evaluation executed exactly once; table with paired stats (CI, Wilcoxon, Holm, win-rate, d_z) for all 4 regimes vs `full`, `cnp_astar`, `collective`.
- [ ] Primary claim stated pre-hoc and reported pass/fail honestly; fairness (p95, long waits, Gini) and energy reported; reward-hacking audit written.
- [ ] `models/liftzero_ppo_v1.onnx` + card committed; latency unchanged (≤ 2 ms at N=8).
- [ ] Safety matrix (9 scenarios × 3 seeds) zero violations for `liftzero_ppo`.
- [ ] Docs, README, TESTING updated; `scripts/reproduce_phase5.sh` runs; ruff clean; all tests green.
- [ ] Branch `phase-5-ppo`, logical commits, not pushed.

## 12. Final report format

Tables first (results with CIs, then curves), then what was built, test counts, compute used, deviations and why, limitations,
verification commands ending with:

```
git checkout phase-5-ppo
uv sync --group learn --group lift-train
uv run pytest
uv run elevator lift card --model models/liftzero_ppo_v1.onnx
uv run elevator lift eval-sim --strategy liftzero_ppo --regimes all --seeds val --n 20
```
