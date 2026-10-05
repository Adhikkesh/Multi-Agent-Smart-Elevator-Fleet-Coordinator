# Phase 4 — LiftZero: a learned Contract Net bidder by imitation (BC + DAgger)

> Owner: **Kavin Karthic (CB.SC.U4CSE23161)** — Simulation Environments & Imitation Learning.
> Code: `src/elevator_mas/learning/lift/`. Tests: `tests/lift/`. Model: `models/liftzero_bc_v1.onnx`.

## 1. What changed in the system

In Phases 1–3 every car answers a Contract Net call-for-proposals (CFP) with a *bid*: the
marginal cost of inserting the call into its own A\*-planned route
(`ElevatorAgent.marginal_cost`). The dispatcher awards the call to the lowest bid.

Phase 4 replaces **only the bid computation** with a neural network — everything else (the
CFP/PROPOSE/ACCEPT protocol, eligibility, A\* routing, SA reassignment, parking, safety rules)
is unchanged:

```
                CFP(call, urgency, waiting)
 Dispatcher ───────────────────────────────▶ every car
                                              │
                         eligible? (classical _refusal: out of service / fire / full)
                           │no                         │yes
                     REFUSE(reason)        LiftZero net on PUBLIC data only
                                           (status board + fleet policy + CFP + building)
                                                       │
                                           PROPOSE(bid = expm1(score_i), breakdown = expm1(aux_i))
 Dispatcher ◀──────────────────────────────────────────┘   [+ shadow A* bid if enabled]
   award = argmin bid (ties → lowest car id) → ACCEPT / REJECT
```

Two strategies are registered, each differing from a classical strategy **in the bidder only**:

| strategy | equals | difference |
| --- | --- | --- |
| `liftzero_bc` | `full` (CNP + A\* + SA + smart parking + adaptive weights) | bids from the network |
| `liftzero_bc_cnp` | `cnp_astar` (CNP + A\*) | bids from the network |

### AIMA view

The LiftZero bidder is a **learning agent** (AIMA §2.4.6) inside each car:

* *performance element* — the network that prices a call from the public board;
* *critic* — the A\* teacher's bid, compared with the network's through the imitation loss;
* *learning element* — the behaviour-cloning trainer and the DAgger loop;
* *problem generator* — domain randomisation (random buildings, traffic phases, faults, fire)
  and, in DAgger, the learner's own rollouts, which visit states the teacher never would.

The car itself remains a *utility-based agent* (it bids an estimated cost); what changed is
that the utility estimate is now learned rather than searched for.

## 2. The network — `LiftZeroNet` (`model.py`)

```
 glob [Kg=10] ─ MLP ─┐
 call [Kc=14] ─ MLP ─┤ + type embedding (global / call / car)       no positional encoding,
 car_i [Kcar=26] ─ MLP ─┘                                             no car-id embedding
        │
   [glob, call, car_1 … car_N]  ── 4 × pre-LN Transformer encoder (d=64, 4 heads, FFN 256)
        │                           padded cars removed by an additive −1e9 key bias
        ├─ car tokens ─ score head (d→d→1)   → score_i  (log-cost; lower is better)
        ├─ car tokens ─ aux head   (d→d→4)   → log1p(wait, ride, crowding, energy)
        ├─ [glob ; call ; mean(cars)] ─ value head → V(s)   (Phase 5/6; untrained here)
        └─ last-layer attention of the call token over the cars → attn_i (explainability)
```

* **237,575 parameters** (budget 0.2–0.4 M).
* **Permutation-equivariant**: permuting the cars permutes every per-car output and leaves
  the value unchanged (tested to 1e-5). One network therefore serves fleets of any size.
* **Padding-invariant**: adding padded cars never changes the real cars' outputs (tested).
* Ineligible or padded cars receive `score + 1e4`, so `argmin(score)` is always a legal car.
* Policy semantics: deterministic `argmin_i score_i` (ties → lowest id); stochastic
  `π(i) = softmax_i(−score_i · e^{−τ})` with a learnable log-temperature `τ` (used by Phase 5).

**Why a set-Transformer.** A fleet is a *set* of cars: there is no natural order, and the
size varies (2–8 in training, up to 16 in the schema, 32 in the latency benchmark). An MLP
over concatenated cars would tie each weight to a car slot, learn spurious slot preferences,
and fail on a fleet size it has not seen. Self-attention lets every car's price depend on
every other car (is someone else already passing that floor?) while staying symmetric.

## 3. Features (schema v2)

The Phase 3 schema (14 call, 26 car, 6 global features) was extended in Phase 4 to **v2**:
the global token now carries the four public cost weights `W1–W4` from the fleet policy on
the status board. The `full` teacher retunes those weights per traffic pattern while
`cnp_astar` keeps the defaults; under v1, identical inputs could carry different teacher
labels, putting a ceiling on achievable agreement. Weights are public (they are on the
board), so the learned bidder may read them.

## 4. Data

`elevator learn record` harvests teacher decisions from the real simulator over randomised
buildings (6–32 floors, 2–8 cars, capacity 6–16, 1–3 s/floor, 1–4 traffic phases from 5
patterns, faults/fire in 25 % of runs), half with `full` and half with `cnp_astar` as the
teacher. Splits are **by run seed**, never by decision:

| split | seeds | role |
| --- | --- | --- |
| train | 0–2258 (expert), 2500–6999 (DAgger) | fitting |
| val | 8000–8247 (expert), 8260–8499 (DAgger-state) | checkpoint selection |
| test | 8500+ | reported once |
| test_large | test seeds with 33–40 floors × 7–8 cars | fleet-size generalisation |

Closed-loop evaluation uses the four fixed Phase 3 regimes (15 floors × 4 cars) with val
seeds 8000–8049 and test seeds 8500–8599 — different buildings from the harvest.

**Observation.** About half of the raw decisions carry no label: in saturated buildings
(every car full) or during a fire recall every car refuses, and the dispatcher re-auctions
each waiting call every tick. Those all-refused rounds are kept in the shards (they are real
system behaviour) but skipped for training; decisions with one eligible car are kept at ≤ 5 %.

## 5. Imitation losses (`losses.py`)

For one decision with eligible cars `E`, teacher bids `c_i`, `y_i = log1p(c_i)`, scores `s_i`:

1. **Listwise cross-entropy** `L_list = −Σ_i t_i log softmax_E(−s_i e^{−τ})`, with `t` one-hot
   on the teacher's winner — or uniform over cars that **tie** for the best bid (the dispatcher
   breaks ties by car id, which an equivariant network cannot and should not see).
2. **Regression** `L_reg = Huber_1(s_i − y_i)` — calibrates scores as log-costs (needed by
   MCTS in Phase 6 and by the UI).
3. **Centred rank** `L_rank = Huber_0.5((s_i − mean_E s) − (y_i − mean_E y))` — learns the
   ordering independent of each decision's cost level.
4. **Auxiliary breakdown** `L_aux = Huber_1(aux_ik − log1p(teacher_k_i))`, k ∈ {wait, ride,
   crowding, energy}.
5. **Pairwise hinge** `L_pair = max(0, 0.1 − (s_runner_up − s_winner))` against the best
   strictly-worse car, weighted ×2 on *hard* decisions (runner-up within 10 % of the best).

`L = 1.0 L_list + 0.5 L_reg + 0.5 L_rank + 0.2 L_aux + 0.3 L_pair`. Padded and ineligible cars
contribute exactly zero (tested for every term).

**Augmentation** (per batch, in torch): random per-decision car permutation; car dropout
(p = 0.3, 1–3 *non-winning* cars, ≥ 2 eligible remain — labels stay exact because a car's
marginal cost does not depend on the other cars); Gaussian noise σ = 0.01 on continuous
features only. *Mirror augmentation* (flip floors, swap up/down) is deliberately not used:
the lobby is at the bottom and up-peak is not the mirror image of down-peak.

**Optimisation.** AdamW (lr 1e-3, wd 1e-2, β = 0.9/0.99), 500-step linear warm-up then cosine
to 1e-5, batch 512, gradient clip 1.0, ≤ 30 epochs, early stopping on validation non-trivial
agreement (patience 4). Three seeds (0, 1, 2); the shipped model is the best seed.

## 6. Training results (behaviour cloning, 3 seeds)

594,950 training decisions after filtering (from 1.2 M recorded), validation on 18 k
non-trivial expert decisions, Apple M2 (MPS). Curves: `docs/lift/runs/*/curves.png`.

| seed | best epoch | val agreement | tie-aware | hard | regret | Kendall τ | wall time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | **88.60 %** | 90.62 % | 64.47 % | 2.77 | 0.81 | 19 min |
| 1 | 20 | 88.42 % | 90.56 % | 63.93 % | 2.56 | 0.81 | 46 min\* |
| 2 | 13 | 87.89 % | 89.91 % | 63.49 % | 2.32 | 0.81 | 25 min |
| **mean ± std** | | **88.30 ± 0.37 %** | **90.36 ± 0.39 %** | 63.96 % | | | |

\*seed 1 shared the GPU with a DAgger fine-tune. **Gate: ≥ 85 % non-trivial validation
agreement (mean of 3 seeds) — passed.** Seed 0 was the best and seeded DAgger.

## 7. DAgger (two rounds)

| round | rollouts (β) | new decisions | teacher-executed | offline agree on DAgger-state val (tie-aware) | closed-loop VAL: worst / mean Δ avg wait vs teacher (8 comparisons) | targets met |
| --- | --- | --- | --- | --- | --- | --- |
| r0 = BC seed 0 | — | — | — | 88.67 % (90.72 %) | +6.2 % / +0.1 % | 7 / 8 |
| r1 | 1 500 (0.5) | 1,052,499 | 0.499 | 88.68 % (90.84 %) | +2.9 % / −0.2 % | 8 / 8 |
| **r2 (shipped)** | 1 500 (0.25) | 941,586 | 0.250 | — | **+1.1 % / −1.5 %** | **8 / 8** |

Validation: 50 paired val seeds × 4 regimes, `liftzero_bc` vs `full` and `liftzero_bc_cnp` vs
`cnp_astar` (`reports/lift/selection/val_r*.md`). The shipped model is chosen by
**validation closed-loop average wait**, not offline agreement.

**What DAgger did — honestly.** The BC policy was already almost as accurate on the states it
induces itself (88.67 %) as on expert states (88.50 %): covariate shift was small here,
because the learner only chooses *which* car takes a call while routing, reassignment and
parking stay classical and keep pulling the system back to familiar states. So DAgger
barely moved offline agreement. It did move the closed loop: the one regime that missed the
±5 % band under BC (`two_way`, +6.2 %) moved to +1.1 %, and the mean deviation turned
slightly negative. β-mixing was exact (0.499 and 0.250 against β = 0.5 and 0.25).

## 8. Offline results of the shipped model (`docs/lift/offline_eval.md`)

| split | non-trivial decisions | agree | tie-aware | hard | regret mean | Kendall τ |
| --- | --- | --- | --- | --- | --- | --- |
| val | 29,516 | 88.63 % | 90.88 % | 62.74 % | 2.11 | 0.83 |
| test | 24,042 | **89.90 %** | **91.69 %** | 68.08 % | 1.77 | 0.84 |
| test_large (33–40 floors × 7–8 cars, never trained on) | 20,146 | 83.74 % | 87.22 % | 58.02 % | 9.05 | 0.79 |

Baselines on the same val decisions: nearest car 49.1 %, lowest ETA 32.3 %, Phase 3
`cost_greedy` 54.6 % (regret 19.7 vs LiftZero's 2.1). Agreement falls smoothly with fleet
size (2 cars 94.5 %, 4 cars 89.5 %, 6 cars 85.6 % on val) — more cars, more near-ties.

**Where the residual error comes from.** No two non-trivial decisions share a feature vector,
and 1-decimal near-duplicates conflict for < 0.01 % — so the error is not label noise
between identical inputs. It is information the public features summarise away: the teacher
plans with every queued stop of a car, while the token carries counts and the plan's end;
the teacher multiplies the wait term by the true number of waiting passengers and their
priority weights, while `call_waiting` is clipped at 20 and priority weights are not on the
board. The largest-regret mistakes are exactly crowded lobby calls in 28–32-floor buildings.
On `test_large`, `plan_end_eta` saturates at its 240 s clip in tall buildings — the expected
pitfall — and regret grows accordingly, while agreement stays at 84–87 %.

## 9. Closed-loop TEST results (`docs/lift/closed_loop.md`)

Real Mesa simulator, test seeds 8500–8599 (100 paired seeds per regime), evaluated once
after selection. 7,800 runs. **Safety violations: 0.**

Average wait (s), the four Phase 3 regimes:

| regime | nearest_car | collective | cnp_astar | full | **liftzero_bc** | **liftzero_bc_cnp** |
| --- | --- | --- | --- | --- | --- | --- |
| up_peak | 56.2 | 47.1 | 42.9 | 29.9 | 31.6 | 43.4 |
| down_peak | 22.4 | 21.1 | 22.0 | 23.7 | 24.0 | 21.9 |
| two_way | 18.7 | 17.4 | 14.5 | 14.5 | 14.9 | 14.7 |
| interfloor | 12.5 | 11.6 | 11.8 | 11.0 | 11.1 | 11.6 |

Learned vs its teacher (paired, Δ % with 95 % bootstrap CI; target |Δ avg wait| ≤ 5 %, p95 ≤ 10 %):

| regime | `liftzero_bc` vs `full`: avg / p95 | `liftzero_bc_cnp` vs `cnp_astar`: avg / p95 |
| --- | --- | --- |
| up_peak | +5.8 [−3.7, +17.1] / +2.3 | +1.1 [−4.0, +6.4] / +1.1 |
| down_peak | +1.4 [−2.0, +5.0] / −0.4 | −0.4 [−2.6, +1.9] / +0.1 |
| two_way | +2.6 [−0.8, +6.2] / +1.2 | +1.5 [−0.9, +3.9] / +1.8 |
| interfloor | +1.1 [−2.0, +4.4] / +0.3 | −1.0 [−3.0, +1.2] / −2.6 |

* `liftzero_bc_cnp` meets both targets in **all 13 regimes** (4 regimes + 9 scenarios).
* `liftzero_bc` meets p95 everywhere and avg wait in 10 of 13. Misses: `up_peak` +5.8 %,
  `morning_up_peak` +9.1 % (CI [−4.9, +26.7]) and `demo_story` +7.0 % (CI [+0.1, +14.1]).
* **Why the `liftzero_bc` up-peak misses are mostly noise, and where they are not.** In `full`,
  simulated-annealing reassignment and parking draw from the same seeded stream as passenger
  arrivals (a Phase 1 design), so once two strategies make one different decision their
  passengers diverge: `full`-family pairs are *seed*-paired, not *passenger*-paired. Up-peak
  is also the most variable regime (`full` alone spans ±25 % across seeds), which is why
  those CIs straddle zero. The bare-CNP pair has no such coupling (arrivals are identical —
  tested), and there the learned bidder is within ±1.5 % in every regime. `demo_story` (a
  scripted mixture with a fire and a fault) is the one miss whose CI excludes zero; it is
  also the scenario furthest from the training distribution.
* Learned awards differ from the shadow teacher's choice in ~9 % of auctions (both
  strategies) yet produce the same waiting times — most disagreements are near-ties.
* `compute_ms_per_tick` for learned strategies includes the shadow teacher used to measure
  disagreement; the network itself costs ~0.2 ms per CFP (below).

### Latency (`elevator lift bench-infer`, Apple M2, 1 thread, onnxruntime)

| cars N | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- |
| median ms | 0.13 | 0.15 | **0.19** | 0.27 | **0.43** |
| p99 ms | 0.15 | 0.18 | 0.22 | 0.30 | 0.48 |

Targets: ≤ 2 ms at N = 8, ≤ 5 ms at N = 32 — met by 10×. Torch↔ONNX parity 5.7e-6.

### A safety bug found by this evaluation (fixed)

The 100-seed matrix exposed a Phase 1 bug that affected **every** strategy: if a fire alarm hit
a car that had just left the lobby, fire recall cleared its target but kept the partial move
timer, so after `fire_clear` the car counted as "moving" while its doors cycled
(`violations()`: "car 0 moving with doors open"; 7 of 1,200 fire-scenario runs). Fixed in
`ElevatorAgent.enter_fire_mode`/`restore_normal_service`, pinned by
`tests/test_fire_recall_regression.py`; the fire scenarios were re-run for all strategies.
Golden metrics of the classical strategies are unchanged (they contain no fire).

## 10. Deviations from the Phase 4 prompt (and why)

| prompt | done | why |
| --- | --- | --- |
| branch `phase-4-imitation`, no push | work on `main`, pushed | team directive (`prompt/00_START_HERE.md`) |
| ONNX opset 17 | opset 18, `torch.export` exporter | PyTorch 2.14's legacy TorchScript exporter is deprecated and diverged from eager mode by ~0.06 on this graph; the new exporter needs opset ≥ 18. Parity 5.7e-6. |
| `nn.MultiheadAttention` in the last layer | explicit masked attention in every layer | identical maths, attention weights for every layer, clean dynamic-shape export |
| ~0.27 M parameters | 237,575 | within the 0.2–0.4 M budget with the specified d=64, L=4, FFN 256 |
| parity on all outputs | parity on real cars' outputs | padded/ineligible scores are `raw + 1e4` sentinels whose float32 ulp is ~1e-3; they are checked with a relative tolerance |
| agreement metric | strict **and** tie-aware reported side by side | the teacher breaks exact ties by car id, which an equivariant network cannot see |
| schema v1 | schema v2 (+4 public cost weights) | v1 gave identical inputs conflicting labels across the two teachers |
| ≤ 45 min training on 8 CPU cores | 19–25 min per seed on Apple M2 (MPS) | `device=auto` picks MPS; CPU-only throughput is ~2 k decisions/s |
| teacher self-consistency by re-labelling with another seed | label conflicts among identical / near-identical feature vectors | the A\* teacher is deterministic given the state, so re-labelling gives 100 % trivially; the meaningful bound is ambiguity of the *public features* |
| DAgger val seeds 8000–8499 | 8260–8499 (expert val used 8000–8247) | fresh states for every round |

## 11. Limitations

* Routing (stop order) is still classical A\*; eligibility is classical; only the bid is learned.
* Imitation can at best match the teacher — Phase 5 optimises the real objective with RL.
* Trained on one building family (single lobby at floor 0, 6–32 floors, 2–8 cars).
* Priority weights and exact waiting counts are not on the public board, so the network
  cannot see what the teacher multiplies by; `plan_end_eta` is clipped at 240 s.
* In `full`-family comparisons, passenger streams diverge between strategies (shared RNG),
  inflating variance; the bare-CNP comparison is the clean one.

## 12. Reproduce

```bash
uv sync --group learn --group lift-train
scripts/reproduce_phase4.sh                                  # smoke pipeline, ~15 min
# full pipeline (what produced the shipped model):
uv run elevator learn record --decisions 1200000 --workers 8 --out data/expert/train --seed-start 0
uv run elevator learn record --decisions 120000  --workers 8 --out data/expert/val   --seed-start 8000
uv run elevator learn record --decisions 150000  --workers 8 --out data/expert/test  --seed-start 8500
uv run elevator lift train-bc --preset default --seed 0      # and --seed 1, --seed 2
uv run elevator lift dagger --round 1 --ckpt runs/bc_default_s0_*/best.pt --beta 0.5
uv run elevator lift dagger --round 2 --ckpt runs/bc_dagger_r1_*/best.pt --beta 0.25
uv run elevator lift export --ckpt runs/bc_dagger_r2_*/best.pt --out models/liftzero_bc_v1.onnx
uv run elevator lift eval-offline --ckpt runs/bc_dagger_r2_*/best.pt
uv run elevator lift eval-sim --regimes all --seeds test --n 100 --workers 7
uv run elevator lift bench-infer
```

Run heavy jobs one at a time on an 8 GB machine: two trainers plus rollout workers drove
this laptop into 5 GB of swap and slowed everything ~20×.
