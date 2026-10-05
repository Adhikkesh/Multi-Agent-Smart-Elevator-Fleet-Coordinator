# Phase 5 — Cooperative PPO (CTDE + KL anchor): design and status

> Owner: **Akash (CB.SC.U4CSE23162)**. Code: `src/elevator_mas/learning/lift/rl/`.
> Status: **implemented and smoke-tested; full training is run on Kaggle**
> (`docs/lift/KAGGLE_TRAINING.md`). No PPO results are claimed until that run is done.

## 1. Why RL after imitation

Imitation (Phase 4) can at best *match* the A\* teacher. RL optimises the true objective —
passenger waiting, riding, energy and long waits — directly, so it can find assignments the
heuristic misses (e.g. down-peak, where `collective` 21.1 s beats `full` 23.7 s on the test
seeds).

## 2. Formulation — a semi-Markov decision process

* **Decision** = one hall call to award. Decisions arrive at irregular times (Δt ≥ 0).
* **State (actor)**: the public Phase 3/4 feature tokens (call, cars, global incl. weights).
* **Action**: an eligible car; policy `π(a|s) = softmax_{a∈E}(−score_a · e^{−τ})`.
* **Team reward** (shared by every car ⇒ cooperative), accumulated per tick until the next
  decision: `r = −(Σ_waiting Δt + 0.5·Σ_riding Δt)/100 − 0.02·floors − 0.5·new_60s_waits`.
* **SMDP discount**: `γ_t = 0.99^(Δt/5)`; GAE `δ_t = r_t + γ_t V(s_{t+1}) − V(s_t)`,
  `A_t = δ_t + γ_t λ A_{t+1}`, λ = 0.95. The episode horizon *truncates* (bootstrap), it
  never terminates.

## 3. CTDE — centralised training, decentralised execution

* **Decentralised execution**: every car runs the same actor on the public board (ONNX, as in
  Phase 4). No car sees another's private state.
* **Asymmetric centralised critic** `V_priv`: additionally sees privileged state no car has —
  waiting counts and oldest-waiter age per floor/direction, riders per destination
  (`privileged.py`, 240 values). Used only for advantages; never an actor input (tested).
* **Public value** `V_pub` (the network's value head) is distilled from `V_priv` so Phase 6
  search can use it with public information only.

## 4. Loss

```
L = −E[min(ρA, clip(ρ,1±0.2)A)] + 0.5·Huber(V_priv − R) + 0.5·MSE(V_pub − sg V_priv)
    − c_e·H[π] + β·KL(π_BC ‖ π)
```
ρ = π/π_old. Entropy `c_e` 0.01 → 0.001; KL anchor β 0.2 → 0 over the first 60 % (keeps the
warm-started policy sane while the critic learns); Adam 3e-4 → 3e-5, grad-clip 0.5, 4 epochs
of 2 048-decision mini-batches, early stop if approx-KL > 0.03. Single-eligible decisions
carry no policy gradient but count for value targets.

## 5. Where it trains — a deliberate deviation

The prompt trains on the Phase 3 twin. Measured: the twin is only **3.7×** faster than the
real simulator (1,823 vs 499 decisions/s, not the documented 10,000×) and its fidelity report
shows 27–126 % wait-time error (its scenario events never fire). So PPO trains **directly on
the real Mesa simulator** through a dispatcher `award_hook` (cars still REFUSE classically;
the policy chooses among proposers). This removes the sim-to-real gap entirely. Training
buildings are sampled 50/50 in the `full`-like and bare-CNP settings with a scheduled
curriculum (easy → full → hard patterns ×2); training seeds are guarded to < 8000.

## 6. Checkpoint selection

Every 10 updates the deterministic policy is run on validation seeds (8000+) in the four
regimes and compared with the `full` teacher; a checkpoint is eligible only if p95 and
long-wait % are within 10 % of the imitation policy's (fairness guard) and there are zero
safety violations. Test seeds are touched once, after training.

## 7. Tests (`tests/lift/rl/`, 12)

Hand-computed SMDP-GAE with irregular Δt and truncation; GAE equals an O(T²) reference;
truncation bootstraps, termination does not; PPO ratio = 1 ⇒ approx-KL = 0 and KL(π‖π) = 0;
clipping caps the objective at (1+ε)A; masked policy gives ineligible cars zero mass and
single-car decisions zero entropy; actor independent of privileged state; no-test-seed guard;
bootstrap CI coverage ≥ 90 %, Holm and Wilcoxon against SciPy; an award hook that forces the
classical choice reproduces the classical run exactly; reward tracker = sum of components.

## 8. Not done (time)

Full 3-seed training, the final paired test table (Wilcoxon + Holm, win-rate, d_z — the
helpers exist in `rl/stats.py`), the reward-hacking audit and the explainability plots. Run
§3–4 of the Kaggle guide to produce them.
