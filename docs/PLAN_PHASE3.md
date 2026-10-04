# Plan: Phase 3 — Learning Environment for LiftZero

## Objective
Build the foundation for LiftZero (Phases 4–6):
1. Versioned feature schema + encoder (`KC=14`, `KCAR=26`, `KG=6`, `MAX_CARS=16`).
2. Expert decision recorder with single behaviour-neutral engine hook (`DecisionEvent`).
3. Fast twin simulator (Numba/NumPy SoA, semi-MDP, ≥20x faster, ±10% fidelity).
4. Gymnasium single & vectorised environments (`ElevatorDecisionEnv`, `VectorDecisionEnv`).
5. Regimes, baseline evaluation harness, CLI commands, and comprehensive reports.

---

## Architecture & Modules

```
src/elevator_mas/
  learning/
    __init__.py             # Public API exports (§9)
    schema.py               # Feature names, constants, bounds, normalization
    view.py                 # DecisionContext, FleetView, from_event(), from_twin()
    features.py             # encode_decision(), encode_batch()
    recorder.py             # DecisionRecorder hook, shard writer, worker multiprocessing
    dataset.py              # Shard reader, splits, dataset stats, fixture generator
    twin/
      __init__.py
      state.py              # Structure-of-arrays state for B parallel envs
      kernels.py            # @njit / pure Python physical stepping & LOOK routing
      traffic.py            # Poisson arrival generators matching traffic/
      policies.py           # nearest, collective, cost_greedy, random_eligible
    env.py                  # ElevatorDecisionEnv + VectorDecisionEnv + RewardConfig
    regimes.py              # configs/learning/regimes.yaml loader
    evaluate.py             # evaluate(), summarise(), bootstrap CIs, paired diffs
configs/learning/regimes.yaml
docs/LEARNING_ENV.md
docs/LEARNING_ENV_fidelity.md
reports/learning_baselines.md
tests/learning/
  test_schema.py
  test_features.py
  test_hook_neutrality.py
  test_recorder.py
  test_twin_invariants.py
  test_twin_fidelity.py
  test_env_api.py
  test_regimes.py
  test_evaluate.py
```

---

## Step-by-Step Execution Plan

### Step 1: Engine Hook & Neutrality Verification
- Create `DecisionEvent` frozen dataclass in `src/elevator_mas/learning/events.py` (or `comms/events.py`).
- Add `self.decision_hooks: list[Callable[[DecisionEvent], None]] = []` in `ElevatorModel.__init__`.
- In `DispatcherAgent.announce()`: capture public snapshot `self._round_view = (tuple(self.model.board.cars()), self.model.board.policy.copy())`.
- In `DispatcherAgent.award()`: fire `decision_hooks` with `DecisionEvent`.
- Add `tests/learning/test_hook_neutrality.py` asserting identical metrics & message counts with and without hook across 3 scenarios × 2 strategies × 2 seeds.

### Step 2: Feature Schema & Unified Encoder
- Implement `src/elevator_mas/learning/schema.py` (`FEATURE_VERSION=1`, `KC=14`, `KCAR=26`, `KG=6`, `MAX_CARS=16`).
- Implement `src/elevator_mas/learning/view.py` (`DecisionContext`, `FleetView`, conversions).
- Implement `src/elevator_mas/learning/features.py` (`encode_decision`, `encode_batch`).
- Write `tests/learning/test_schema.py` and `tests/learning/test_features.py` (shapes, ranges, no NaN/inf, masking, permutation invariance, route truth-table, bit-identical parity between real and twin).

### Step 3: Fast Twin Simulator
- Implement `src/elevator_mas/learning/twin/state.py` (SoA batch arrays for B environments).
- Implement `src/elevator_mas/learning/twin/kernels.py` (Numba `@njit` kernels with pure Python fallback via `ELEVATOR_NO_NUMBA=1`).
- Implement `src/elevator_mas/learning/twin/traffic.py` (Poisson arrivals & 5 OD patterns matching real generator).
- Implement `src/elevator_mas/learning/twin/policies.py` (`nearest`, `collective`, `cost_greedy`, `random_eligible`).
- Write `tests/learning/test_twin_invariants.py` (conservation, door safety, capacity, determinism, drain, fault/fire events).

### Step 4: Decision Recorder & Expert Dataset
- Implement `src/elevator_mas/learning/recorder.py` (multiprocessing shard writer with domain randomisation).
- Implement `src/elevator_mas/learning/dataset.py` (shard loading, train/val/test splits, hard decision fraction calculation).
- Generate small committed fixture: `tests/learning/fixtures/expert_sample.npz` (1 000 decisions, <300 kB).
- Write `tests/learning/test_recorder.py`.

### Step 5: Gymnasium Environment
- Implement `src/elevator_mas/learning/env.py` (`ElevatorDecisionEnv`, `VectorDecisionEnv`, `RewardConfig`).
- Support observation space Dict, Discrete(MAX_CARS) action space, action masking, invalid action fallback, and semi-MDP stepping.
- Write `tests/learning/test_env_api.py` (`check_env`, determinism, vector env equivalence, 100-episode soak).

### Step 6: Regimes & Evaluation Harness
- Create `configs/learning/regimes.yaml`.
- Implement `src/elevator_mas/learning/regimes.py`.
- Implement `src/elevator_mas/learning/evaluate.py` (bootstrap CIs, paired diffs).
- Write `tests/learning/test_regimes.py` and `tests/learning/test_evaluate.py`.

### Step 7: CLI Integration & Benchmark / Fidelity Runs
- Add `learn` command group to `src/elevator_mas/cli.py` (`record`, `dataset-info`, `validate-twin`, `bench-env`, `baselines`).
- Harvest ≥50 000 expert decisions into `data/expert/`.
- Run twin speed benchmark (`elevator learn bench-env`) and verify ≥20x speedup.
- Run twin fidelity validation (`elevator learn validate-twin`) and generate `docs/LEARNING_ENV_fidelity.md`.
- Run baselines (`elevator learn baselines`) and generate `reports/learning_baselines.md` and `.csv`.
- Add `tests/learning/test_twin_fidelity.py` (reduced slow test).

### Step 8: Documentation & Verification Checklist
- Write `docs/LEARNING_ENV.md`.
- Ensure all tests pass with both `uv run pytest` and `uv sync --group learn && uv run pytest`.
- Verify `ruff check` and `ruff format` are clean.
- Commit all changes cleanly to `phase-3-learning-env` (do not touch `main`, do not push).
