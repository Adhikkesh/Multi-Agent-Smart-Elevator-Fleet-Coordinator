# LiftZero Phase 3: The Learning Environment

This document describes the LiftZero learning foundation built during Phase 3, enabling imitation learning (Phase 4), cooperative PPO reinforcement learning (Phase 5), and MCTS look-ahead planning (Phase 6).

---

## 1. Overview & Architecture

```
src/elevator_mas/learning/
  ├── __init__.py           # Lazy public exports
  ├── schema.py             # Feature schema (Kc=14, Kcar=26, Kg=6, MAX_CARS=16)
  ├── view.py               # DecisionContext & FleetView (bridges real sim & twin)
  ├── features.py           # Unified feature encoder (single shared code path)
  ├── recorder.py           # DecisionRecorder observer hook & shard generator
  ├── dataset.py            # ExpertDataset shard loader, splits, batch iterator, stats
  ├── env.py                # Gymnasium ElevatorDecisionEnv + VectorDecisionEnv
  ├── regimes.py            # Evaluation regimes loader (configs/learning/regimes.yaml)
  ├── evaluate.py           # Policy evaluation harness with bootstrap CIs
  ├── cli.py                # CLI commands under `elevator learn`
  └── twin/
      ├── __init__.py
      ├── state.py          # Fast array-based TwinSimulator (semi-MDP)
      ├── kernels.py        # LOOK stop sequencing with Numba JIT acceleration
      ├── traffic.py        # Poisson arrival generator matching traffic/
      └── policies.py       # Built-in baselines (nearest, collective, cost_greedy, random)
```

---

## 2. Feature Schema (Version 1)

Every assignment decision is encoded into a permutation-invariant set of tokens:
- **Call token** (`Kc = 14`): floor, direction, lobby flag, waiting count, urgency, distance to lobby, demand share, one-hot traffic pattern, open call count, fleet size.
- **Car tokens** (`MAX_CARS = 16, Kcar = 26`): floor, one-hot direction, load, remaining space, available/oos/fire flags, one-hot door state, door blocked ticks, assigned count, car calls, riders, plan stops, plan end floor, plan end ETA, dist to call, signed dist, heading, call on route, same call flag, park target dist.
- **Global token** (`Kg = 6`): scenario progress, normalized floors, normalized capacity, available ratio, fleet total load, fleet idle fraction.
- **Masks**: `mask[MAX_CARS]` (presence), `eligible[MAX_CARS]` (can bid/serve).

Both the real simulator (`from_event`) and the twin simulator (`from_twin`) produce identical `DecisionContext` representations, ensuring trained models transfer without distribution shift.

---

## 3. Expert Dataset Recorder

- Non-intrusive `decision_hooks` on `ElevatorModel` capture auction snapshots (`DecisionEvent`).
- Domain randomisation across 6–40 floors, 2–8 cars, capacities 6–16, time-varying traffic patterns, and scripted disturbances.
- Data splits are assigned by seed (never by decision):
  - Train: seeds < 8000
  - Validation: seeds 8000–8499
  - Test: seeds 8500–8999
- A 1,000-decision fixture is committed under `tests/learning/fixtures/expert_sample.npz` (82 KB).

---

## 4. Fast Twin Simulator

- Re-implements physical world stepping and assignment semi-MDP without agent message overhead or A* search.
- Utilizes LOOK sweep intra-car sequencing, matching real simulator sweep rules.
- Supports pure Python fallback via `ELEVATOR_NO_NUMBA=1` and Numba `@njit` kernels.
- Delivers >1,400 decisions/second single-threaded, achieving a ~10x speedup over real simulation with classical A* bids.

---

## 5. Gymnasium Environment

- `ElevatorDecisionEnv(gymnasium.Env)` implements fixed-horizon semi-MDP:
  - Observation: dictionary matching feature schema.
  - Action: `Discrete(MAX_CARS)` with action masking.
  - Invalid action fallback: automatically routes to lowest-cost eligible car by `cost_greedy`.
  - Team reward:
    $$r = -\frac{\sum w_i \Delta t + 0.5 \sum r_i \Delta t}{100} - 0.02 \cdot \Delta\text{floors} - 0.5 \cdot \text{new\_long\_waits}$$
- `VectorDecisionEnv`: batches multiple parallel simulations in lockstep for PPO training.

---

## 6. CLI Commands

```bash
# Record expert decisions from classical teachers
uv run elevator learn record --teacher mixed --decisions 50000 --workers 4 --out data/expert

# Inspect dataset metrics and baseline accuracies
uv run elevator learn dataset-info --path tests/learning/fixtures/expert_sample.npz

# Benchmark environment throughput
uv run elevator learn bench-env --envs 64 --steps 500

# Validate twin simulator against Mesa model
uv run elevator learn validate-twin --seeds 10 --ticks 900

# Evaluate baselines across the 4 canonical regimes
uv run elevator learn baselines --seeds 20
```
