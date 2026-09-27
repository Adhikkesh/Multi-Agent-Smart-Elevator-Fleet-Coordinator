# Implementation Plan

Milestones, in order. The app stays runnable after every milestone; `ruff check .` and
`pytest` run clean at the end of each.

## M1 — Engine, agents, Contract Net, A*, tests

- `config.py`: pydantic models for building, timing, traffic and scenario YAML.
- `comms/`: FIPA-ACL-style `Message` (performative, sender, receiver, conversation_id,
  content, tick) and a logging `MessageBus`.
- `planning/`: generic `SearchProblem` and one shared node/frontier implementation
  driving BFS, UCS, Greedy best-first and A*; each returns path, cost, nodes expanded,
  max frontier size and runtime. Car routing expressed as a `SearchProblem` with
  collective-control legality and the weighted-arrival-time cost.
- `agents/`: Passenger (simple reflex), Floor (model-based reflex), Elevator
  (goal + utility based), Dispatcher (coordinator/auctioneer), TrafficMonitor
  (learning), Safety (knowledge-based).
- `model.py`: Mesa 3 `Model` with deterministic staged activation
  (sense → communicate → decide → act → collect) and a `DataCollector`.
- Tests: A* cost == UCS cost, A* expands ≤ UCS, heuristic admissible and consistent,
  plans obey collective legality, run invariants, determinism by seed.

**Exit:** `uv run elevator run --scenario morning_up_peak` prints a metric summary.

## M2 — Live dashboard

- `api/`: FastAPI REST + WebSocket streaming of tick snapshots; play/pause/step/reset,
  speed, seed, scenario, strategy, floors/cars.
- `web/`: single-page vanilla HTML/CSS/JS + Canvas. Animated building with interpolation,
  live KPIs, auction panel, agent inspector, message log, rules-fired log, AWT chart.
  Chart.js vendored in `static/vendor/` so the demo works offline.

**Exit:** `uv run elevator serve` shows a live, animated, interactive simulation.

## M3 — Local search, parking, safety rules, faults, all scenarios

- `optimization/`: simulated annealing and a hill-climbing variant over the assignment of
  not-yet-picked-up calls, with a hysteresis threshold; minimax with alpha-beta for
  worst-case idle-car parking.
- `rules/`: forward-chaining production rules for fire recall, overload and car faults,
  plus the same rules as Prolog in `docs/safety_rules.pl`.
- `traffic/`: EWMA arrival-rate estimation and rule-based pattern classification.
- `strategies/`: NearestCar, Collective/LOOK, CNP + A*, Full — behind a registry.
- All 9 scenarios in `configs/scenarios/`, each with asserted expected behaviour.

**Exit:** every scenario runs; fault and fire injection work live.

## M4 — Search Lab, benchmark, docs

- Search Lab tab: BFS/UCS/Greedy/A* side by side on a snapshotted live problem, SA
  convergence curve, minimax vs alpha-beta node counts.
- Benchmark tab + `elevator bench`: strategies x scenarios x seeds, mean ± std tables and
  PNG charts into `reports/`.
- Theory tab and `docs/DESIGN.md`, `TESTING.md`, `DEMO_SCRIPT.md`, `VIVA_QA.md`.

**Exit:** all tests pass, all 4 tabs work, Full beats NearestCar on average wait and
% long waits in the peak scenarios, docs complete.
