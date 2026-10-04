# Multi-Agent Smart Elevator Fleet Coordinator

A multi-agent simulation and live web dashboard in which autonomous agents coordinate a
fleet of elevators in a multi-storey building. Built as a university case study for
**Fundamentals of AI** (Russell & Norvig, *AIMA* 4e).

Every design decision maps to a course concept: agent types, PEAS, environment
properties, state-space search (BFS / UCS / Greedy / A*), local search (hill climbing,
simulated annealing), adversarial search (minimax with alpha-beta) and rule-based
reasoning (forward chaining).

## Quick start

```bash
uv sync                 # fetches Python 3.12 and all pinned dependencies
uv run elevator serve   # → http://localhost:8000
```

`uv sync` bootstraps its own CPython 3.12, so no system Python is required.

## Commands

| Command | What it does |
| --- | --- |
| `uv run elevator serve` | Start the FastAPI + WebSocket server and the 4-tab dashboard. |
| `uv run elevator run --scenario morning_up_peak` | Headless run, prints the metric summary. |
| `uv run elevator run --scenario demo_story --csv reports/demo.csv` | Headless run, exports per-tick metrics. |
| `uv run elevator bench` | Strategies x scenarios x seeds benchmark → `reports/*.csv` + PNG charts. |
| `uv run elevator scenarios` | List the bundled scenarios. |
| `uv run pytest` | Full test suite. |
| `uv run pytest --cov` | Test suite with coverage of the core logic. |
| `uv run ruff check .` | Lint. |

## The dashboard: LiftZero Control Room (Phase 2)

Running `uv run elevator serve` serves the modern **LiftZero Control Room** web application at `http://localhost:8000`:
- **Mission Control (`/`)**: High-density operational dashboard featuring an animated building shaft visualizer with interpolated car physics, door states, priority badges, and fire smoke particles; 8 live KPI sparkline cards; Contract Net Protocol auction bidding panel; filterable FIPA-ACL message stream; and real-time disturbance & chaos injection (car faults, fire alarms, passenger rushes).
- **Multi-Agent Architecture (`/agents`)**: Interactive directed agent graph (`@xyflow/react`), FIPA-ACL sequence diagram swimlanes, shared blackboard viewer, and comprehensive PEAS / AIMA slide-out inspector.
- **Search & Optimization Lab (`/lab`)**: Live comparative search benchmarks (BFS, UCS, Greedy, A* with heuristic visualization and interactive step-through debugger), Simulated Annealing temperature cooling and energy convergence curves, and adversarial Minimax vs Greedy dispatch games.
- **Experiments & Benchmarking (`/experiments`)**: Multi-seed Monte-Carlo matrix benchmark runner, side-by-side synchronized replay with timeline scrubber (Compare Mode), and CSV/JSON/PNG export.
- **Theory & Viva Primer (`/theory`)**: Interactive AIMA 4e cheat sheet, PEAS formulations, CNP state machine, admissibility proofs, and 12 viva defence answers.
- **Story Mode (`/story`)**: Guided 10-beat interactive presentation walkthrough.
- **Classic Fallback (`/classic`)**: The original lightweight single-page HTML/Canvas dashboard is preserved at `/classic`.

See [`docs/UI.md`](docs/UI.md) and [`frontend/README.md`](frontend/README.md) for full frontend architecture, build instructions, and testing details.

## Documentation

- [`docs/UI.md`](docs/UI.md) — LiftZero Control Room React UI architecture, features, bundle metrics, and test coverage.
- [`docs/DESIGN.md`](docs/DESIGN.md) — PEAS, environment analysis, state-space
  formulation, algorithms with complexity and the heuristic admissibility/consistency
  proofs, tool-selection rationale, team work split.
- [`docs/PLAN.md`](docs/PLAN.md) — milestones.
- [`docs/TESTING.md`](docs/TESTING.md) — scenario results table and test coverage guide (pytest, vitest, playwright).
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — the 5-minute live demo and Story Mode walkthrough.
- [`docs/VIVA_QA.md`](docs/VIVA_QA.md) — 20 likely examiner questions with answers.
- [`AGENTS.md`](AGENTS.md) — repo conventions.

## Layout

```
src/elevator_mas/
  config.py        pydantic + YAML configuration models
  model.py         the Mesa Model: staged activation, wiring, DataCollector
  agents/          Passenger, Floor, Elevator, Dispatcher, TrafficMonitor, Safety
  comms/           FIPA-ACL-style Message and the MessageBus
  planning/        generic SearchProblem + BFS/UCS/Greedy/A*, car routing problem
  optimization/    simulated annealing, hill climbing, minimax + alpha-beta
  strategies/      pluggable dispatch strategies (registry pattern)
  traffic/         Poisson arrival generator and traffic-pattern profiles
  rules/           forward-chaining production rules for safety
  metrics.py       metric definitions and aggregation
  sim/             headless runner and the benchmark harness
  api/             FastAPI REST + WebSocket streaming
  web/             single-page vanilla HTML/CSS/JS + Canvas dashboard
configs/scenarios/ *.yaml
tests/ docs/ reports/
```
