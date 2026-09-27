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

## The dashboard

1. **Live Simulation** — animated building, per-car state, hall calls coloured by the
   assigned car, live KPIs, the last Contract Net auction round as a bid chart, an agent
   inspector, the colour-coded message log and the rules-fired log. Inject a car fault,
   a fire alarm or a passenger rush at any time.
2. **Search Lab** — snapshot a live car's routing problem and run BFS / UCS / Greedy / A*
   side by side; simulated-annealing convergence curve; minimax vs alpha-beta node counts.
3. **Benchmark** — run strategies x scenarios x seeds headless and compare mean ± std.
4. **Theory** — PEAS table per agent, environment classification, architecture diagram
   and the CNP sequence diagram.

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — PEAS, environment analysis, state-space
  formulation, algorithms with complexity and the heuristic admissibility/consistency
  proofs, tool-selection rationale, team work split.
- [`docs/PLAN.md`](docs/PLAN.md) — milestones.
- [`docs/TESTING.md`](docs/TESTING.md) — scenario results table and what each test proves.
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — the 5-minute live demo walkthrough.
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
