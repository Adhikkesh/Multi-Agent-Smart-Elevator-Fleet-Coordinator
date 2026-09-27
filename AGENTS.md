# Repo conventions

## Toolchain

Python 3.12, managed by [uv](https://docs.astral.sh/uv/). `uv sync` fetches the
interpreter itself, so the host Python version does not matter.

```bash
uv sync
uv run elevator serve          # dashboard at http://localhost:8000
uv run elevator run --scenario morning_up_peak
uv run elevator bench
uv run pytest
uv run pytest --cov            # coverage of core logic
uv run ruff check . && uv run ruff format --check .
```

## Layout

`src/` layout, package `elevator_mas`. The headless engine (`model`, `agents`,
`planning`, `optimization`, `strategies`, `traffic`, `rules`, `metrics`, `sim`) never
imports from `api` or `web`; the UI is strictly a consumer.

## Rules

- **Mesa 3 API only.** `Agent.__init__(self, model, ...)` calling `super().__init__(model)`;
  no schedulers — staged activation goes through `model.agents_by_type[Cls].do("stage")`;
  removal via `agent.remove()`. Construct the model with `rng=<seed>` (Mesa 3.5 deprecates
  `seed=`).
- **All randomness from `model.random` / `model.rng`.** Never `random.*` or
  `numpy.random.*` at module level — it breaks seed reproducibility, which the tests assert.
- **No global mutable state.** Configuration is passed in; strategies come from the registry.
- Type hints and docstrings throughout. Each agent's docstring states its AIMA agent type
  and its PEAS.
- Anything changeable from YAML (floors, cars, strategies, scenarios) must not require a
  code edit.
- Line length 100, `ruff` for lint and format.

## Tests

`tests/` mirrors the package layout. Correctness properties (A* == UCS, admissibility,
consistency, collective legality, run invariants, determinism, CNP protocol invariants)
are asserted over many seeded random instances rather than single hand-picked cases.
