"""Regenerate `tests/golden/strategy_metrics.json`.

The golden file pins the metrics of the four classical strategies so that later phases can
prove their engine changes are behaviour-neutral. Run it only when a change to the classical
strategies is *intended*:

    uv run python scripts/generate_golden_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

from elevator_mas.sim import run_scenario

STRATEGIES = ("nearest_car", "collective", "cnp_astar", "full")
SCENARIOS = ("morning_up_peak", "evening_down_peak", "lunch_two_way")
SEEDS = (42, 101)
TICKS = 300
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "strategy_metrics.json"


def golden_key(strategy: str, scenario: str, seed: int) -> str:
    """The lookup key of one golden run."""
    return f"{strategy}/{scenario}/{seed}"


def golden_metrics(strategy: str, scenario: str, seed: int) -> dict[str, float | int]:
    """Metrics of one run, without the wall-clock-dependent field."""
    result = run_scenario(scenario, strategy=strategy, seed=seed, ticks=TICKS)
    metrics = result.metrics.as_dict()
    metrics.pop("compute_ms_per_tick", None)
    return metrics


def main() -> None:
    golden = {
        golden_key(st, sc, sd): golden_metrics(st, sc, sd)
        for st in STRATEGIES
        for sc in SCENARIOS
        for sd in SEEDS
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(golden)} golden runs to {OUT}")


if __name__ == "__main__":
    main()
