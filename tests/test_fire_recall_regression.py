"""Regression: a fire alarm during a move must never leave a car "moving" with open doors.

Found by the Phase 4 closed-loop test matrix (100 seeds per scenario): when the alarm hit a
car that had just left the lobby, fire recall cleared its target but kept the partial move
timer; after `fire_clear` the car got a new target while its doors cycled at the lobby, and
`violations()` reported "moving with doors open". The seeds below reproduced it.
"""

from __future__ import annotations

import pytest

from elevator_mas.sim import run_scenario


@pytest.mark.parametrize(
    ("strategy", "scenario", "seed"),
    [
        ("full", "fire_emergency", 8501),
        ("full", "fire_emergency", 8593),
        ("nearest_car", "fire_emergency", 8594),
        ("collective", "demo_story", 8533),
    ],
)
def test_fire_recall_mid_move_keeps_the_invariants(strategy: str, scenario: str, seed: int) -> None:
    result = run_scenario(scenario, strategy=strategy, seed=seed, drain=True)
    assert result.violations == []
    assert result.delivered_all


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["nearest_car", "collective", "cnp_astar", "full"])
def test_fire_scenarios_clean_over_many_seeds(strategy: str) -> None:
    for seed in range(8500, 8530):
        for scenario in ("fire_emergency", "demo_story"):
            assert run_scenario(scenario, strategy=strategy, seed=seed).violations == []
