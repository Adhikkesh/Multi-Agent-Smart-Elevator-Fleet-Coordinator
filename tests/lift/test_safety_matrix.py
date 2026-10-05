"""liftzero_bc on every scenario x 3 seeds: zero invariant violations, everyone delivered."""

from __future__ import annotations

from pathlib import Path

import pytest

from elevator_mas.comms.board import DecisionEvent
from elevator_mas.config import ScenarioConfig, available_scenarios
from elevator_mas.model import ElevatorModel

SCENARIOS = available_scenarios()


def test_all_nine_scenarios_are_covered() -> None:
    assert len(SCENARIOS) == 9


@pytest.mark.slow
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_liftzero_bc_is_safe(scenario: str, seed: int, shipped_model: Path) -> None:
    cfg = ScenarioConfig.load(scenario).model_copy(update={"strategy": "liftzero_bc", "seed": seed})
    model = ElevatorModel(cfg)
    bypass: list[str] = []

    def no_refuse_bypass(e: DecisionEvent) -> None:
        if e.winner is None:
            return
        s = next(x for x in e.statuses if x.car_id == e.winner)
        if not s.available or s.space <= 0:
            bypass.append(f"tick {e.tick}: ineligible car {e.winner} won {e.call}")

    model.decision_hooks.append(no_refuse_bypass)
    violations: list[str] = []
    for _ in range(cfg.duration):  # car_fault / fire_alarm / rush events fire in here
        model.step()
        violations.extend(model.violations())
    model.drain()
    violations.extend(model.violations())
    assert violations == []
    assert bypass == []
    assert all(r.delivered for r in model.collector.records.values())
