"""Test that adding a decision hook to ElevatorModel is strictly behaviour-neutral.

Runs 3 scenarios × 2 strategies × 2 seeds with and without a hook, asserting
identical Metrics.as_dict() and identical bus.total_sent.
"""

from __future__ import annotations

import pytest

from elevator_mas.config import ScenarioConfig
from elevator_mas.model import ElevatorModel


@pytest.mark.parametrize("scenario_name", ["morning_up_peak", "evening_down_peak", "lunch_two_way"])
@pytest.mark.parametrize("strategy", ["nearest_car", "cnp_astar"])
@pytest.mark.parametrize("seed", [42, 101])
def test_hook_neutrality(scenario_name: str, strategy: str, seed: int) -> None:
    # 1. Run without hook
    base_cfg = ScenarioConfig.load(scenario_name)
    base_cfg.strategy = strategy
    base_cfg.seed = seed
    base_cfg.duration = 180  # fast runs for test suite

    model_no_hook = ElevatorModel(base_cfg, seed=seed)
    for _ in range(base_cfg.duration):
        model_no_hook.step()
    metrics_no_hook = model_no_hook.latest_metrics
    msgs_no_hook = model_no_hook.bus.total_sent

    # 2. Run with a no-op hook that records events
    hook_calls: list[int] = []

    def noop_hook(evt: object) -> None:
        hook_calls.append(getattr(evt, "tick", 0))

    model_with_hook = ElevatorModel(base_cfg, seed=seed)
    model_with_hook.decision_hooks.append(noop_hook)

    # Step through ticks identically
    for _ in range(base_cfg.duration):
        model_with_hook.step()

    metrics_with_hook = model_with_hook.latest_metrics
    msgs_with_hook = model_with_hook.bus.total_sent

    # 3. Assert exact equality of all simulation metrics
    d_no_hook = metrics_no_hook.as_dict()
    d_with_hook = metrics_with_hook.as_dict()
    d_no_hook.pop("compute_ms_per_tick", None)
    d_with_hook.pop("compute_ms_per_tick", None)

    assert d_with_hook == d_no_hook
    assert msgs_with_hook == msgs_no_hook
    # Ensure hook actually fired during the run (calls were auctioned)
    assert len(hook_calls) > 0
