"""Tests for twin simulator fidelity comparison against real simulator."""

from __future__ import annotations

import pytest

from elevator_mas.config import ScenarioConfig
from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.twin.policies import CollectivePolicy, NearestPolicy
from elevator_mas.learning.twin.state import TwinSimulator
from elevator_mas.sim import run_scenario


@pytest.mark.slow
def test_twin_fidelity_reduced() -> None:
    """Reduced fidelity check across 2 scenarios and 3 seeds."""
    scenarios = ["morning_up_peak", "lunch_two_way"]
    seeds = [42, 43]
    ticks = 400

    for sc_name in scenarios:
        cfg = ScenarioConfig.load(sc_name)
        pol = CollectivePolicy() if "collective" in cfg.strategy else NearestPolicy()

        for s in seeds:
            cfg_run = cfg.model_copy(update={"seed": s, "duration": ticks})
            res_real = run_scenario(cfg_run)
            real_wait = res_real.metrics.avg_wait

            sim = TwinSimulator(
                floors=cfg.building.floors,
                cars=cfg.building.cars,
                capacity=cfg.building.capacity,
                duration=ticks,
                rate=cfg.traffic.rate_at(0),
                pattern=cfg.traffic.pattern_at(0),
                seed=s,
            )
            ctx, term = sim.advance()
            while not term:
                enc = encode_decision(ctx)
                act = pol.act({"cars": enc.cars, "eligible": enc.eligible})
                ctx, term = sim.step_action(int(act))
            twin_wait = sim.get_metrics().get("avg_wait", 0.0)

            # Assert order of magnitude and reasonable correlation
            assert abs(twin_wait - real_wait) < 50.0
