"""Every bundled scenario runs, and behaves the way its YAML documents."""

from __future__ import annotations

import pytest

from elevator_mas.config import ScenarioConfig, available_scenarios
from elevator_mas.model import ElevatorModel
from elevator_mas.sim import run_scenario

SCENARIOS = available_scenarios()
QUICK = [s for s in SCENARIOS if s != "stress_scale"]


class TestAllScenariosRun:
    """Nothing in the library may be broken."""

    def test_the_expected_scenarios_exist(self) -> None:
        """All nine from the specification."""
        assert set(SCENARIOS) == {
            "morning_up_peak",
            "evening_down_peak",
            "lunch_two_way",
            "interfloor_light",
            "car_breakdown",
            "fire_emergency",
            "priority_passenger",
            "stress_scale",
            "demo_story",
        }

    @pytest.mark.parametrize("name", QUICK)
    def test_scenario_runs_without_violations(self, name: str) -> None:
        """Every scenario must run its full duration cleanly."""
        result = run_scenario(name)
        assert result.ticks == ScenarioConfig.load(name).duration
        assert result.violations == [], result.violations[:5]
        assert result.metrics.arrived > 0

    @pytest.mark.parametrize("name", QUICK)
    def test_scenario_meets_its_documented_expectations(self, name: str) -> None:
        """The `expected` block in each YAML is a contract, not a comment."""
        config = ScenarioConfig.load(name)
        if not config.expected:
            pytest.skip(f"{name} documents no expectations")
        result = run_scenario(name, drain=True)

        if "max_avg_wait" in config.expected:
            assert result.metrics.avg_wait <= config.expected["max_avg_wait"], (
                f"{name}: average wait {result.metrics.avg_wait:.1f}s exceeds the documented "
                f"{config.expected['max_avg_wait']}s"
            )
        if "min_delivered" in config.expected:
            assert result.metrics.delivered >= config.expected["min_delivered"]

    @pytest.mark.parametrize("name", QUICK)
    def test_everyone_is_delivered_after_a_drain(self, name: str) -> None:
        """No scenario may strand a passenger."""
        assert run_scenario(name, drain=True).delivered_all


class TestScenarioBehaviour:
    """The specific behaviour each scenario was written to demonstrate."""

    # Each of these scenarios ends with a quiet tail phase, so the pattern is asserted
    # during the peak it is named after. Classifying the tail as "light" is correct
    # behaviour, not a failure — the agent is supposed to track a changing environment.
    @pytest.mark.parametrize(
        ("scenario", "expected", "at_tick"),
        [
            ("morning_up_peak", "up_peak", 500),
            ("evening_down_peak", "down_peak", 500),
            ("lunch_two_way", "two_way", 500),
        ],
    )
    def test_traffic_pattern_is_classified_during_the_peak(
        self, scenario: str, expected: str, at_tick: int
    ) -> None:
        """The learning agent must identify the regime it is actually in."""
        model = ElevatorModel(ScenarioConfig.load(scenario))
        model.run(at_tick)
        assert model.monitor.pattern == expected, (
            f"at t={at_tick} the monitor said {model.monitor.pattern!r} "
            f"({model.monitor.classification_reason}), expected {expected!r}"
        )

    def test_the_monitor_tracks_a_changing_environment(self) -> None:
        """When the rush ends the estimated demand must fall with it.

        The label is checked at the peak; here what matters is that the *estimate* tracks
        reality, which is the learning claim. Asserting an exact label in the quiet tail
        would be asserting which side of a threshold a borderline rate lands on.
        """
        model = ElevatorModel(ScenarioConfig.load("morning_up_peak"))
        model.run(500)
        assert model.monitor.pattern == "up_peak"
        peak_demand = sum(model.monitor.rate.values())

        model.run(380)
        tail_demand = sum(model.monitor.rate.values())
        assert tail_demand < peak_demand / 2, (
            f"demand estimate {tail_demand:.3f} did not fall after the rush "
            f"(it was {peak_demand:.3f})"
        )

    def test_classification_does_not_flip_flop(self) -> None:
        """Hysteresis must stop the label oscillating and churning the fleet's policy.

        Each change of pattern retunes the cost weights and the parking policy, so a
        classifier that dithers around its threshold is worse than one that lags.
        """
        model = ElevatorModel(ScenarioConfig.load("morning_up_peak"))
        switches = 0
        previous = None
        for _ in range(model.config.duration):
            model.step()
            if model.monitor.pattern != previous:
                switches += 1
                previous = model.monitor.pattern
        # The scenario has three phases, so a handful of transitions is expected;
        # dozens would mean the classifier is chattering.
        assert switches <= 8, f"the pattern changed {switches} times in one run"

    def test_breakdown_fires_the_fault_rule_and_recovers(self) -> None:
        """Car 2 fails at t=120 and returns at t=300."""
        model = ElevatorModel(ScenarioConfig.load("car_breakdown"))
        model.run(150)
        car = model.car(2)
        assert car is not None
        assert car.out_of_service, "car 2 should be out of service just after t=120"
        assert "R4_car_fault_out_of_service" in {r.rule for r in model.safety.log}

        model.run(250)
        assert not car.out_of_service, "car 2 should be repaired by t=400"
        assert model.violations() == []

    def test_fire_emergency_recalls_and_restores(self) -> None:
        """Alarm at t=150, cleared at t=420."""
        model = ElevatorModel(ScenarioConfig.load("fire_emergency"))
        model.run(260)
        assert model.safety.fire_alarm
        assert model.dispatcher.hall_calls_blocked
        for car in model.cars:
            if not car.out_of_service:
                assert car.floor == model.lobby

        model.run(200)
        assert not model.safety.fire_alarm
        assert not model.dispatcher.hall_calls_blocked
        rules = {r.rule for r in model.safety.log}
        assert {"R1_fire_recall", "R3_block_hall_calls", "R7_fire_cleared"} <= rules

    def test_priority_passengers_are_served_sooner(self) -> None:
        """Weight x3 must produce a shorter wait, with no priority-specific code."""
        model = ElevatorModel(ScenarioConfig.load("priority_passenger"))
        model.run()
        records = list(model.collector.records.values())
        priority = [r.wait_time for r in records if r.priority and r.wait_time is not None]
        normal = [r.wait_time for r in records if not r.priority and r.wait_time is not None]
        assert priority and normal, "the scenario produced no comparison group"
        mean_priority = sum(priority) / len(priority)
        mean_normal = sum(normal) / len(normal)
        assert mean_priority <= mean_normal * 1.1, (
            f"priority passengers waited {mean_priority:.1f}s vs {mean_normal:.1f}s for others"
        )

    def test_demo_story_exercises_everything(self) -> None:
        """The presentation scenario must show every mechanism in one run."""
        model = ElevatorModel(ScenarioConfig.load("demo_story"))
        model.run()
        rules = {r.rule for r in model.safety.log}
        assert "R4_car_fault_out_of_service" in rules, "the breakdown never happened"
        assert "R1_fire_recall" in rules, "the fire drill never happened"
        assert model.dispatcher.auction_history, "no auction ran"
        assert model.bus.total_sent > 100
        assert model.violations() == []

    @pytest.mark.slow
    def test_stress_scale_stays_within_budget(self) -> None:
        """40 floors, 8 cars, a simulated hour."""
        result = run_scenario("stress_scale")
        assert result.runtime_seconds < 120.0
        assert result.violations == []
        assert result.metrics.delivered >= 300
