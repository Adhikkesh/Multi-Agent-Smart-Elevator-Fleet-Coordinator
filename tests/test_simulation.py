"""Whole-run invariants: safety, determinism and no starvation."""

from __future__ import annotations

import pytest

from elevator_mas.config import (
    ArrivalPhase,
    BuildingConfig,
    ScenarioConfig,
    TrafficConfig,
)
from elevator_mas.domain import DoorState
from elevator_mas.model import ElevatorModel
from elevator_mas.sim import run_scenario
from elevator_mas.strategies import strategy_names


def runnable_strategies() -> list[str]:
    """Every registered strategy whose model (if it needs one) is present.

    Learned strategies whose ONNX model has not been trained yet (e.g. ``liftzero_ppo`` before
    the Kaggle run) are reported unavailable by the registry and are skipped here.
    """
    from elevator_mas.learning.lift.bidder import strategy_availability
    from elevator_mas.strategies import get_strategy

    return [n for n in strategy_names() if strategy_availability(get_strategy(n))[0]]


class TestInvariants:
    """Properties that must hold at every tick of every run."""

    @pytest.mark.parametrize("strategy", runnable_strategies())
    def test_no_violation_in_any_strategy(
        self, small_config: ScenarioConfig, strategy: str
    ) -> None:
        """Capacity, doors, bounds and service state, checked every tick."""
        model = ElevatorModel(small_config.model_copy(update={"strategy": strategy}))
        for _ in range(small_config.duration):
            model.step()
            assert model.violations() == []

    def test_capacity_is_never_exceeded(self, small_config: ScenarioConfig) -> None:
        """Boarding must respect the car's capacity."""
        model = ElevatorModel(small_config)
        for _ in range(small_config.duration):
            model.step()
            for car in model.cars:
                assert car.load <= car.capacity

    def test_doors_are_never_open_while_moving(self, small_config: ScenarioConfig) -> None:
        """The hard safety property."""
        model = ElevatorModel(small_config)
        for _ in range(small_config.duration):
            model.step()
            for car in model.cars:
                if car.moving:
                    assert car.door_state is DoorState.CLOSED

    def test_cars_stay_inside_the_building(self, small_config: ScenarioConfig) -> None:
        """No car may leave the shaft."""
        model = ElevatorModel(small_config)
        for _ in range(small_config.duration):
            model.step()
            for car in model.cars:
                assert 0 <= car.floor < model.floors_count

    def test_a_passenger_is_only_ever_in_one_place(self, small_config: ScenarioConfig) -> None:
        """Nobody may be aboard two cars, or aboard and waiting at once."""
        model = ElevatorModel(small_config)
        for _ in range(120):
            model.step()
            aboard = [p.unique_id for car in model.cars for p in car.riders]
            assert len(aboard) == len(set(aboard)), "a passenger is in two cars"
            waiting = {p.unique_id for p in model.passengers if p.waiting}
            assert not (waiting & set(aboard)), "a passenger is waiting and riding at once"

    def test_delivered_passengers_reach_their_destination(
        self, small_config: ScenarioConfig
    ) -> None:
        """Delivery means arriving at the requested floor, not merely leaving a car."""
        model = ElevatorModel(small_config)
        model.run(200)
        for record in model.collector.records.values():
            if record.delivered:
                assert record.alight_tick is not None
                assert record.alight_tick >= record.arrival_tick


class TestNoStarvation:
    """Everyone must eventually be delivered."""

    def test_every_passenger_is_delivered_after_a_drain(self) -> None:
        """The definitive no-starvation check."""
        config = ScenarioConfig(
            name="drain",
            duration=400,
            seed=5,
            strategy="full",
            building=BuildingConfig(floors=12, cars=3, capacity=8),
            traffic=TrafficConfig(phases=[ArrivalPhase(until=400, rate=0.2, pattern="two_way")]),
        )
        result = run_scenario(config, drain=True)
        assert result.delivered_all, (
            f"{result.metrics.arrived - result.metrics.delivered} passenger(s) never arrived"
        )
        assert result.violations == []

    @pytest.mark.parametrize("strategy", runnable_strategies())
    def test_no_starvation_under_any_strategy(self, strategy: str) -> None:
        """Even the reflex baseline must not strand anybody."""
        config = ScenarioConfig(
            name="drain",
            duration=300,
            seed=9,
            strategy=strategy,
            building=BuildingConfig(floors=10, cars=2, capacity=8),
            traffic=TrafficConfig(phases=[ArrivalPhase(until=300, rate=0.15, pattern="up_peak")]),
        )
        assert run_scenario(config, drain=True).delivered_all


#: Wall-clock timings cannot be reproducible, so they are excluded from the comparison.
NON_DETERMINISTIC = {"compute_ms_per_tick"}


def simulation_values(metrics) -> dict[str, float | int]:  # noqa: ANN001
    """Every metric except the wall-clock ones."""
    return {k: v for k, v in metrics.as_dict().items() if k not in NON_DETERMINISTIC}


class TestDeterminism:
    """The same seed must give exactly the same run."""

    def test_identical_seeds_give_identical_metrics(self, small_config: ScenarioConfig) -> None:
        """Reproducibility is what makes the benchmark meaningful."""
        first = ElevatorModel(small_config).run(200)
        second = ElevatorModel(small_config).run(200)
        assert simulation_values(first) == simulation_values(second)

    def test_identical_seeds_give_identical_car_positions(
        self, small_config: ScenarioConfig
    ) -> None:
        """Not just the aggregates — the whole trajectory must match."""
        a = ElevatorModel(small_config)
        b = ElevatorModel(small_config)
        for _ in range(150):
            a.step()
            b.step()
            assert [c.floor for c in a.cars] == [c.floor for c in b.cars]
            assert [c.load for c in a.cars] == [c.load for c in b.cars]

    def test_different_seeds_give_different_runs(self, small_config: ScenarioConfig) -> None:
        """A seed that changed nothing would mean the RNG is not really wired in."""
        first = ElevatorModel(small_config.model_copy(update={"seed": 1})).run(200)
        second = ElevatorModel(small_config.model_copy(update={"seed": 2})).run(200)
        assert simulation_values(first) != simulation_values(second)

    def test_message_traffic_is_reproducible(self, small_config: ScenarioConfig) -> None:
        """Even the protocol log must replay identically."""
        a = ElevatorModel(small_config)
        b = ElevatorModel(small_config)
        a.run(120)
        b.run(120)
        assert a.bus.total_sent == b.bus.total_sent
        assert [(m.performative, m.sender) for m in a.bus.history] == [
            (m.performative, m.sender) for m in b.bus.history
        ]


class TestScalability:
    """Configuration-only scaling, and the performance budget."""

    def test_the_building_is_configurable_without_code_changes(self) -> None:
        """Floors and cars come from config alone."""
        config = ScenarioConfig(
            name="big",
            duration=80,
            building=BuildingConfig(floors=30, cars=6, capacity=15),
            traffic=TrafficConfig(phases=[ArrivalPhase(until=80, rate=0.2, pattern="up_peak")]),
        )
        model = ElevatorModel(config)
        model.run(80)
        assert len(model.cars) == 6
        assert len(model.floors) == 30
        assert model.violations() == []

    def test_a_simulated_hour_at_scale_is_fast(self) -> None:
        """`stress_scale` must stay well inside the two-minute budget."""
        result = run_scenario("stress_scale")
        assert result.ticks == 3600
        assert result.runtime_seconds < 120.0, (
            f"a simulated hour took {result.runtime_seconds:.1f}s"
        )
        assert result.violations == []


class TestMetrics:
    """The performance measures themselves."""

    def test_waiting_passengers_count_towards_the_average(
        self, small_config: ScenarioConfig
    ) -> None:
        """A starving call must not be hidden by averaging only the lucky ones."""
        model = ElevatorModel(small_config)
        model.run(150)
        metrics = model.latest_metrics
        if metrics.waiting > 0:
            assert metrics.max_wait > 0

    def test_counts_are_consistent(self, small_config: ScenarioConfig) -> None:
        """arrived == delivered + waiting + riding, always."""
        model = ElevatorModel(small_config)
        for _ in range(0, 200, 25):
            model.run(25)
            m = model.latest_metrics
            assert m.arrived == m.delivered + m.waiting + m.riding

    def test_energy_only_grows(self, small_config: ScenarioConfig) -> None:
        """Energy is a cumulative counter."""
        model = ElevatorModel(small_config)
        previous = 0.0
        for _ in range(100):
            model.step()
            assert model.latest_metrics.energy >= previous
            previous = model.latest_metrics.energy
