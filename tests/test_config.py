"""Configuration, the strategy registry and the metric helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elevator_mas.config import (
    ArrivalPhase,
    BuildingConfig,
    ScenarioConfig,
    TrafficConfig,
    available_scenarios,
)
from elevator_mas.domain import Direction, HallCall, PassengerRecord, Stop, StopKind
from elevator_mas.metrics import MetricsCollector, percentile
from elevator_mas.strategies import get_strategy, strategy_names


class TestConfig:
    """Validation must catch a broken scenario file at load time."""

    def test_defaults_match_the_specification(self) -> None:
        """15 floors, 4 cars, capacity 10, 2 s/floor, 2 s doors."""
        config = ScenarioConfig(name="d")
        assert config.building.floors == 15
        assert config.building.cars == 4
        assert config.building.capacity == 10
        assert config.timing.seconds_per_floor == 2
        assert config.timing.door_open == 2
        assert config.timing.door_close == 2
        assert config.timing.boarding_per_passenger == 1

    def test_every_bundled_scenario_parses(self) -> None:
        """A malformed YAML must not reach the model."""
        for name in available_scenarios():
            config = ScenarioConfig.load(name)
            assert config.name == name
            assert config.duration > 0

    def test_a_lobby_outside_the_building_is_rejected(self) -> None:
        """Cross-field validation."""
        with pytest.raises(ValidationError):
            BuildingConfig(floors=5, lobby=9)

    def test_impossible_sizes_are_rejected(self) -> None:
        """Bounds on floors and cars."""
        with pytest.raises(ValidationError):
            BuildingConfig(floors=1)
        with pytest.raises(ValidationError):
            BuildingConfig(cars=0)

    def test_unknown_scenario_names_the_alternatives(self) -> None:
        """A helpful error, not a bare KeyError."""
        with pytest.raises(FileNotFoundError, match="available"):
            ScenarioConfig.load("no_such_scenario")

    def test_traffic_phases_are_time_ordered(self) -> None:
        """The right phase must apply at the right tick."""
        traffic = TrafficConfig(
            phases=[
                ArrivalPhase(until=100, rate=0.1, pattern="up_peak"),
                ArrivalPhase(until=200, rate=0.3, pattern="down_peak"),
            ]
        )
        assert traffic.pattern_at(50) == "up_peak"
        assert traffic.pattern_at(150) == "down_peak"
        assert traffic.rate_at(50) == 0.1
        assert traffic.rate_at(150) == 0.3
        assert traffic.rate_at(500) == 0.0  # past the last phase


class TestStrategies:
    """The registry."""

    def test_the_strategy_ladder_is_registered(self) -> None:
        """The classical comparison ladder from the specification, then the learned bidders."""
        assert strategy_names() == [
            "nearest_car",
            "collective",
            "cnp_astar",
            "full",
            "liftzero_bc",
            "liftzero_bc_cnp",
        ]

    def test_the_full_strategy_enables_everything(self) -> None:
        """It is the union of every mechanism."""
        full = get_strategy("full")
        assert full.uses_auction
        assert full.uses_reassignment
        assert full.uses_smart_parking
        assert full.adapts_weights
        assert full.routing == "astar"

    def test_the_baseline_enables_nothing(self) -> None:
        """NearestCar is a pure reflex control."""
        baseline = get_strategy("nearest_car")
        assert not baseline.uses_auction
        assert not baseline.uses_reassignment
        assert not baseline.uses_smart_parking
        assert baseline.routing == "look"

    def test_unknown_strategy_names_the_alternatives(self) -> None:
        """Helpful failure."""
        with pytest.raises(ValueError, match="available"):
            get_strategy("nope")


class TestDomain:
    """The value types."""

    def test_a_pickup_needs_a_direction(self) -> None:
        """A hall call without a direction is meaningless."""
        with pytest.raises(ValueError, match="direction"):
            Stop(3, StopKind.PICKUP, Direction.IDLE)

    def test_a_hall_call_cannot_be_idle(self) -> None:
        """Likewise for the call itself."""
        with pytest.raises(ValueError, match="UP or DOWN"):
            HallCall(3, Direction.IDLE)

    def test_passenger_timings(self) -> None:
        """Wait, ride and system time."""
        record = PassengerRecord(1, 0, 5, arrival_tick=10)
        assert record.wait_time is None
        record.board_tick = 30
        assert record.wait_time == 20
        assert record.ride_time is None
        record.alight_tick = 70
        assert record.ride_time == 40
        assert record.system_time == 60
        assert record.delivered

    def test_direction_is_inferred_from_the_trip(self) -> None:
        """Up if the destination is higher."""
        assert PassengerRecord(1, 0, 5, 0).direction is Direction.UP
        assert PassengerRecord(2, 5, 0, 0).direction is Direction.DOWN


class TestMetrics:
    """The performance measures."""

    def test_percentile(self) -> None:
        """Linear interpolation, and a safe answer for an empty sample."""
        values = [float(i) for i in range(1, 101)]
        assert percentile(values, 0.95) == pytest.approx(95.05)
        assert percentile(values, 0.5) == pytest.approx(50.5)
        assert percentile([], 0.95) == 0.0
        assert percentile([7.0], 0.95) == 7.0

    def test_waiting_passengers_are_counted(self) -> None:
        """Someone still waiting must appear in the average, not be ignored."""
        collector = MetricsCollector(long_wait_threshold=60)
        delivered = PassengerRecord(1, 0, 5, arrival_tick=0)
        delivered.board_tick = 10
        delivered.alight_tick = 40
        collector.register(delivered)
        collector.register(PassengerRecord(2, 3, 0, arrival_tick=0))  # still waiting

        metrics = collector.snapshot(tick=100)
        assert metrics.arrived == 2
        assert metrics.delivered == 1
        assert metrics.waiting == 1
        # (10 boarded + 100 still waiting) / 2
        assert metrics.avg_wait == pytest.approx(55.0)
        assert metrics.max_wait == pytest.approx(100.0)
        assert metrics.long_wait_pct == pytest.approx(50.0)

    def test_energy_weighting(self) -> None:
        """Floors, stops and reversals all contribute."""
        collector = MetricsCollector()
        collector.energy.floors_travelled = 10
        collector.energy.stops = 4
        collector.energy.reversals = 2
        assert collector.energy.total == pytest.approx(10 * 1.0 + 4 * 2.0 + 2 * 1.0)
