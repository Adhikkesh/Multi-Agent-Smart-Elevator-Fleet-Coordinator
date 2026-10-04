"""Tests for feature encoder, shapes, ranges, padding, permutation, and truth tables."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st

from elevator_mas.comms.board import CarStatus, DecisionEvent, FleetPolicy
from elevator_mas.config import BuildingConfig, CostWeights
from elevator_mas.domain import Bid, Direction, HallCall
from elevator_mas.learning.features import encode_batch, encode_decision
from elevator_mas.learning.schema import KC, KCAR, KG, MAX_CARS
from elevator_mas.learning.view import DecisionContext, FleetView, from_event


def make_dummy_context(
    n_cars: int = 4,
    floors: int = 15,
    call_floor: int = 7,
    call_dir: int = 1,
) -> DecisionContext:
    car_floors = np.array([i % floors for i in range(n_cars)], dtype=np.int16)
    directions = np.array([1 if i % 2 == 0 else -1 for i in range(n_cars)], dtype=np.int8)
    loads = np.array([i % 5 for i in range(n_cars)], dtype=np.int16)
    capacities = np.full(n_cars, 10, dtype=np.int16)
    availables = np.ones(n_cars, dtype=bool)
    oos = np.zeros(n_cars, dtype=bool)
    fire = np.zeros(n_cars, dtype=bool)
    doors = np.zeros(n_cars, dtype=np.int8)
    blocked = np.zeros(n_cars, dtype=np.int16)
    n_assigned = np.array([i % 3 for i in range(n_cars)], dtype=np.int16)
    n_car_calls = np.array([i % 2 for i in range(n_cars)], dtype=np.int16)
    riders = loads.copy()
    planned_stops = np.array([(i % 4) + 1 for i in range(n_cars)], dtype=np.int16)
    plan_end_floors = np.array([(i * 3) % floors for i in range(n_cars)], dtype=np.int16)
    plan_end_etas = np.array([float(i * 5) for i in range(n_cars)], dtype=np.float32)
    has_same = np.zeros(n_cars, dtype=bool)
    park_targets = np.full(n_cars, -1, dtype=np.int16)

    fleet = FleetView(
        n_cars=n_cars,
        floors=car_floors,
        directions=directions,
        loads=loads,
        capacities=capacities,
        availables=availables,
        out_of_service=oos,
        fire_mode=fire,
        doors=doors,
        door_blocked_ticks=blocked,
        n_assigned=n_assigned,
        n_car_calls=n_car_calls,
        riders=riders,
        planned_stops=planned_stops,
        plan_end_floors=plan_end_floors,
        plan_end_etas=plan_end_etas,
        has_same_call=has_same,
        park_targets=park_targets,
    )

    return DecisionContext(
        call_floor=call_floor,
        call_direction=call_dir,
        call_waiting=3,
        call_urgency=1,
        floors=floors,
        lobby=0,
        capacity=10,
        pattern="two_way",
        demand={0: 0.5, 7: 0.5},
        tick=120,
        open_calls_count=4,
        fleet=fleet,
    )


def test_encoded_shapes_and_types() -> None:
    ctx = make_dummy_context(n_cars=4)
    enc = encode_decision(ctx)

    assert enc.call.shape == (KC,)
    assert enc.call.dtype == np.float32
    assert enc.cars.shape == (MAX_CARS, KCAR)
    assert enc.cars.dtype == np.float32
    assert enc.glob.shape == (KG,)
    assert enc.glob.dtype == np.float32
    assert enc.mask.shape == (MAX_CARS,)
    assert enc.mask.dtype == bool
    assert enc.eligible.shape == (MAX_CARS,)
    assert enc.eligible.dtype == bool

    # No NaN or Inf anywhere
    assert not np.isnan(enc.call).any()
    assert not np.isnan(enc.cars).any()
    assert not np.isnan(enc.glob).any()
    assert not np.isinf(enc.call).any()
    assert not np.isinf(enc.cars).any()
    assert not np.isinf(enc.glob).any()


def test_padding_is_zero_and_masked() -> None:
    n_cars = 3
    ctx = make_dummy_context(n_cars=n_cars)
    enc = encode_decision(ctx)

    for i in range(MAX_CARS):
        if i < n_cars:
            assert enc.mask[i] is True or enc.mask[i] == 1
        else:
            assert enc.mask[i] is False or enc.mask[i] == 0
            assert enc.eligible[i] is False or enc.eligible[i] == 0
            assert np.all(enc.cars[i] == 0.0)


def test_route_truth_tables() -> None:
    """Test heading_to_call (#22) and call_on_route (#23) on explicit cases."""
    # Case A: Car at 3 moving UP to 10. Call is at 6 UP.
    # Should be heading_to_call = 1, call_on_route = 1.
    fleet = FleetView(
        n_cars=1,
        floors=np.array([3], dtype=np.int16),
        directions=np.array([1], dtype=np.int8),
        loads=np.array([0], dtype=np.int16),
        capacities=np.array([10], dtype=np.int16),
        availables=np.array([True]),
        out_of_service=np.array([False]),
        fire_mode=np.array([False]),
        doors=np.array([0], dtype=np.int8),
        door_blocked_ticks=np.array([0], dtype=np.int16),
        n_assigned=np.array([0], dtype=np.int16),
        n_car_calls=np.array([0], dtype=np.int16),
        riders=np.array([0], dtype=np.int16),
        planned_stops=np.array([1], dtype=np.int16),
        plan_end_floors=np.array([10], dtype=np.int16),
        plan_end_etas=np.array([10.0], dtype=np.float32),
        has_same_call=np.array([False]),
        park_targets=np.array([-1], dtype=np.int16),
    )
    ctx_up_match = DecisionContext(
        call_floor=6,
        call_direction=1,
        call_waiting=1,
        call_urgency=0,
        floors=15,
        lobby=0,
        capacity=10,
        pattern="interfloor",
        demand={},
        tick=50,
        open_calls_count=1,
        fleet=fleet,
    )
    enc = encode_decision(ctx_up_match)
    assert enc.cars[0, 22] == 1.0  # heading to call
    assert enc.cars[0, 23] == 1.0  # call on route

    # Case B: Same car at 3 moving UP to 10. Call is at 6 DOWN.
    # Heading to call is still 1, but call_on_route must be 0 (wrong direction!).
    ctx_up_wrong_dir = DecisionContext(
        call_floor=6,
        call_direction=-1,
        call_waiting=1,
        call_urgency=0,
        floors=15,
        lobby=0,
        capacity=10,
        pattern="interfloor",
        demand={},
        tick=50,
        open_calls_count=1,
        fleet=fleet,
    )
    enc_wrong_dir = encode_decision(ctx_up_wrong_dir)
    assert enc_wrong_dir.cars[0, 22] == 1.0
    assert enc_wrong_dir.cars[0, 23] == 0.0

    # Case C: Car at 8 moving UP to 10. Call is behind at 2 UP.
    # Moving away: heading_to_call = -1, call_on_route = 0.
    ctx_behind = DecisionContext(
        call_floor=2,
        call_direction=1,
        call_waiting=1,
        call_urgency=0,
        floors=15,
        lobby=0,
        capacity=10,
        pattern="interfloor",
        demand={},
        tick=50,
        open_calls_count=1,
        fleet=fleet,
    )
    enc_behind = encode_decision(ctx_behind)
    assert enc_behind.cars[0, 22] == -1.0
    assert enc_behind.cars[0, 23] == 0.0


def test_permutation_invariance_of_token_set() -> None:
    """Shuffling car order permutes the car token rows identically."""
    ctx = make_dummy_context(n_cars=4)
    enc1 = encode_decision(ctx)

    # Permute cars
    perm = np.array([2, 0, 3, 1])
    perm_fleet = FleetView(
        n_cars=4,
        floors=ctx.fleet.floors[perm],
        directions=ctx.fleet.directions[perm],
        loads=ctx.fleet.loads[perm],
        capacities=ctx.fleet.capacities[perm],
        availables=ctx.fleet.availables[perm],
        out_of_service=ctx.fleet.out_of_service[perm],
        fire_mode=ctx.fleet.fire_mode[perm],
        doors=ctx.fleet.doors[perm],
        door_blocked_ticks=ctx.fleet.door_blocked_ticks[perm],
        n_assigned=ctx.fleet.n_assigned[perm],
        n_car_calls=ctx.fleet.n_car_calls[perm],
        riders=ctx.fleet.riders[perm],
        planned_stops=ctx.fleet.planned_stops[perm],
        plan_end_floors=ctx.fleet.plan_end_floors[perm],
        plan_end_etas=ctx.fleet.plan_end_etas[perm],
        has_same_call=ctx.fleet.has_same_call[perm],
        park_targets=ctx.fleet.park_targets[perm],
    )
    ctx2 = DecisionContext(
        call_floor=ctx.call_floor,
        call_direction=ctx.call_direction,
        call_waiting=ctx.call_waiting,
        call_urgency=ctx.call_urgency,
        floors=ctx.floors,
        lobby=ctx.lobby,
        capacity=ctx.capacity,
        pattern=ctx.pattern,
        demand=ctx.demand,
        tick=ctx.tick,
        open_calls_count=ctx.open_calls_count,
        fleet=perm_fleet,
    )
    enc2 = encode_decision(ctx2)

    # Call and global tokens must be identical
    np.testing.assert_array_equal(enc1.call, enc2.call)
    np.testing.assert_array_equal(enc1.glob, enc2.glob)

    # Cars tokens permuted identically
    for old_i, new_i in enumerate(perm):
        np.testing.assert_array_equal(enc1.cars[new_i], enc2.cars[old_i])
        assert enc1.eligible[new_i] == enc2.eligible[old_i]


def test_parity_real_event_vs_fleet_view() -> None:
    """Build a fleet state both via real CarStatus objects and FleetView: bit-identical."""
    statuses = (
        CarStatus(
            car_id=0,
            address="car-0",
            tick=100,
            floor=3,
            direction=Direction.UP,
            load=4,
            capacity=10,
            space=6,
            available=True,
            out_of_service=False,
            fire_mode=False,
            door="closed",
            door_blocked_ticks=0,
            assigned_calls=frozenset([HallCall(floor=8, direction=Direction.UP)]),
            car_calls=frozenset([7]),
            riders=4,
            plan_end_floor=8,
            plan_end_eta=12.5,
            planned_stops=2,
            park_target=None,
        ),
        CarStatus(
            car_id=1,
            address="car-1",
            tick=100,
            floor=11,
            direction=Direction.DOWN,
            load=10,
            capacity=10,
            space=0,
            available=False,
            out_of_service=False,
            fire_mode=False,
            door="open",
            door_blocked_ticks=2,
            assigned_calls=frozenset(),
            car_calls=frozenset([0]),
            riders=10,
            plan_end_floor=0,
            plan_end_eta=25.0,
            planned_stops=3,
            park_target=0,
        ),
    )
    policy = FleetPolicy(pattern="up_peak", demand={0: 0.8, 5: 0.2}, tick=100)
    building = BuildingConfig(floors=16, lobby=0, capacity=10, cars=2)
    evt = DecisionEvent(
        tick=100,
        call=HallCall(floor=5, direction=Direction.UP),
        urgency=2,
        waiting=4,
        statuses=statuses,
        policy=policy,
        bids=(),
        winner=0,
        building=building,
        seed=42,
    )
    ctx1 = from_event(evt)
    enc1 = encode_decision(ctx1)

    # Equivalent direct FleetView
    fleet2 = FleetView(
        n_cars=2,
        floors=np.array([3, 11], dtype=np.int16),
        directions=np.array([1, -1], dtype=np.int8),
        loads=np.array([4, 10], dtype=np.int16),
        capacities=np.array([10, 10], dtype=np.int16),
        availables=np.array([True, False]),
        out_of_service=np.array([False, False]),
        fire_mode=np.array([False, False]),
        doors=np.array([0, 2], dtype=np.int8),
        door_blocked_ticks=np.array([0, 2], dtype=np.int16),
        n_assigned=np.array([1, 0], dtype=np.int16),
        n_car_calls=np.array([1, 1], dtype=np.int16),
        riders=np.array([4, 10], dtype=np.int16),
        planned_stops=np.array([2, 3], dtype=np.int16),
        plan_end_floors=np.array([8, 0], dtype=np.int16),
        plan_end_etas=np.array([12.5, 25.0], dtype=np.float32),
        has_same_call=np.array([False, False]),
        park_targets=np.array([-1, 0], dtype=np.int16),
    )
    ctx2 = DecisionContext(
        call_floor=5,
        call_direction=1,
        call_waiting=4,
        call_urgency=2,
        floors=16,
        lobby=0,
        capacity=10,
        pattern="up_peak",
        demand={0: 0.8, 5: 0.2},
        tick=100,
        open_calls_count=2,
        fleet=fleet2,
    )
    enc2 = encode_decision(ctx2)

    # Parity assertion: bit-identical
    np.testing.assert_array_equal(enc1.call, enc2.call)
    np.testing.assert_array_equal(enc1.cars, enc2.cars)
    np.testing.assert_array_equal(enc1.glob, enc2.glob)
    np.testing.assert_array_equal(enc1.mask, enc2.mask)
    np.testing.assert_array_equal(enc1.eligible, enc2.eligible)


@given(
    floors=st.integers(min_value=4, max_value=40),
    n_cars=st.integers(min_value=1, max_value=MAX_CARS),
    call_floor=st.integers(min_value=0, max_value=39),
    call_dir=st.sampled_from([-1, 1]),
)
def test_hypothesis_feature_bounds(floors: int, n_cars: int, call_floor: int, call_dir: int) -> None:
    c_fl = min(call_floor, floors - 1)
    ctx = make_dummy_context(n_cars=n_cars, floors=floors, call_floor=c_fl, call_dir=call_dir)
    enc = encode_decision(ctx)

    assert not np.isnan(enc.call).any()
    assert not np.isnan(enc.cars).any()
    assert not np.isnan(enc.glob).any()

    # Call and global token in [0, 1] or [-1, 1]
    assert np.all(enc.call >= -1.0) and np.all(enc.call <= 1.0)
    assert np.all(enc.glob >= 0.0) and np.all(enc.glob <= 1.0)
