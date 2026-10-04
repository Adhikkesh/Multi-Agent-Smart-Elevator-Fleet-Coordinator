"""Unified decision context and FleetView representation.

Provides the single bridge between raw simulation states (Mesa real sim or
vectorised twin simulator) and the feature encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elevator_mas.comms.board import CarStatus, DecisionEvent, FleetPolicy
from elevator_mas.domain import Direction


@dataclass
class FleetView:
    """Structure-of-arrays representation of fleet state for N cars (N <= MAX_CARS)."""

    n_cars: int
    floors: np.ndarray  # int16 [N]
    directions: np.ndarray  # int8 [N]: +1 UP, -1 DOWN, 0 IDLE
    loads: np.ndarray  # int16 [N]
    capacities: np.ndarray  # int16 [N]
    availables: np.ndarray  # bool [N]
    out_of_service: np.ndarray  # bool [N]
    fire_mode: np.ndarray  # bool [N]
    doors: np.ndarray  # int8 [N]: 0 closed, 1 opening, 2 open, 3 closing
    door_blocked_ticks: np.ndarray  # int16 [N]
    n_assigned: np.ndarray  # int16 [N]
    n_car_calls: np.ndarray  # int16 [N]
    riders: np.ndarray  # int16 [N]
    planned_stops: np.ndarray  # int16 [N]
    plan_end_floors: np.ndarray  # int16 [N]
    plan_end_etas: np.ndarray  # float32 [N]
    has_same_call: np.ndarray  # bool [N]
    park_targets: np.ndarray  # int16 [N]: -1 if None


@dataclass(frozen=True)
class DecisionContext:
    """Complete context of an auction / assignment decision."""

    call_floor: int
    call_direction: int  # +1 UP, -1 DOWN
    call_waiting: int
    call_urgency: int

    floors: int
    lobby: int
    capacity: int

    pattern: str
    demand: dict[int, float]
    tick: int
    open_calls_count: int

    fleet: FleetView


def _dir_to_int(d: Direction | int | str) -> int:
    if isinstance(d, Direction):
        if d is Direction.UP:
            return 1
        if d is Direction.DOWN:
            return -1
        return 0
    if isinstance(d, int):
        return 1 if d > 0 else (-1 if d < 0 else 0)
    s = str(d).upper()
    if "UP" in s:
        return 1
    if "DOWN" in s:
        return -1
    return 0


def _door_to_int(door: str | int) -> int:
    if isinstance(door, int):
        return max(0, min(3, door))
    s = str(door).lower()
    if "opening" in s:
        return 1
    if "closed" in s:
        return 0
    if "closing" in s:
        return 3
    if "open" in s:
        return 2
    return 0


def from_event(event: DecisionEvent) -> DecisionContext:
    """Build DecisionContext from a real simulator DecisionEvent."""
    statuses = event.statuses
    n_cars = len(statuses)
    building = event.building

    floors = np.zeros(n_cars, dtype=np.int16)
    directions = np.zeros(n_cars, dtype=np.int8)
    loads = np.zeros(n_cars, dtype=np.int16)
    capacities = np.zeros(n_cars, dtype=np.int16)
    availables = np.zeros(n_cars, dtype=bool)
    out_of_service = np.zeros(n_cars, dtype=bool)
    fire_mode = np.zeros(n_cars, dtype=bool)
    doors = np.zeros(n_cars, dtype=np.int8)
    door_blocked_ticks = np.zeros(n_cars, dtype=np.int16)
    n_assigned = np.zeros(n_cars, dtype=np.int16)
    n_car_calls = np.zeros(n_cars, dtype=np.int16)
    riders = np.zeros(n_cars, dtype=np.int16)
    planned_stops = np.zeros(n_cars, dtype=np.int16)
    plan_end_floors = np.zeros(n_cars, dtype=np.int16)
    plan_end_etas = np.zeros(n_cars, dtype=np.float32)
    has_same_call = np.zeros(n_cars, dtype=bool)
    park_targets = np.full(n_cars, -1, dtype=np.int16)

    call_f = event.call.floor
    call_d = 1 if event.call.direction is Direction.UP else -1

    for i, s in enumerate(statuses):
        floors[i] = s.floor
        directions[i] = _dir_to_int(s.direction)
        loads[i] = s.load
        capacities[i] = s.capacity
        availables[i] = s.available
        out_of_service[i] = s.out_of_service
        fire_mode[i] = s.fire_mode
        doors[i] = _door_to_int(s.door)
        door_blocked_ticks[i] = s.door_blocked_ticks
        n_assigned[i] = len(s.assigned_calls)
        n_car_calls[i] = len(s.car_calls)
        riders[i] = s.riders
        planned_stops[i] = s.planned_stops
        plan_end_floors[i] = s.plan_end_floor if s.plan_end_floor is not None else s.floor
        plan_end_etas[i] = s.plan_end_eta
        has_same_call[i] = any(
            c.floor == call_f and (1 if c.direction is Direction.UP else -1) == call_d
            for c in s.assigned_calls
        )
        park_targets[i] = s.park_target if s.park_target is not None else -1

    fleet = FleetView(
        n_cars=n_cars,
        floors=floors,
        directions=directions,
        loads=loads,
        capacities=capacities,
        availables=availables,
        out_of_service=out_of_service,
        fire_mode=fire_mode,
        doors=doors,
        door_blocked_ticks=door_blocked_ticks,
        n_assigned=n_assigned,
        n_car_calls=n_car_calls,
        riders=riders,
        planned_stops=planned_stops,
        plan_end_floors=plan_end_floors,
        plan_end_etas=plan_end_etas,
        has_same_call=has_same_call,
        park_targets=park_targets,
    )

    open_calls = int(np.sum(n_assigned)) + 1

    return DecisionContext(
        call_floor=call_f,
        call_direction=call_d,
        call_waiting=event.waiting,
        call_urgency=event.urgency,
        floors=building.floors,
        lobby=building.lobby,
        capacity=building.capacity,
        pattern=event.policy.pattern,
        demand=dict(event.policy.demand),
        tick=event.tick,
        open_calls_count=open_calls,
        fleet=fleet,
    )


def from_board(
    call_floor: int,
    call_direction: int,
    call_waiting: int,
    call_urgency: int,
    cars: list[CarStatus],
    policy: FleetPolicy,
    building: Any,
    tick: int,
) -> DecisionContext:
    """Build DecisionContext directly from board state."""
    from elevator_mas.domain import HallCall

    h_dir = Direction.UP if call_direction > 0 else Direction.DOWN
    evt = DecisionEvent(
        tick=tick,
        call=HallCall(floor=call_floor, direction=h_dir),
        urgency=call_urgency,
        waiting=call_waiting,
        statuses=tuple(cars),
        policy=policy,
        bids=(),
        winner=None,
        building=building,
        seed=0,
    )
    return from_event(evt)
