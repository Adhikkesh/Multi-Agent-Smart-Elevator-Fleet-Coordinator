"""The safety rule base, as declarative productions.

These are the rules the SafetyAgent reasons with. They are written as data so that the
knowledge and the inference are separated the way AIMA describes a knowledge-based agent
(§7.1): the engine in `engine.py` knows nothing about elevators, and these rules contain
no control flow.

The same rules are mirrored as Horn clauses in `docs/safety_rules.pl`, which can be run
in Prolog to check they entail the same conclusions.

Salience layers, highest first:

    100  fire recall        — life safety overrides everything
     90  fire door hold     — cars that have arrived at the lobby stay open
     80  block hall calls   — no new pickups while in fire mode
     70  car fault          — take a broken car out of service and re-auction its calls
     60  overload           — hold the doors and refuse further boarding
     50  door obstruction   — a door blocked too long re-opens
     40  fire clear         — restore normal service once the alarm is reset
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from elevator_mas.rules.engine import Fact, ForwardChainingEngine, Rule


def _fire_recall_condition(wm: set[Fact], world: Any) -> list[dict[str, Any]]:
    """Fire alarm active and some car has not yet been told to recall."""
    if ("fire_alarm", True) not in wm:
        return []
    return [
        {"car": car.car_id}
        for car in world.cars
        if not car.out_of_service and ("recalling", car.car_id) not in wm
    ]


def _fire_recall_conclude(binding: dict[str, Any], world: Any) -> Iterable[Fact]:
    """Assert that this car is now recalling."""
    del world
    return [("recalling", binding["car"])]


def _fire_recall_act(binding: dict[str, Any], world: Any) -> str:
    """Send the car to the lobby in fire mode, dropping its assignments."""
    car = world.car(binding["car"])
    car.enter_fire_mode(world.lobby)
    return f"car {car.car_id} recalled to lobby {world.lobby}"


def _fire_doors_condition(wm: set[Fact], world: Any) -> list[dict[str, Any]]:
    """A recalling car that has reached the lobby must sit with its doors open."""
    if ("fire_alarm", True) not in wm:
        return []
    return [
        {"car": car.car_id}
        for car in world.cars
        if car.fire_mode and car.floor == world.lobby and ("doors_held_open", car.car_id) not in wm
    ]


def _fire_doors_conclude(binding: dict[str, Any], world: Any) -> Iterable[Fact]:
    """Record that this car's doors are held open."""
    del world
    return [("doors_held_open", binding["car"])]


def _fire_doors_act(binding: dict[str, Any], world: Any) -> str:
    """Hold the doors open so occupants can leave."""
    car = world.car(binding["car"])
    car.hold_doors_open()
    return f"car {car.car_id} holding doors open at lobby"


def _block_calls_condition(wm: set[Fact], world: Any) -> list[dict[str, Any]]:
    """Fire mode blocks new hall calls exactly once."""
    del world
    if ("fire_alarm", True) not in wm or ("hall_calls_blocked", True) in wm:
        return []
    return [{}]


def _block_calls_conclude(binding: dict[str, Any], world: Any) -> Iterable[Fact]:
    """Record the block."""
    del binding, world
    return [("hall_calls_blocked", True)]


def _block_calls_act(binding: dict[str, Any], world: Any) -> str:
    """Stop accepting hall calls and clear the ones outstanding."""
    del binding
    world.block_hall_calls()
    return "hall calls blocked; pending calls cleared"


def _fault_condition(wm: set[Fact], world: Any) -> list[dict[str, Any]]:
    """A faulted car that has not yet been taken out of service."""
    return [
        {"car": car_id}
        for (kind, car_id) in list(wm)
        if kind == "car_fault" and ("out_of_service", car_id) not in wm
        for car in [world.car(car_id)]
        if car is not None
    ]


def _fault_conclude(binding: dict[str, Any], world: Any) -> Iterable[Fact]:
    """Record the out-of-service status."""
    del world
    return [("out_of_service", binding["car"])]


def _fault_act(binding: dict[str, Any], world: Any) -> str:
    """Stop the car at the next floor, turn out its riders and re-auction its calls."""
    car_id = binding["car"]
    reassigned = world.take_out_of_service(car_id)
    return f"car {car_id} out of service; {reassigned} call(s) re-auctioned"


def _overload_condition(wm: set[Fact], world: Any) -> list[dict[str, Any]]:
    """A car carrying more than its capacity."""
    del wm
    return [
        {"car": car.car_id, "load": car.load, "capacity": car.capacity}
        for car in world.cars
        if car.load > car.capacity
    ]


def _overload_conclude(binding: dict[str, Any], world: Any) -> Iterable[Fact]:
    """Record the overload."""
    del world
    return [("overloaded", binding["car"])]


def _overload_act(binding: dict[str, Any], world: Any) -> str:
    """Hold the doors and refuse further boarding until the load is legal."""
    car = world.car(binding["car"])
    car.refuse_boarding()
    return (
        f"car {car.car_id} overloaded ({binding['load']}>{binding['capacity']}): boarding refused"
    )


def _obstruction_condition(wm: set[Fact], world: Any) -> list[dict[str, Any]]:
    """Doors that have been blocked longer than the tolerated time."""
    del wm
    return [
        {"car": car.car_id, "blocked": car.door_blocked_ticks}
        for car in world.cars
        if car.door_blocked_ticks > world.door_obstruction_limit
    ]


def _obstruction_act(binding: dict[str, Any], world: Any) -> str:
    """Re-open the doors and restart the dwell."""
    car = world.car(binding["car"])
    car.reopen_doors()
    return f"car {car.car_id} door obstruction cleared by re-opening"


def _fire_clear_condition(wm: set[Fact], world: Any) -> list[dict[str, Any]]:
    """The alarm has been reset but cars are still in fire mode."""
    del world
    if ("fire_alarm", True) in wm or ("fire_cleared", True) in wm:
        return []
    if not any(kind == "recalling" for (kind, *_rest) in wm):
        return []
    return [{}]


def _fire_clear_conclude(binding: dict[str, Any], world: Any) -> Iterable[Fact]:
    """Record that normal service has been restored."""
    del binding, world
    return [("fire_cleared", True)]


def _fire_clear_act(binding: dict[str, Any], world: Any) -> str:
    """Restore normal service across the fleet."""
    del binding
    world.restore_normal_service()
    return "fire alarm cleared; normal service restored"


SAFETY_RULES: list[Rule] = [
    Rule(
        name="R1_fire_recall",
        salience=100,
        condition=_fire_recall_condition,
        conclude=_fire_recall_conclude,
        act=_fire_recall_act,
        description="IF fire_alarm AND car in service THEN recall car to the lobby",
    ),
    Rule(
        name="R2_fire_doors_open",
        salience=90,
        condition=_fire_doors_condition,
        conclude=_fire_doors_conclude,
        act=_fire_doors_act,
        description="IF fire_alarm AND car at lobby THEN hold its doors open",
    ),
    Rule(
        name="R3_block_hall_calls",
        salience=80,
        condition=_block_calls_condition,
        conclude=_block_calls_conclude,
        act=_block_calls_act,
        description="IF fire_alarm THEN block all hall calls",
    ),
    Rule(
        name="R4_car_fault_out_of_service",
        salience=70,
        condition=_fault_condition,
        conclude=_fault_conclude,
        act=_fault_act,
        description="IF car_fault THEN out of service AND re-auction its calls",
    ),
    Rule(
        name="R5_overload_refuse_boarding",
        salience=60,
        condition=_overload_condition,
        conclude=_overload_conclude,
        act=_overload_act,
        description="IF load > capacity THEN hold doors AND refuse boarding",
    ),
    Rule(
        name="R6_door_obstruction",
        salience=50,
        condition=_obstruction_condition,
        act=_obstruction_act,
        description="IF doors blocked too long THEN re-open the doors",
    ),
    Rule(
        name="R7_fire_cleared",
        salience=40,
        condition=_fire_clear_condition,
        conclude=_fire_clear_conclude,
        act=_fire_clear_act,
        description="IF NOT fire_alarm AND cars recalling THEN restore normal service",
    ),
]


def build_safety_engine() -> ForwardChainingEngine:
    """A fresh engine loaded with the safety rule base."""
    return ForwardChainingEngine(SAFETY_RULES)
