"""The passenger: a simple reflex agent."""

from __future__ import annotations

from typing import Any

from mesa import Agent

from elevator_mas.domain import Direction, PassengerRecord


class PassengerAgent(Agent):
    """A person waiting for, riding in, and leaving a lift.

    **AIMA agent type: simple reflex agent** (§2.4.2). It keeps no model of the building
    and does no planning: it maps its current percept straight to an action through
    condition-action rules — *if a car is here, its doors are open, it serves my
    direction and it has room, then board*. That is the whole policy. It is the right
    type for this role, and it is also what makes the environment *partially observable*
    for the other agents: the passenger's destination is a private fact until they board
    and press a car button.

    PEAS
      - **P** its own wait time, ride time, and being delivered at all.
      - **E** the floor it waits on, the hall buttons, and the arriving cars.
      - **A** press a hall button, board, press a car button, alight.
      - **S** which car is at its floor, that car's direction, its door state and
        whether there is space.
    """

    agent_type = "Simple reflex agent"
    peas = {
        "performance": "own wait time, ride time, being delivered",
        "environment": "its floor, hall buttons, arriving cars",
        "actuators": "press hall button, board, press car button, alight",
        "sensors": "car presence, car direction, door state, remaining space",
    }

    def __init__(
        self,
        model: Any,
        origin: int,
        destination: int,
        weight: float = 1.0,
        priority: bool = False,
    ) -> None:
        super().__init__(model)
        if origin == destination:
            raise ValueError("a passenger must actually be going somewhere")
        self.origin = origin
        self.destination = destination
        self.weight = weight
        self.priority = priority
        self.current_floor = origin
        self.boarded = False
        self.car_id: int | None = None
        self.record = PassengerRecord(
            passenger_id=self.unique_id,
            origin=origin,
            destination=destination,
            arrival_tick=model.tick,
            weight=weight,
            priority=priority,
        )

    @property
    def direction(self) -> Direction:
        """The direction this passenger wants to travel."""
        return Direction.UP if self.destination > self.origin else Direction.DOWN

    @property
    def waiting(self) -> bool:
        """True while still on a landing, not yet in a car."""
        return not self.boarded and not self.record.delivered

    def board(self, car_id: int, tick: int) -> None:
        """Board a car and press the car button, revealing the destination."""
        self.boarded = True
        self.car_id = car_id
        self.record.board_tick = tick
        self.record.car_id = car_id

    def alight(self, tick: int) -> None:
        """Leave the car at the destination floor."""
        self.boarded = False
        self.current_floor = self.destination
        self.record.alight_tick = tick

    def requeue(self, floor: int, tick: int) -> None:
        """Be turned out of a failed car and start waiting again.

        The original arrival time is deliberately kept, so a passenger caught in a
        breakdown is not silently given a fresh, flattering wait time.
        """
        del tick
        self.boarded = False
        self.car_id = None
        self.origin = floor
        self.current_floor = floor
        self.record.origin = floor
        self.record.board_tick = None
        self.record.car_id = None
        self.record.requeued += 1

    def step(self) -> None:
        """Mesa's default hook; this agent is driven by the model's staged activation."""
