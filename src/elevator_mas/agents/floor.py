"""The floor (landing) controller: a model-based reflex agent."""

from __future__ import annotations

from typing import Any

from elevator_mas.agents.base import CommunicatingAgent
from elevator_mas.comms import Performative
from elevator_mas.domain import Direction, HallCall


class FloorAgent(CommunicatingAgent):
    """One landing: its hall buttons, its waiting crowd, and its hall lanterns.

    **AIMA agent type: model-based reflex agent** (§2.4.3). A simple reflex agent cannot
    do this job, because the correct action depends on history that is not in the current
    percept: whether a call has *already* been requested, and how long it has been
    outstanding. So this agent keeps internal state — `active_calls`, `call_since`,
    `assigned_car` — and updates it from percepts, which is exactly the model-based
    extension.

    That internal state also provides the fairness guarantee. A pure lowest-bid
    dispatcher can starve an unpopular floor indefinitely, because there is always a
    cheaper call somewhere else. This agent re-escalates a call that has gone unserved
    for longer than `escalate_after` seconds, with rising urgency, and the dispatcher
    turns that urgency into a bid discount. No call can be ignored forever.

    PEAS
      - **P** how long its callers wait; whether any call is starved.
      - **E** the landing, its waiting passengers, the shafts and the cars serving it.
      - **A** hall button lamps, hall lanterns/displays, REQUEST messages.
      - **S** hall buttons, the occupancy sensor (waiting count), car arrivals, messages.
    """

    agent_type = "Model-based reflex agent"
    peas = {
        "performance": "wait time of its callers, no starved call",
        "environment": "the landing, waiting passengers, shafts, cars",
        "actuators": "hall lamps, hall lanterns/displays, REQUEST messages",
        "sensors": "hall buttons, occupancy sensor, car arrivals, messages",
    }

    def __init__(self, model: Any, floor: int) -> None:
        super().__init__(model, address=f"floor-{floor}")
        self.floor = floor
        # --- internal model of this landing ---
        self.active_calls: dict[Direction, bool] = {Direction.UP: False, Direction.DOWN: False}
        self.call_since: dict[Direction, int] = {}
        self.assigned_car: dict[Direction, int | None] = {
            Direction.UP: None,
            Direction.DOWN: None,
        }
        self.eta: dict[Direction, float | None] = {Direction.UP: None, Direction.DOWN: None}
        self.escalations: dict[Direction, int] = {Direction.UP: 0, Direction.DOWN: 0}
        self.waiting_counts: dict[Direction, int] = {Direction.UP: 0, Direction.DOWN: 0}
        self._pending_requests: list[tuple[Direction, int]] = []

    # ------------------------------------------------------------------ sensing

    def sense(self) -> None:
        """Read the occupancy sensor and the inbox, and update the internal model.

        The occupancy sensor reports *how many* people are waiting in each direction but
        not where any of them is going — the partial observability the dispatcher has to
        plan around.
        """
        self.collect_mail()
        waiting = self.model.waiting_at(self.floor)
        self.waiting_counts = {
            Direction.UP: sum(1 for p in waiting if p.direction is Direction.UP),
            Direction.DOWN: sum(1 for p in waiting if p.direction is Direction.DOWN),
        }

        for message in self.inbox:
            content = message.content
            if message.performative is Performative.INFORM:
                direction = content.get("direction")
                if isinstance(direction, Direction):
                    self.assigned_car[direction] = content.get("car_id")
                    self.eta[direction] = content.get("eta")
            elif message.performative in (Performative.FAILURE, Performative.CANCEL):
                direction = content.get("direction")
                if isinstance(direction, Direction):
                    # Our car is gone: forget the lantern and let the call be re-raised.
                    self.assigned_car[direction] = None
                    self.eta[direction] = None

        # A new button press: someone is waiting in a direction with no live call.
        for direction, count in self.waiting_counts.items():
            if count > 0 and not self.active_calls[direction]:
                self.active_calls[direction] = True
                self.call_since[direction] = self.model.tick
                self.escalations[direction] = 0
                self._pending_requests.append((direction, 0))
            elif count > 0 and self.assigned_car[direction] is None:
                # The call is live but nobody is coming: the car that was assigned has
                # been and gone (it filled up, or it was reassigned or failed) while
                # people kept arriving. Without this the landing would stay silent and
                # the queue would grow forever, because the button is already "pressed".
                self._pending_requests.append((direction, self.escalations[direction]))

        # Fairness / aging: re-escalate a call that has waited too long.
        threshold = self.model.config.fairness.escalate_after
        for direction, active in self.active_calls.items():
            if not active or self.waiting_counts[direction] == 0:
                continue
            age = self.model.tick - self.call_since.get(direction, self.model.tick)
            escalations_due = age // threshold
            if escalations_due > self.escalations[direction]:
                self.escalations[direction] = int(escalations_due)
                self._pending_requests.append((direction, int(escalations_due)))

        # Clear a call once nobody is waiting in that direction any more.
        for direction, count in self.waiting_counts.items():
            if count == 0 and self.active_calls[direction]:
                self.active_calls[direction] = False
                self.assigned_car[direction] = None
                self.eta[direction] = None
                self.escalations[direction] = 0
                self.call_since.pop(direction, None)

    # ------------------------------------------------------------ communication

    def communicate(self) -> None:
        """REQUEST service from the dispatcher for each new or escalated call."""
        for direction, urgency in self._pending_requests:
            call = HallCall(self.floor, direction)
            self.send(
                Performative.REQUEST,
                receiver="dispatcher",
                conversation_id=self.model.new_conversation_id(),
                content={
                    "call": call,
                    "floor": self.floor,
                    "direction": direction,
                    "waiting": self.waiting_counts[direction],
                    "urgency": urgency,
                    "weight": self.total_weight(direction),
                    "since": self.call_since.get(direction, self.model.tick),
                },
            )
        self._pending_requests.clear()

    # --------------------------------------------------------------- reporting

    def total_weight(self, direction: Direction) -> float:
        """Summed passenger weight waiting in a direction (VIPs count for more)."""
        return sum(p.weight for p in self.model.waiting_at(self.floor) if p.direction is direction)

    def oldest_wait(self) -> int:
        """Ticks the longest-waiting passenger on this floor has been here."""
        waiting = self.model.waiting_at(self.floor)
        if not waiting:
            return 0
        return max(self.model.tick - p.record.arrival_tick for p in waiting)

    def clear_calls(self) -> None:
        """Drop every hall call: used when the SafetyAgent puts the fleet in fire mode."""
        for direction in (Direction.UP, Direction.DOWN):
            self.active_calls[direction] = False
            self.assigned_car[direction] = None
            self.eta[direction] = None
            self.escalations[direction] = 0
        self.call_since.clear()
        self._pending_requests.clear()

    def snapshot(self) -> dict[str, Any]:
        """What the dashboard draws for this landing."""
        return {
            "floor": self.floor,
            "up": self.active_calls[Direction.UP],
            "down": self.active_calls[Direction.DOWN],
            "waiting_up": self.waiting_counts[Direction.UP],
            "waiting_down": self.waiting_counts[Direction.DOWN],
            "assigned_up": self.assigned_car[Direction.UP],
            "assigned_down": self.assigned_car[Direction.DOWN],
            "eta_up": self.eta[Direction.UP],
            "eta_down": self.eta[Direction.DOWN],
            "escalations": max(self.escalations.values()),
            "oldest_wait": self.oldest_wait(),
        }

    def describe(self) -> dict[str, Any]:
        """Agent-inspector payload."""
        base = super().describe()
        base.update(
            {
                "floor": self.floor,
                "active_calls": [d.name for d, a in self.active_calls.items() if a],
                "waiting": {
                    "up": self.waiting_counts[Direction.UP],
                    "down": self.waiting_counts[Direction.DOWN],
                },
                "escalations": {d.name: n for d, n in self.escalations.items()},
                "oldest_wait": self.oldest_wait(),
            }
        )
        return base
