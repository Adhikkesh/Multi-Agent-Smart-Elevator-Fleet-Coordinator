"""The fleet status board: a shared blackboard of *public* agent state.

Messages carry intentions (a request, a bid, an award); the board carries facts every
agent is allowed to see at a glance (where each car is, how full it is, what the learned
demand looks like). This is the classic blackboard architecture of multi-agent systems:
agents publish to it and read from it, and never reach into one another's objects.

Two rules keep it honest:

- **Only the owner writes its entry.** A car publishes its own status; the traffic
  monitor publishes the fleet policy. Nobody writes on another agent's behalf.
- **Entries are immutable snapshots.** A reader gets a frozen copy, so it cannot change
  a car by editing what it read, and it sees one consistent moment rather than a car
  half-way through an update.

The status board is also the "shared view" that the LiftZero network reads in later
phases: every car's features come from here, not from the other cars' private state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from elevator_mas.config import CostWeights
from elevator_mas.domain import Bid, Direction, HallCall


@dataclass(frozen=True)
class CarStatus:
    """One car's public status, as it published it."""

    car_id: int
    address: str
    tick: int
    floor: int
    direction: Direction
    load: int
    capacity: int
    space: int
    available: bool
    out_of_service: bool
    fire_mode: bool
    door: str
    door_blocked_ticks: int
    assigned_calls: frozenset[HallCall] = frozenset()
    car_calls: frozenset[int] = frozenset()
    riders: int = 0
    plan_end_floor: int | None = None
    plan_end_eta: float = 0.0
    planned_stops: int = 0
    park_target: int | None = None

    @property
    def idle(self) -> bool:
        """In service, with no work: a candidate for parking."""
        return self.available and not self.assigned_calls and self.riders == 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API."""
        return {
            "car_id": self.car_id,
            "tick": self.tick,
            "floor": self.floor,
            "direction": self.direction.name,
            "load": self.load,
            "capacity": self.capacity,
            "space": self.space,
            "available": self.available,
            "out_of_service": self.out_of_service,
            "fire_mode": self.fire_mode,
            "door": self.door,
            "assigned_calls": [
                f"{c.floor}{c.direction.name[0]}"
                for c in sorted(self.assigned_calls, key=lambda c: (c.floor, c.direction.value))
            ],
            "car_calls": sorted(self.car_calls),
            "riders": self.riders,
            "plan_end_floor": self.plan_end_floor,
            "plan_end_eta": round(self.plan_end_eta, 2),
            "planned_stops": self.planned_stops,
            "park_target": self.park_target,
        }


@dataclass
class FleetPolicy:
    """What the learning agent publishes: the current pattern, weights and demand."""

    pattern: str = "interfloor"
    weights: CostWeights = field(default_factory=CostWeights)
    demand: dict[int, float] = field(default_factory=dict)
    published_by: str = "config"
    tick: int = 0


def copy_policy(policy: FleetPolicy) -> FleetPolicy:
    """Create a shallow copy of FleetPolicy with an isolated demand dictionary."""
    return FleetPolicy(
        pattern=policy.pattern,
        weights=policy.weights.model_copy() if hasattr(policy.weights, "model_copy") else policy.weights,
        demand=dict(policy.demand),
        published_by=policy.published_by,
        tick=policy.tick,
    )


@dataclass(frozen=True)
class DecisionEvent:
    """Snapshot of a single auction decision emitted to observer hooks."""

    tick: int
    call: HallCall
    urgency: int
    waiting: int
    statuses: tuple[CarStatus, ...]
    policy: FleetPolicy
    bids: tuple[Bid, ...]
    winner: int | None
    building: Any
    seed: int



class StatusBoard:
    """The shared blackboard. Cars publish their status; the monitor publishes policy."""

    def __init__(self, weights: CostWeights) -> None:
        self._cars: dict[int, CarStatus] = {}
        self.policy = FleetPolicy(weights=weights)
        self.writes: int = 0

    # ------------------------------------------------------------------ writing

    def publish_car(self, status: CarStatus) -> None:
        """A car posts its own, current status."""
        self._cars[status.car_id] = status
        self.writes += 1

    def publish_policy(
        self,
        publisher: str,
        tick: int,
        *,
        pattern: str | None = None,
        weights: CostWeights | None = None,
        demand: dict[int, float] | None = None,
    ) -> None:
        """The learning agent posts the fleet-wide policy it currently recommends."""
        if pattern is not None:
            self.policy.pattern = pattern
        if weights is not None:
            self.policy.weights = weights
        if demand is not None:
            self.policy.demand = dict(demand)
        self.policy.published_by = publisher
        self.policy.tick = tick
        self.writes += 1

    # ------------------------------------------------------------------ reading

    def car(self, car_id: int | None) -> CarStatus | None:
        """One car's latest status, or None."""
        if car_id is None:
            return None
        return self._cars.get(car_id)

    def cars(self) -> list[CarStatus]:
        """Every car's latest status, in car-id order."""
        return [self._cars[k] for k in sorted(self._cars)]

    def available_car_ids(self, with_space: bool = True) -> list[int]:
        """Cars that may take new work (and have room, by default)."""
        return [s.car_id for s in self.cars() if s.available and (s.space > 0 or not with_space)]

    @property
    def weights(self) -> CostWeights:
        """The cost weights every car bids with."""
        return self.policy.weights

    def demand(self) -> dict[int, float]:
        """The learned per-floor arrival rates."""
        return dict(self.policy.demand)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API."""
        return {
            "cars": [s.as_dict() for s in self.cars()],
            "policy": {
                "pattern": self.policy.pattern,
                "weights": self.policy.weights.model_dump(),
                "published_by": self.policy.published_by,
                "tick": self.policy.tick,
            },
            "writes": self.writes,
        }
