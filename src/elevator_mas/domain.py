"""Core value types shared by every layer.

Kept free of Mesa and FastAPI imports so the planners and tests can use them in
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    """Direction of travel of a car, or of a hall call."""

    UP = 1
    DOWN = -1
    IDLE = 0

    @property
    def sign(self) -> int:
        """+1 for UP, -1 for DOWN, 0 for IDLE."""
        return self.value

    def opposite(self) -> Direction:
        """The reversed direction; IDLE is its own opposite."""
        if self is Direction.UP:
            return Direction.DOWN
        if self is Direction.DOWN:
            return Direction.UP
        return Direction.IDLE


class DoorState(Enum):
    """Door position. Doors must never be anything but CLOSED while a car moves."""

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"


class CarState(Enum):
    """High-level car state, used for colouring in the dashboard."""

    IDLE = "idle"
    MOVING_UP = "moving_up"
    MOVING_DOWN = "moving_down"
    DOORS = "doors"
    OUT_OF_SERVICE = "out_of_service"
    FIRE_RECALL = "fire_recall"


class StopKind(Enum):
    """Why a car must stop at a floor."""

    PICKUP = "pickup"
    DROPOFF = "dropoff"


@dataclass(frozen=True, order=True)
class Stop:
    """A pending stop in a car's plan.

    A pickup carries the hall-call direction and the number of people waiting; a
    drop-off carries the number of riders alighting. `weight` is what the routing
    cost multiplies arrival time by, so heavier stops are served sooner.
    """

    floor: int
    kind: StopKind
    direction: Direction = Direction.IDLE
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind is StopKind.PICKUP and self.direction is Direction.IDLE:
            raise ValueError("a pickup stop must carry a hall-call direction")

    @property
    def is_pickup(self) -> bool:
        """True for a hall pickup, False for a car drop-off."""
        return self.kind is StopKind.PICKUP


@dataclass(frozen=True)
class HallCall:
    """A hall (landing) button press: someone on `floor` wants to go `direction`."""

    floor: int
    direction: Direction

    def __post_init__(self) -> None:
        if self.direction is Direction.IDLE:
            raise ValueError("a hall call must be UP or DOWN")


@dataclass
class PassengerRecord:
    """Timestamps for one passenger, the source of all the wait/ride metrics."""

    passenger_id: int
    origin: int
    destination: int
    arrival_tick: int
    weight: float = 1.0
    priority: bool = False
    board_tick: int | None = None
    alight_tick: int | None = None
    car_id: int | None = None
    requeued: int = 0

    @property
    def direction(self) -> Direction:
        """The direction this passenger's hall call requests."""
        return Direction.UP if self.destination > self.origin else Direction.DOWN

    @property
    def wait_time(self) -> int | None:
        """Ticks from arrival to boarding, or None while still waiting."""
        if self.board_tick is None:
            return None
        return self.board_tick - self.arrival_tick

    @property
    def ride_time(self) -> int | None:
        """Ticks from boarding to alighting, or None while still aboard."""
        if self.board_tick is None or self.alight_tick is None:
            return None
        return self.alight_tick - self.board_tick

    @property
    def system_time(self) -> int | None:
        """Ticks from arrival to alighting, or None if not yet delivered."""
        if self.alight_tick is None:
            return None
        return self.alight_tick - self.arrival_tick

    @property
    def delivered(self) -> bool:
        """True once the passenger has reached their destination."""
        return self.alight_tick is not None


@dataclass
class Bid:
    """A car's PROPOSE in a Contract Net round: marginal cost, with a breakdown.

    The breakdown is what the dashboard's auction panel renders, and it is what makes
    the utility function legible to an examiner.
    """

    car_id: int
    total: float
    wait: float = 0.0
    ride: float = 0.0
    crowding: float = 0.0
    energy: float = 0.0
    eta: float = 0.0
    refused: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form for the API."""
        return {
            "car_id": self.car_id,
            "total": round(self.total, 3),
            "wait": round(self.wait, 3),
            "ride": round(self.ride, 3),
            "crowding": round(self.crowding, 3),
            "energy": round(self.energy, 3),
            "eta": round(self.eta, 2),
            "refused": self.refused,
            "reason": self.reason,
        }


@dataclass
class AuctionRound:
    """One completed Contract Net round, kept for the dashboard."""

    tick: int
    conversation_id: str
    call: HallCall
    bids: list[Bid] = field(default_factory=list)
    winner: int | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form for the API."""
        return {
            "tick": self.tick,
            "conversation_id": self.conversation_id,
            "floor": self.call.floor,
            "direction": self.call.direction.name,
            "bids": [b.as_dict() for b in self.bids],
            "winner": self.winner,
        }
