"""Adversarial search for worst-case idle-car parking (AIMA 4e chapter 5).

Hill-climbing parking minimises the *expected* response time, which is the right thing
to do on average but says nothing about the worst case: it will happily leave a whole
wing uncovered if that wing is quiet. So the dispatcher can instead treat parking as a
two-player game against nature:

- **MAX** (the dispatcher) chooses where to park the idle cars.
- **MIN** ("nature") then chooses the floor the next hall call appears on, picking the
  floor that hurts most among the plausible ones.

The minimax value is therefore a *guarantee*: whatever floor calls next, the response
distance is no worse than this. Alpha-beta pruning returns the identical value while
expanding far fewer nodes, which the Search Lab shows side by side and the tests assert.

Nature is modelled as an adversary rather than as chance deliberately: it yields a
worst-case bound, and it is the clean AIMA-textbook illustration. `docs/VIVA_QA.md`
discusses the expectimax alternative.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

Parking = tuple[int, ...]


@dataclass
class MinimaxResult:
    """The value MAX can guarantee, plus the node counts that show pruning working."""

    value: float
    best_parking: Parking
    nodes_minimax: int = 0
    nodes_alphabeta: int = 0
    worst_floor: int | None = None
    candidates: list[Parking] = field(default_factory=list)

    @property
    def pruning_saving_pct(self) -> float:
        """Percentage of nodes alpha-beta avoided relative to plain minimax."""
        if self.nodes_minimax == 0:
            return 0.0
        saved = self.nodes_minimax - self.nodes_alphabeta
        return 100.0 * saved / self.nodes_minimax

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form for the Search Lab."""
        return {
            "value": round(self.value, 3),
            "best_parking": list(self.best_parking),
            "nodes_minimax": self.nodes_minimax,
            "nodes_alphabeta": self.nodes_alphabeta,
            "pruning_saving_pct": round(self.pruning_saving_pct, 1),
            "worst_floor": self.worst_floor,
        }


class ParkingGame:
    """The parking game: MAX places cars, MIN picks the next call floor.

    `utility` is negative response distance, so MAX maximising it is the same as
    minimising the worst-case distance from a parked car to the next call.
    """

    def __init__(
        self,
        idle_cars: int,
        floors: int,
        candidate_spots: list[int],
        likely_floors: list[int],
    ) -> None:
        if idle_cars < 1:
            raise ValueError("the parking game needs at least one idle car")
        self.idle_cars = idle_cars
        self.floors = floors
        self.candidate_spots = sorted(set(candidate_spots))
        self.likely_floors = sorted(set(likely_floors))

    def max_moves(self) -> list[Parking]:
        """Every parking configuration MAX may choose.

        Combinations with repetition, since two cars may share a floor, and sorted so
        the ordering — and therefore the node counts — is deterministic.
        """
        return sorted(itertools.combinations_with_replacement(self.candidate_spots, self.idle_cars))

    def min_moves(self) -> list[int]:
        """The floors nature may put the next call on."""
        return list(self.likely_floors)

    def utility(self, parking: Parking, call_floor: int) -> float:
        """Negative distance from the nearest parked car to the call."""
        return -float(min(abs(spot - call_floor) for spot in parking))


def _minimax(game: ParkingGame, counter: list[int]) -> tuple[float, Parking, int | None]:
    """Plain minimax over the depth-2 parking game, counting every node visited."""
    best_value = -float("inf")
    best_parking: Parking = ()
    best_worst_floor: int | None = None

    for parking in game.max_moves():
        counter[0] += 1
        worst = float("inf")
        worst_floor: int | None = None
        for floor in game.min_moves():
            counter[0] += 1
            value = game.utility(parking, floor)
            if value < worst:
                worst, worst_floor = value, floor
        if worst > best_value:
            best_value, best_parking, best_worst_floor = worst, parking, worst_floor

    return best_value, best_parking, best_worst_floor


def _alphabeta(game: ParkingGame, counter: list[int]) -> tuple[float, Parking, int | None]:
    """The same game with alpha-beta pruning; returns the identical value.

    Once MIN has found a reply worth no more than alpha, MAX would never choose this
    parking configuration, so the remaining replies cannot change the result and the
    branch is cut.
    """
    alpha = -float("inf")
    best_parking: Parking = ()
    best_worst_floor: int | None = None

    for parking in game.max_moves():
        counter[0] += 1
        worst = float("inf")
        worst_floor: int | None = None
        for floor in game.min_moves():
            counter[0] += 1
            value = game.utility(parking, floor)
            if value < worst:
                worst, worst_floor = value, floor
            if worst <= alpha:
                break  # beta cut-off: MAX already has an at-least-as-good option
        if worst > alpha:
            alpha, best_parking, best_worst_floor = worst, parking, worst_floor

    return alpha, best_parking, best_worst_floor


def minimax_parking(
    idle_cars: int,
    floors: int,
    candidate_spots: list[int],
    likely_floors: list[int],
) -> MinimaxResult:
    """Solve the parking game both ways and report the value and both node counts."""
    game = ParkingGame(idle_cars, floors, candidate_spots, likely_floors)
    plain_counter = [0]
    pruned_counter = [0]
    value, parking, worst_floor = _minimax(game, plain_counter)
    ab_value, ab_parking, _ = _alphabeta(game, pruned_counter)

    if abs(value - ab_value) > 1e-9:
        raise AssertionError(
            f"alpha-beta value {ab_value} != minimax value {value}; pruning is unsound"
        )

    return MinimaxResult(
        value=value,
        best_parking=ab_parking or parking,
        nodes_minimax=plain_counter[0],
        nodes_alphabeta=pruned_counter[0],
        worst_floor=worst_floor,
        candidates=game.max_moves(),
    )
