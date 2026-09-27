"""Car routing as a state-space search problem.

Formulation (AIMA 4e §3.1; the proofs are in `docs/DESIGN.md`):

- **State** `(current_floor, frozenset of pending stops)`. A stop is a pickup
  `(floor, direction, weight=waiting count)` or a drop-off `(floor, weight=riders)`.
  Same-floor stops are merged before the search starts, so a state never holds two
  stops the car could serve in one door cycle.
- **Actions** "go and serve one pending stop". The branching factor is the number of
  *legal* pending stops, not the number of floors, which is what keeps the state space
  small enough for A* to be exact in practice.
- **Legality (collective control).** An empty car may serve any pickup. With riders on
  board, a pickup `(f, UP)` is legal only if `cur <= f <= every pending drop-off`, and
  the mirror for DOWN. This is the rule that stops a passenger being carried backwards
  past their own floor — the behaviour real collective-control elevators have.
- **Step cost** `(travel + dwell) x total weight of still-unserved stops`, plus an
  energy term. Summing that over a path telescopes into
  `sum(weight x arrival time) + energy`, i.e. total passenger-seconds — so the optimal
  path is the one that minimises aggregate waiting, not the one that drives fewest
  floors.
- **Goal** no pending stops.
- **Heuristic** `h = sum(w_i x travel(cur, f_i)) + energy lower bound (line span)`,
  which is admissible and consistent (proved in the docs, asserted in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

from elevator_mas.domain import Direction, Stop, StopKind
from elevator_mas.planning.problem import SearchProblem, SearchResult
from elevator_mas.planning.search import search

RouteState = tuple[int, frozenset[Stop]]


@dataclass(frozen=True)
class RoutingCosts:
    """Timing and energy constants the routing cost function needs."""

    seconds_per_floor: float = 2.0
    dwell: float = 5.0
    energy_per_floor: float = 0.5
    energy_per_stop: float = 1.0

    def travel(self, a: int, b: int) -> float:
        """Travel time in seconds between two floors."""
        return abs(a - b) * self.seconds_per_floor


def merge_stops(stops: list[Stop]) -> frozenset[Stop]:
    """Merge stops a car could serve in one door cycle.

    Two stops merge when they share a floor, a kind and a direction; their weights add.
    Merging is what lets the state space stay small and keeps `is_goal` honest, since a
    single stop then really is a single door cycle.
    """
    merged: dict[tuple[int, StopKind, Direction], float] = {}
    for stop in stops:
        key = (stop.floor, stop.kind, stop.direction)
        merged[key] = merged.get(key, 0.0) + stop.weight
    return frozenset(
        Stop(floor=f, kind=k, direction=d, weight=w) for (f, k, d), w in merged.items()
    )


def legal_stops(current: int, pending: frozenset[Stop]) -> list[Stop]:
    """The stops the car may serve next under collective control.

    Two separate constraints, and both matter:

    1. **Do not carry a rider backwards.** Someone already on board must not be taken past
       their own floor the wrong way, so a pickup is legal only if every pending drop-off
       lies beyond it in the same direction.
    2. **Serve a call in the direction it asked for.** A DOWN hall call must be collected
       by a car that will then continue *downwards*, so it may only be taken once no other
       pickup lies further up — the car climbs to the top of its sweep first, then descends
       through the calls. Omitting this was a real bug: from floor 7, A* would plan
       `8v, 11v, 12v, 14v`, boarding four down-travelling passengers while driving upwards
       and carrying every one of them away from their destination. Each pickup looked
       cheap because it was "on the way", and it was why down-peak traffic performed worse
       than the reflex baseline.

    Drop-offs are always legal: a rider aboard must be delivered.
    """
    dropoffs = [s.floor for s in pending if s.kind is StopKind.DROPOFF]
    pickups = [s for s in pending if s.kind is StopKind.PICKUP]
    out: list[Stop] = []

    for stop in pending:
        if stop.kind is StopKind.DROPOFF:
            out.append(stop)
            continue

        # Constraint 1: never carry a committed rider backwards.
        if dropoffs:
            if stop.direction is Direction.UP:
                if not (current <= stop.floor and all(stop.floor <= d for d in dropoffs)):
                    continue
            elif not (current >= stop.floor and all(stop.floor >= d for d in dropoffs)):
                continue

        # Constraint 2: take the call only at the turning point of the sweep, so the car
        # is travelling the way the caller asked to go.
        if stop.direction is Direction.DOWN:
            if any(other.floor > stop.floor for other in pickups):
                continue
        elif any(other.floor < stop.floor for other in pickups):
            continue

        out.append(stop)

    # Deterministic ordering: the search must not depend on set iteration order.
    out.sort(key=lambda s: (s.floor, s.kind.value, s.direction.value))
    return out


class CarRoutingProblem(SearchProblem[RouteState, Stop]):
    """One car's stop-sequencing problem, as a `SearchProblem`.

    An action *is* a stop, so a solution path is directly the car's stop sequence.
    """

    def __init__(
        self,
        current_floor: int,
        pending: frozenset[Stop] | list[Stop],
        costs: RoutingCosts | None = None,
        boarding_per_passenger: float = 1.0,
    ) -> None:
        self.current_floor = current_floor
        self.pending = merge_stops(list(pending))
        self.costs = costs or RoutingCosts()
        self.boarding_per_passenger = boarding_per_passenger

    def initial_state(self) -> RouteState:
        """Start at the car's floor with every stop still pending."""
        return (self.current_floor, self.pending)

    def actions(self, state: RouteState) -> list[Stop]:
        """The legal next stops (see `legal_stops`)."""
        current, pending = state
        return legal_stops(current, pending)

    def result(self, state: RouteState, action: Stop) -> RouteState:
        """Serving a stop moves the car to that floor and removes the stop."""
        _current, pending = state
        return (action.floor, pending - {action})

    def dwell_for(self, stop: Stop) -> float:
        """Door-cycle time at a stop, longer when more people board or alight."""
        return self.costs.dwell + self.boarding_per_passenger * max(0.0, stop.weight - 1.0)

    def action_cost(
        self,
        state: RouteState,
        action: Stop,
        next_state: RouteState,  # noqa: ARG002 - part of the SearchProblem interface
    ) -> float:
        """Weighted delay inflicted on everyone still unserved, plus energy.

        Charging each step by the total weight of the stops that are *still* pending
        (including the one being served) is the telescoping trick: summed along a path
        it equals `sum(weight x arrival time)`. Minimising path cost therefore minimises
        total passenger-seconds.
        """
        current, pending = state
        duration = self.costs.travel(current, action.floor) + self.dwell_for(action)
        remaining_weight = sum(s.weight for s in pending)
        energy = (
            self.costs.energy_per_floor * abs(action.floor - current) + self.costs.energy_per_stop
        )
        return duration * remaining_weight + energy

    def is_goal(self, state: RouteState) -> bool:
        """Goal reached when nothing is pending."""
        return not state[1]

    def h(self, state: RouteState) -> float:
        """Admissible, consistent heuristic.

        Two independent lower bounds are added:

        1. **Weighted travel.** Every pending stop `i` must eventually be reached, and
           the car cannot get there faster than travelling straight to it, so it costs
           at least `w_i x travel(cur, f_i)`. Dwell time and the delay other stops
           inflict are both dropped, which only lowers the estimate.
        2. **Energy line span.** The car must visit the lowest and the highest pending
           floor, so it travels at least the span of the line through its current
           position, and must open its doors at least once per remaining stop.

        Both are lower bounds on disjoint parts of the true cost, so their sum is still
        a lower bound: h is admissible. Consistency is proved in `docs/DESIGN.md` and
        checked numerically over random instances in the tests.
        """
        current, pending = state
        if not pending:
            return 0.0
        travel_bound = sum(s.weight * self.costs.travel(current, s.floor) for s in pending)
        floors = [s.floor for s in pending]
        lo, hi = min(floors), max(floors)
        span = (hi - lo) + min(abs(current - lo), abs(current - hi))
        energy_bound = self.costs.energy_per_floor * span + self.costs.energy_per_stop * len(
            pending
        )
        return travel_bound + energy_bound


def look_order(current: int, direction: Direction, pending: frozenset[Stop]) -> list[Stop]:
    """LOOK / collective-control stop ordering, used as the bounded-rationality fallback.

    AIMA 4e §3.6 on bounded rationality: above a threshold of pending stops the optimal
    search is too expensive to run every tick, so the car switches to this sweep — serve
    everything ahead in the current direction in order, then reverse and sweep back.
    It is what real controllers do, and it is O(n log n) instead of exponential.
    """
    if not pending:
        return []
    sign = direction.sign or 1
    ahead = sorted(
        (s for s in pending if (s.floor - current) * sign > 0),
        key=lambda s: abs(s.floor - current),
    )
    here = sorted(
        (s for s in pending if s.floor == current), key=lambda s: (s.kind.value, s.direction.value)
    )
    behind = sorted(
        (s for s in pending if (s.floor - current) * sign < 0),
        key=lambda s: abs(s.floor - current),
        reverse=True,
    )
    return here + ahead + behind


def plan_route(
    current_floor: int,
    direction: Direction,
    pending: frozenset[Stop] | list[Stop],
    costs: RoutingCosts | None = None,
    *,
    algorithm: str = "astar",
    max_stops_for_search: int = 10,
    boarding_per_passenger: float = 1.0,
) -> tuple[list[Stop], SearchResult | None]:
    """Plan a car's stop sequence, falling back to LOOK when the problem is too big.

    Returns the stop sequence and the `SearchResult` (None when the fallback was used,
    so callers can tell that no search ran).
    """
    problem = CarRoutingProblem(current_floor, pending, costs, boarding_per_passenger)
    merged = problem.pending
    if not merged:
        return [], None
    if len(merged) > max_stops_for_search:
        return look_order(current_floor, direction, merged), None
    result = search(problem, algorithm)
    if not result.found:
        # Legality can make a set of stops unorderable in one sweep; LOOK always yields
        # an order, and the car re-plans once it has dropped its riders.
        return look_order(current_floor, direction, merged), result
    return list(result.actions), result
