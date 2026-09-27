"""Local search over the call-to-car assignment (AIMA 4e §4.1).

Contract Net assigns each call the moment it appears, greedily and one at a time, so the
resulting global assignment can drift badly out of shape as traffic changes. Every
`reassign_interval` seconds the dispatcher therefore runs a local search over the
assignment of the calls nobody has picked up yet, which is exactly the setting AIMA
describes for hill climbing and simulated annealing: a large discrete space where the
path to a solution is irrelevant and only the final configuration matters.

State: a map from hall call to car id. Neighbours: move one call to another car, or swap
two calls between cars. Objective: the summed insertion-cost estimate over all cars
(lower is better).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from elevator_mas.domain import HallCall

Assignment = dict[HallCall, int]
Objective = Callable[[Assignment], float]


@dataclass
class LocalSearchResult:
    """The outcome of a local search, including the curve the Search Lab plots."""

    algorithm: str
    assignment: Assignment
    initial_cost: float
    final_cost: float
    iterations: int
    accepted: int = 0
    curve: list[float] = field(default_factory=list)
    best_curve: list[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """How much cost the search removed; never negative for these searches."""
        return self.initial_cost - self.final_cost

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form for the Search Lab."""
        return {
            "algorithm": self.algorithm,
            "initial_cost": round(self.initial_cost, 3),
            "final_cost": round(self.final_cost, 3),
            "improvement": round(self.improvement, 3),
            "iterations": self.iterations,
            "accepted": self.accepted,
            "curve": [round(c, 3) for c in self.curve],
            "best_curve": [round(c, 3) for c in self.best_curve],
            "assignment": {
                f"{c.floor}{c.direction.name[0]}": v for c, v in self.assignment.items()
            },
        }


def _neighbour(
    assignment: Assignment, cars: Sequence[int], rng: random.Random
) -> Assignment | None:
    """One random neighbour: move a call to another car, or swap two calls.

    Returns None when no move is possible (fewer than one call, or only one car).
    """
    calls = list(assignment)
    if not calls or len(cars) < 2:
        return None
    candidate = dict(assignment)
    if len(calls) >= 2 and rng.random() < 0.3:
        a, b = rng.sample(calls, 2)
        candidate[a], candidate[b] = candidate[b], candidate[a]
        if candidate == assignment:
            return None
        return candidate
    call = rng.choice(calls)
    others = [c for c in cars if c != assignment[call]]
    if not others:
        return None
    candidate[call] = rng.choice(others)
    return candidate


def simulated_annealing(
    initial: Assignment,
    cars: Sequence[int],
    objective: Objective,
    rng: random.Random,
    *,
    iterations: int = 400,
    initial_temp: float = 12.0,
    cooling: float = 0.985,
    min_temp: float = 1e-3,
) -> LocalSearchResult:
    """Simulated annealing over the assignment (AIMA 4e §4.1.2).

    A worse neighbour is accepted with probability `exp(-dE / T)`, which lets the search
    escape the local optima plain hill climbing gets stuck in; `T` decays geometrically.

    The *best* configuration ever seen is tracked separately and returned, so the result
    can never be worse than the initial assignment — a property the tests assert, and
    the reason it is safe to run this on a live fleet.
    """
    current = dict(initial)
    current_cost = objective(current)
    best = dict(current)
    best_cost = current_cost
    temp = initial_temp
    accepted = 0
    curve: list[float] = [current_cost]
    best_curve: list[float] = [best_cost]

    for _ in range(iterations):
        temp = max(temp * cooling, min_temp)
        candidate = _neighbour(current, cars, rng)
        if candidate is None:
            curve.append(current_cost)
            best_curve.append(best_cost)
            continue
        candidate_cost = objective(candidate)
        delta = candidate_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current, current_cost = candidate, candidate_cost
            accepted += 1
            if current_cost < best_cost:
                best, best_cost = dict(current), current_cost
        curve.append(current_cost)
        best_curve.append(best_cost)

    return LocalSearchResult(
        algorithm="simulated_annealing",
        assignment=best,
        initial_cost=curve[0],
        final_cost=best_cost,
        iterations=iterations,
        accepted=accepted,
        curve=curve,
        best_curve=best_curve,
    )


def hill_climbing(
    initial: Assignment,
    cars: Sequence[int],
    objective: Objective,
    rng: random.Random,
    *,
    iterations: int = 400,
    sample_size: int = 12,
) -> LocalSearchResult:
    """Stochastic hill climbing: take the best of a sample of neighbours, or stop.

    Included as the contrast case for simulated annealing — it is strictly greedy, so it
    halts at the first local optimum, which the Search Lab shows as a curve that
    flattens early and higher than the annealing curve.
    """
    current = dict(initial)
    current_cost = objective(current)
    curve = [current_cost]
    steps = 0

    for _ in range(iterations):
        best_candidate: Assignment | None = None
        best_cost = current_cost
        for _ in range(sample_size):
            candidate = _neighbour(current, cars, rng)
            if candidate is None:
                continue
            cost = objective(candidate)
            if cost < best_cost:
                best_candidate, best_cost = candidate, cost
        if best_candidate is None:
            break  # local optimum: no sampled neighbour improves on it
        current, current_cost = best_candidate, best_cost
        steps += 1
        curve.append(current_cost)

    return LocalSearchResult(
        algorithm="hill_climbing",
        assignment=current,
        initial_cost=curve[0],
        final_cost=current_cost,
        iterations=steps,
        accepted=steps,
        curve=curve,
        best_curve=list(curve),
    )


def hill_climbing_parking(
    idle_cars: Sequence[int],
    floors: int,
    demand: dict[int, float],
    rng: random.Random,
    *,
    restarts: int = 6,
    iterations: int = 60,
) -> tuple[dict[int, int], float, list[float]]:
    """Hill climbing with random restarts for idle-car parking.

    Minimises the expected response `sum_f lambda_f * min_c |p_c - f|`: the demand-weighted
    distance from each floor to its nearest parked car, using the arrival rates the
    TrafficMonitorAgent estimated online. Random restarts are AIMA's standard remedy for
    the local optima a single hill climb would settle into.

    Returns the parking map, its cost, and the best-cost-per-restart curve.
    """
    if not idle_cars:
        return {}, 0.0, []

    def cost(positions: dict[int, int]) -> float:
        spots = list(positions.values())
        return sum(
            weight * min(abs(spot - floor) for spot in spots)
            for floor, weight in demand.items()
            if weight > 0
        )

    best: dict[int, int] = {}
    best_cost = float("inf")
    curve: list[float] = []

    for restart in range(max(1, restarts)):
        if restart == 0:
            current = dict.fromkeys(idle_cars, 0)  # the lobby baseline as one start
        else:
            current = {car: rng.randrange(floors) for car in idle_cars}
        current_cost = cost(current)
        for _ in range(iterations):
            improved = False
            for car in idle_cars:
                for delta in (-1, 1, -3, 3):
                    target = current[car] + delta
                    if not 0 <= target < floors:
                        continue
                    candidate = dict(current)
                    candidate[car] = target
                    candidate_cost = cost(candidate)
                    if candidate_cost < current_cost - 1e-9:
                        current, current_cost = candidate, candidate_cost
                        improved = True
            if not improved:
                break
        if current_cost < best_cost:
            best, best_cost = dict(current), current_cost
        curve.append(best_cost)

    return best, best_cost, curve
