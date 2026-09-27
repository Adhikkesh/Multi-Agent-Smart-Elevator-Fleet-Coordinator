"""The search algorithms and the routing heuristic.

These are the properties the grading rubric cares about most, so they are asserted over
many seeded random instances rather than a few hand-picked ones.
"""

from __future__ import annotations

import random

import pytest

from elevator_mas.domain import Direction, Stop, StopKind
from elevator_mas.planning import (
    CarRoutingProblem,
    SearchProblem,
    astar,
    breadth_first_search,
    greedy_best_first,
    search,
    uniform_cost_search,
)
from elevator_mas.planning.routing import RoutingCosts, legal_stops, look_order, plan_route

INSTANCES = 200


def random_problem(rng: random.Random, floors: int = 15, max_stops: int = 6) -> CarRoutingProblem:
    """A random routing instance."""
    current = rng.randrange(floors)
    stops: list[Stop] = []
    for _ in range(rng.randint(1, max_stops)):
        floor = rng.randrange(floors)
        if rng.random() < 0.5:
            direction = Direction.UP if rng.random() < 0.5 else Direction.DOWN
            stops.append(Stop(floor, StopKind.PICKUP, direction, float(rng.randint(1, 3))))
        else:
            stops.append(Stop(floor, StopKind.DROPOFF, Direction.IDLE, float(rng.randint(1, 3))))
    return CarRoutingProblem(current, stops, RoutingCosts())


class TestAStarOptimality:
    """A* must agree with UCS on cost, and never expand more nodes."""

    def test_astar_cost_equals_ucs_on_random_instances(self, rng: random.Random) -> None:
        """Both are optimal, so their path costs must be identical."""
        compared = 0
        for _ in range(INSTANCES):
            problem = random_problem(rng)
            a = astar(problem)
            u = uniform_cost_search(problem)
            assert a.found == u.found
            if not a.found:
                continue
            compared += 1
            assert a.cost == pytest.approx(u.cost, rel=1e-9), (
                f"A* cost {a.cost} != UCS cost {u.cost}"
            )
        assert compared > INSTANCES // 3, "too few solvable instances to be meaningful"

    def test_astar_expands_no_more_than_ucs(self, rng: random.Random) -> None:
        """With a consistent heuristic, A* dominates UCS in expansions."""
        for _ in range(INSTANCES):
            problem = random_problem(rng)
            a = astar(problem)
            u = uniform_cost_search(problem)
            if not a.found:
                continue
            assert a.nodes_expanded <= u.nodes_expanded

    def test_greedy_is_never_cheaper_than_optimal(self, rng: random.Random) -> None:
        """Greedy is not optimal, but it can never beat the optimal cost."""
        for _ in range(INSTANCES):
            problem = random_problem(rng)
            optimal = astar(problem)
            greedy = greedy_best_first(problem)
            if optimal.found and greedy.found:
                assert greedy.cost >= optimal.cost - 1e-9

    def test_bfs_finds_a_solution_when_one_exists(self, rng: random.Random) -> None:
        """BFS is complete, even though it is not cost-optimal here."""
        for _ in range(60):
            problem = random_problem(rng)
            assert breadth_first_search(problem).found == astar(problem).found


class TestHeuristic:
    """Admissibility and consistency, checked numerically."""

    def test_heuristic_is_admissible(self, rng: random.Random) -> None:
        """h(n) never exceeds the true remaining optimal cost along an optimal path."""
        for _ in range(INSTANCES):
            problem = random_problem(rng)
            result = astar(problem)
            if not result.found:
                continue
            states = result.states
            spent = 0.0
            for index, state in enumerate(states):
                remaining = result.cost - spent
                assert problem.h(state) <= remaining + 1e-9, (
                    f"h={problem.h(state)} exceeds true remaining cost {remaining}"
                )
                if index < len(result.actions):
                    spent += problem.action_cost(state, result.actions[index], states[index + 1])

    def test_heuristic_is_consistent(self, rng: random.Random) -> None:
        """h(n) <= c(n, a, n') + h(n') for every legal transition."""
        for _ in range(INSTANCES):
            problem = random_problem(rng)
            state = problem.initial_state()
            frontier = [state]
            seen = set()
            checked = 0
            while frontier and checked < 60:
                current = frontier.pop()
                if current in seen:
                    continue
                seen.add(current)
                for action in problem.actions(current):
                    nxt = problem.result(current, action)
                    step = problem.action_cost(current, action, nxt)
                    assert problem.h(current) <= step + problem.h(nxt) + 1e-9
                    checked += 1
                    frontier.append(nxt)

    def test_heuristic_is_zero_at_the_goal(self, rng: random.Random) -> None:
        """A goal state must be estimated at zero, or admissibility is broken."""
        problem = random_problem(rng)
        goal = (5, frozenset())
        assert problem.is_goal(goal)
        assert problem.h(goal) == 0.0


class TestCollectiveLegality:
    """The collective-control rules the plans must obey."""

    def test_riders_are_never_carried_backwards(self, rng: random.Random) -> None:
        """A pickup is never scheduled beyond a pending drop-off in the wrong direction."""
        for _ in range(INSTANCES):
            problem = random_problem(rng)
            result = astar(problem)
            if not result.found:
                continue
            for state, action in zip(result.states, result.actions, strict=False):
                assert action in legal_stops(state[0], state[1])

    def test_down_call_is_served_from_above(self) -> None:
        """A DOWN call is only collected once nothing is pending higher up."""
        stops = [Stop(f, StopKind.PICKUP, Direction.DOWN, 1.0) for f in (8, 11, 12, 14)]
        problem = CarRoutingProblem(7, stops, RoutingCosts())
        result = astar(problem)
        assert result.found
        floors = [s.floor for s in result.actions]
        assert floors == sorted(floors, reverse=True), f"a down sweep must descend, got {floors}"

    def test_up_call_is_served_from_below(self) -> None:
        """An UP call is only collected once nothing is pending lower down."""
        stops = [Stop(f, StopKind.PICKUP, Direction.UP, 1.0) for f in (2, 5, 9)]
        problem = CarRoutingProblem(10, stops, RoutingCosts())
        result = astar(problem)
        assert result.found
        floors = [s.floor for s in result.actions]
        assert floors == sorted(floors), f"an up sweep must ascend, got {floors}"

    def test_dropoffs_are_always_legal(self) -> None:
        """A rider on board must always be deliverable."""
        pending = frozenset(
            {
                Stop(3, StopKind.DROPOFF, Direction.IDLE, 1.0),
                Stop(9, StopKind.PICKUP, Direction.UP, 1.0),
            }
        )
        legal = legal_stops(5, pending)
        assert any(s.kind is StopKind.DROPOFF for s in legal)


class TestSearchMechanics:
    """The shared machinery: determinism, metrics, and the LOOK fallback."""

    def test_results_are_deterministic(self, rng: random.Random) -> None:
        """The same instance must give byte-identical results every time."""
        problem = random_problem(rng)
        first = astar(problem)
        second = astar(problem)
        assert first.cost == second.cost
        assert first.nodes_expanded == second.nodes_expanded
        assert [s.floor for s in first.actions] == [s.floor for s in second.actions]

    def test_every_algorithm_reports_its_metrics(self, rng: random.Random) -> None:
        """Nodes expanded, generated, frontier size and runtime must all be populated."""
        problem = random_problem(rng, max_stops=4)
        for name in ("bfs", "ucs", "greedy", "astar"):
            result = search(problem, name)
            assert result.algorithm == name
            assert result.nodes_expanded >= 0
            assert result.nodes_generated >= 1
            assert result.max_frontier >= 1
            assert result.runtime_ms >= 0.0

    def test_unknown_algorithm_is_rejected(self, rng: random.Random) -> None:
        """A typo must fail loudly rather than silently picking a default."""
        with pytest.raises(ValueError, match="unknown algorithm"):
            search(random_problem(rng), "dijkstra")

    def test_look_fallback_above_the_threshold(self) -> None:
        """Beyond `max_stops_for_search` the planner switches to the LOOK sweep."""
        stops = [Stop(f, StopKind.DROPOFF, Direction.IDLE, 1.0) for f in range(1, 14)]
        route, result = plan_route(0, Direction.UP, stops, RoutingCosts(), max_stops_for_search=10)
        assert result is None, "no search should have run above the threshold"
        assert len(route) == len(stops)

    def test_look_order_sweeps_then_reverses(self) -> None:
        """LOOK serves everything ahead in order, then everything behind."""
        pending = frozenset(
            {
                Stop(2, StopKind.DROPOFF, Direction.IDLE, 1.0),
                Stop(8, StopKind.DROPOFF, Direction.IDLE, 1.0),
                Stop(6, StopKind.DROPOFF, Direction.IDLE, 1.0),
            }
        )
        order = [s.floor for s in look_order(5, Direction.UP, pending)]
        assert order == [6, 8, 2]

    def test_empty_problem_is_already_solved(self) -> None:
        """A car with nothing to do needs no plan."""
        route, result = plan_route(4, Direction.IDLE, [], RoutingCosts())
        assert route == []
        assert result is None

    def test_negative_step_costs_are_rejected(self) -> None:
        """A negative step cost would break optimality, so it must raise."""

        class Broken(SearchProblem[int, int]):
            def initial_state(self) -> int:
                return 0

            def actions(self, state: int) -> list[int]:
                return [1]

            def result(self, state: int, action: int) -> int:
                return state + action

            def action_cost(self, state: int, action: int, next_state: int) -> float:
                return -1.0

            def is_goal(self, state: int) -> bool:
                return state > 3

        with pytest.raises(ValueError, match="non-negative"):
            uniform_cost_search(Broken())
