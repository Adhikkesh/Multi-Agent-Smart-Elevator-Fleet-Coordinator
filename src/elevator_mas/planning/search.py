"""BFS, UCS, Greedy best-first and A*, sharing one best-first implementation.

AIMA 4e §3.4 makes the point this module is built around: uniform-cost search, greedy
best-first search and A* are all *best-first search* differing only in the evaluation
function f, and breadth-first search is best-first on depth with an early goal test.
So there is exactly one frontier/reached implementation here and the four public
functions only supply f. That is also why the node counts the Search Lab compares are
meaningful: the bookkeeping is literally the same code.

    f(n) = g(n)            -> uniform-cost search
    f(n) = h(n)            -> greedy best-first
    f(n) = g(n) + h(n)     -> A*
    f(n) = depth(n)        -> breadth-first (FIFO, with an early goal test)
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable
from typing import Literal

from elevator_mas.planning.problem import Node, SearchProblem, SearchResult

EvalFn = Callable[[Node, float], float]

Algorithm = Literal["bfs", "ucs", "greedy", "astar"]


def best_first_search[State, Action](
    problem: SearchProblem[State, Action],
    f: EvalFn,
    *,
    name: str,
    early_goal_test: bool = False,
    node_limit: int = 2_000_000,
) -> SearchResult[State, Action]:
    """Best-first graph search on a priority queue ordered by `f`.

    `f` receives the node and its heuristic value, so a caller can build g, h, g+h or
    depth without this function knowing which algorithm it is running.

    With `early_goal_test` the goal is checked when a node is *generated* rather than
    when it is expanded. That is correct for breadth-first search on unit costs, and is
    what makes BFS return the shallowest solution; it is not used for UCS or A*, where
    testing on expansion is what guarantees optimality under varying step costs.

    A node is re-expanded only when a strictly cheaper path to its state is found, the
    standard graph-search treatment of re-reached states.
    """
    start = time.perf_counter()
    counter = itertools.count()
    initial = problem.initial_state()
    root = Node(f=0.0, order=next(counter), state=initial, path_cost=0.0, depth=0)
    root.f = f(root, problem.h(initial))

    if problem.is_goal(initial):
        return _result(name, root, 0, 1, 1, start)

    frontier: list[Node[State, Action]] = [root]
    reached: dict[State, float] = {initial: 0.0}
    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        node = heapq.heappop(frontier)
        # Stale entry: a cheaper path to this state was queued after this one.
        if node.path_cost > reached.get(node.state, float("inf")):
            continue
        if not early_goal_test and problem.is_goal(node.state):
            return _result(name, node, expanded, generated, max_frontier, start)
        expanded += 1
        if expanded > node_limit:
            break
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            step = problem.action_cost(node.state, action, child_state)
            if step < 0:
                raise ValueError("step costs must be non-negative")
            child_cost = node.path_cost + step
            if child_cost >= reached.get(child_state, float("inf")):
                continue
            child = Node(
                f=0.0,
                order=next(counter),
                state=child_state,
                parent=node,
                action=action,
                path_cost=child_cost,
                depth=node.depth + 1,
            )
            child.f = f(child, problem.h(child_state))
            reached[child_state] = child_cost
            generated += 1
            if early_goal_test and problem.is_goal(child_state):
                return _result(name, child, expanded, generated, max_frontier, start)
            heapq.heappush(frontier, child)
            max_frontier = max(max_frontier, len(frontier))

    return SearchResult(
        algorithm=name,
        found=False,
        nodes_expanded=expanded,
        nodes_generated=generated,
        max_frontier=max_frontier,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
    )


def _result[State, Action](
    name: str,
    node: Node[State, Action],
    expanded: int,
    generated: int,
    max_frontier: int,
    start: float,
) -> SearchResult[State, Action]:
    """Package a solution node into a `SearchResult`."""
    return SearchResult(
        algorithm=name,
        found=True,
        actions=node.action_sequence(),
        states=[n.state for n in node.path()],
        cost=node.path_cost,
        nodes_expanded=expanded,
        nodes_generated=generated,
        max_frontier=max_frontier,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
        depth=node.depth,
    )


def breadth_first_search[State, Action](
    problem: SearchProblem[State, Action],
) -> SearchResult[State, Action]:
    """Breadth-first search: best-first on depth, with an early goal test.

    Complete, and optimal only when every step costs the same — which is *not* true of
    the elevator routing problem, so BFS appears in the Search Lab precisely to show a
    cheap-but-suboptimal baseline.
    """
    return best_first_search(
        problem, lambda node, _h: float(node.depth), name="bfs", early_goal_test=True
    )


def uniform_cost_search[State, Action](
    problem: SearchProblem[State, Action],
) -> SearchResult[State, Action]:
    """Uniform-cost search: best-first on g. Optimal for non-negative step costs."""
    return best_first_search(problem, lambda node, _h: node.path_cost, name="ucs")


def greedy_best_first[State, Action](
    problem: SearchProblem[State, Action],
) -> SearchResult[State, Action]:
    """Greedy best-first search: best-first on h alone. Fast, not optimal."""
    return best_first_search(problem, lambda _node, h: h, name="greedy")


def astar[State, Action](problem: SearchProblem[State, Action]) -> SearchResult[State, Action]:
    """A*: best-first on f = g + h.

    Optimal when h is admissible, and expands no more nodes than UCS when h is
    consistent — both proved for the routing heuristic in `docs/DESIGN.md` and asserted
    over random instances in the tests.
    """
    return best_first_search(problem, lambda node, h: node.path_cost + h, name="astar")


ALGORITHMS: dict[str, Callable[[SearchProblem], SearchResult]] = {
    "bfs": breadth_first_search,
    "ucs": uniform_cost_search,
    "greedy": greedy_best_first,
    "astar": astar,
}


def search(problem: SearchProblem, algorithm: str = "astar") -> SearchResult:
    """Run one named algorithm on `problem`."""
    try:
        fn = ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHMS)}"
        ) from None
    return fn(problem)
