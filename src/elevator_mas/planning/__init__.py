"""State-space search: the generic AIMA problem interface and the car routing problem."""

from elevator_mas.planning.problem import Node, SearchProblem, SearchResult
from elevator_mas.planning.routing import CarRoutingProblem, look_order, plan_route
from elevator_mas.planning.search import (
    ALGORITHMS,
    astar,
    breadth_first_search,
    greedy_best_first,
    search,
    uniform_cost_search,
)

__all__ = [
    "ALGORITHMS",
    "CarRoutingProblem",
    "Node",
    "SearchProblem",
    "SearchResult",
    "astar",
    "breadth_first_search",
    "greedy_best_first",
    "look_order",
    "plan_route",
    "search",
    "uniform_cost_search",
]
