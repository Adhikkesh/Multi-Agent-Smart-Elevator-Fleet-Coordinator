"""The generic search-problem interface, following AIMA 4e chapter 3.

A `SearchProblem` is defined by the five things AIMA asks for — an initial state, the
actions legal in a state, the result of applying one, the cost of a step, and a goal
test — plus an optional heuristic. Every algorithm in `search.py` consumes only this
interface, so the same BFS/UCS/Greedy/A* code runs on the elevator routing problem and
on the toy problems used in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SearchProblem[State, Action]:
    """Abstract state-space problem (AIMA 4e §3.1)."""

    def initial_state(self) -> State:
        """The state the agent starts in."""
        raise NotImplementedError

    def actions(self, state: State) -> list[Action]:
        """The actions legal in `state`."""
        raise NotImplementedError

    def result(self, state: State, action: Action) -> State:
        """The state reached by applying `action` in `state` (the transition model)."""
        raise NotImplementedError

    def action_cost(self, state: State, action: Action, next_state: State) -> float:
        """The non-negative step cost of one transition."""
        raise NotImplementedError

    def is_goal(self, state: State) -> bool:
        """Whether `state` satisfies the goal."""
        raise NotImplementedError

    def h(self, state: State) -> float:  # noqa: ARG002
        """Heuristic estimate of the cheapest cost from `state` to a goal.

        The default is the trivially admissible zero heuristic, which turns A* into
        uniform-cost search.
        """
        return 0.0


@dataclass(order=True)
class Node[State, Action]:
    """A node in the search tree (AIMA 4e §3.3.1).

    Ordering is by `f` then an insertion counter, which makes the priority queue a
    deterministic tie-break instead of depending on state hashing — the runs have to be
    reproducible from the seed alone.
    """

    f: float
    order: int
    state: State = field(compare=False)
    parent: Node[State, Action] | None = field(default=None, compare=False)
    action: Action | None = field(default=None, compare=False)
    path_cost: float = field(default=0.0, compare=False)
    depth: int = field(default=0, compare=False)

    def path(self) -> list[Node[State, Action]]:
        """This node's ancestors, root first, ending at this node."""
        node: Node[State, Action] | None = self
        out: list[Node[State, Action]] = []
        while node is not None:
            out.append(node)
            node = node.parent
        out.reverse()
        return out

    def action_sequence(self) -> list[Action]:
        """The actions taken from the root to reach this node."""
        return [n.action for n in self.path() if n.action is not None]


@dataclass
class SearchResult[State, Action]:
    """What every algorithm reports, so the Search Lab can compare them fairly."""

    algorithm: str
    found: bool
    actions: list[Action] = field(default_factory=list)
    states: list[Any] = field(default_factory=list)
    cost: float = 0.0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier: int = 0
    runtime_ms: float = 0.0
    depth: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API and the Search Lab table."""
        return {
            "algorithm": self.algorithm,
            "found": self.found,
            "cost": round(self.cost, 3),
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "max_frontier": self.max_frontier,
            "runtime_ms": round(self.runtime_ms, 4),
            "depth": self.depth,
            "route": [str(a) for a in self.actions],
        }
