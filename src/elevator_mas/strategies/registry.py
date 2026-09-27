"""The dispatch-strategy registry.

Four strategies, forming a ladder from a pure reflex baseline to the full agent system.
The benchmark runs them against each other, which is how the case study demonstrates
that each added AI technique actually buys something measurable rather than just being
present in the code.

A strategy is *declarative*: it says which mechanisms are switched on. That keeps the
comparison honest, because every strategy runs through exactly the same simulation code
and the same agents — only the mechanisms differ.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DispatchStrategy:
    """Which coordination mechanisms a run uses."""

    name: str
    label: str
    description: str
    #: How a hall call is assigned: "nearest" (reflex), "look" (sweep), or "cnp" (auction).
    assignment: str = "cnp"
    #: Whether cars sequence their stops with the A* search or with the LOOK fallback.
    routing: str = "astar"
    #: Periodic global re-optimisation: None, "simulated_annealing" or "hill_climbing".
    reassignment: str | None = None
    #: Idle-car parking: None (stay put), "lobby", "hill_climb" or "minimax".
    parking_policy: str | None = None
    #: Whether the TrafficMonitorAgent is allowed to retune the cost weights.
    adapts_weights: bool = False

    @property
    def uses_reassignment(self) -> bool:
        """Whether the dispatcher runs a periodic local search."""
        return self.reassignment is not None

    @property
    def uses_smart_parking(self) -> bool:
        """Whether idle cars are repositioned at all."""
        return self.parking_policy is not None

    @property
    def uses_auction(self) -> bool:
        """Whether hall calls are awarded by Contract Net."""
        return self.assignment == "cnp"

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form for the dashboard's strategy selector."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "assignment": self.assignment,
            "routing": self.routing,
            "reassignment": self.reassignment,
            "parking_policy": self.parking_policy,
            "adapts_weights": self.adapts_weights,
        }


STRATEGIES: dict[str, DispatchStrategy] = {}


def register_strategy(strategy: DispatchStrategy) -> DispatchStrategy:
    """Add a strategy to the registry."""
    STRATEGIES[strategy.name] = strategy
    return strategy


def get_strategy(name: str) -> DispatchStrategy:
    """Look a strategy up by name."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; available: {', '.join(sorted(STRATEGIES))}"
        ) from None


def strategy_names() -> list[str]:
    """Every registered strategy name, in ladder order."""
    return list(STRATEGIES)


register_strategy(
    DispatchStrategy(
        name="nearest_car",
        label="Nearest Car (reflex baseline)",
        description=(
            "A simple reflex dispatcher: send the closest free car, sequence stops by a "
            "LOOK sweep, never reconsider. No search, no negotiation, no learning — the "
            "control condition the other strategies are measured against."
        ),
        assignment="nearest",
        routing="look",
    )
)

register_strategy(
    DispatchStrategy(
        name="collective",
        label="Collective / LOOK control",
        description=(
            "Classical collective control: a call is taken by a car already sweeping "
            "towards it, and stops are served in sweep order. This is how most real "
            "lifts behave, and it is a genuinely strong baseline."
        ),
        assignment="look",
        routing="look",
    )
)

register_strategy(
    DispatchStrategy(
        name="cnp_astar",
        label="Contract Net + A*",
        description=(
            "Hall calls are auctioned by Contract Net, with each car bidding the marginal "
            "cost of inserting the call into the plan its own A* search produced. Adds "
            "negotiation and optimal per-car routing."
        ),
        assignment="cnp",
        routing="astar",
    )
)

register_strategy(
    DispatchStrategy(
        name="full",
        label="Full (CNP + A* + SA + smart parking)",
        description=(
            "Everything: Contract Net with A* bidding, periodic global reassignment by "
            "simulated annealing, idle-car parking chosen by local or adversarial search, "
            "and cost weights retuned online by the learning agent."
        ),
        assignment="cnp",
        routing="astar",
        reassignment="simulated_annealing",
        parking_policy="hill_climb",
        adapts_weights=True,
    )
)
