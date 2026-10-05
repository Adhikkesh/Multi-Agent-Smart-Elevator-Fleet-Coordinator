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
    #: Who computes a car's Contract Net bid: "classical" (the rule named by `assignment`)
    #: or "learned" (the LiftZero network; eligibility stays classical).
    bidder: str = "classical"
    #: ONNX model of a learned bidder (repo-relative); None = the default LiftZero model.
    model_path: str | None = None
    #: The classical strategy a learned one imitates (its shadow teacher / DAgger labeller).
    teacher: str | None = None

    @property
    def uses_reassignment(self) -> bool:
        """Whether the dispatcher runs a periodic local search."""
        return self.reassignment is not None

    @property
    def uses_smart_parking(self) -> bool:
        """Whether idle cars are repositioned at all."""
        return self.parking_policy is not None

    @property
    def uses_learned_bidder(self) -> bool:
        """Whether cars bid with the LiftZero network."""
        return self.bidder == "learned"

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
            "bidder": self.bidder,
            "teacher": self.teacher,
            "learning": self.uses_learned_bidder,
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

register_strategy(
    DispatchStrategy(
        name="liftzero_bc",
        label="LiftZero (imitation)",
        description=(
            "Contract Net where each car's bid is computed by the LiftZero network "
            "(a set-Transformer trained by imitating the A* bidder, with DAgger). "
            "Everything else equals the 'full' strategy, so the bidder is the only "
            "difference."
        ),
        assignment="cnp",
        routing="astar",
        reassignment="simulated_annealing",
        parking_policy="hill_climb",
        adapts_weights=True,
        bidder="learned",
        teacher="full",
    )
)

register_strategy(
    DispatchStrategy(
        name="liftzero_bc_cnp",
        label="LiftZero (imitation, bare CNP)",
        description=(
            "The LiftZero learned bidder inside plain Contract Net + A* routing, with no "
            "reassignment, parking or adaptive weights: equals 'cnp_astar' except the "
            "bidder, isolating the network's effect from the extras."
        ),
        assignment="cnp",
        routing="astar",
        bidder="learned",
        teacher="cnp_astar",
    )
)

register_strategy(
    DispatchStrategy(
        name="liftzero_ppo",
        label="LiftZero (reinforcement learning)",
        description=(
            "The LiftZero bidder fine-tuned with cooperative PPO on the real simulator "
            "(team reward on passenger waiting, riding, energy and long waits), anchored to "
            "the imitation policy by a KL penalty. Everything else equals 'full'."
        ),
        assignment="cnp",
        routing="astar",
        reassignment="simulated_annealing",
        parking_policy="hill_climb",
        adapts_weights=True,
        bidder="learned",
        model_path="models/liftzero_ppo_v1.onnx",
        teacher="full",
    )
)

register_strategy(
    DispatchStrategy(
        name="liftzero_ppo_cnp",
        label="LiftZero (RL, bare CNP)",
        description=(
            "The PPO-trained LiftZero bidder inside plain Contract Net + A* routing: equals "
            "'cnp_astar' except the bidder."
        ),
        assignment="cnp",
        routing="astar",
        bidder="learned",
        model_path="models/liftzero_ppo_v1.onnx",
        teacher="cnp_astar",
    )
)
