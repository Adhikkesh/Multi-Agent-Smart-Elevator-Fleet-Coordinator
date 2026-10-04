"""The traffic monitor: a learning agent."""

from __future__ import annotations

from typing import Any

from elevator_mas.agents.base import CommunicatingAgent
from elevator_mas.comms import Performative
from elevator_mas.config import CostWeights, TrafficPattern
from elevator_mas.domain import Direction

#: Cost weights tuned per traffic pattern. Changing these visibly changes fleet behaviour,
#: which is the point: the learning agent's output is not a number on a dashboard, it is a
#: different dispatching policy.
PATTERN_WEIGHTS: dict[TrafficPattern, CostWeights] = {
    # Up peak: the lobby queue is what everyone sees, so weight waiting heavily and
    # tolerate the energy cost of running cars back down empty.
    "up_peak": CostWeights(wait=1.4, ride=0.3, crowding=0.5, energy=0.1),
    # Down peak: cars fill at the top and must not be diverted, so ride time and
    # crowding matter more.
    "down_peak": CostWeights(wait=1.1, ride=0.6, crowding=0.6, energy=0.1),
    # Lunch: both directions at once; balance everything and spread the load.
    "two_way": CostWeights(wait=1.2, ride=0.5, crowding=0.7, energy=0.15),
    "interfloor": CostWeights(wait=1.0, ride=0.5, crowding=0.3, energy=0.2),
    # Light traffic: nobody is waiting long, so it is worth saving electricity.
    "light": CostWeights(wait=0.9, ride=0.4, crowding=0.2, energy=0.5),
}


class TrafficMonitorAgent(CommunicatingAgent):
    """Estimates demand online and retunes the fleet's policy to match.

    **AIMA agent type: learning agent** (§2.4.6). It has the four components AIMA
    describes:

    - *Performance element* — the dispatcher and cars doing the actual work.
    - *Critic* — the observed arrivals per floor and direction, the feedback signal.
    - *Learning element* — an EWMA estimator of the per-floor arrival rate lambda_f. Demand
      is genuinely unknown to the agents (only the hidden generator knows the true rate),
      so this is learning, not reading a parameter.
    - *Problem generator* — the rule-based pattern classifier, which proposes a *different
      policy* (a new weight set and parking policy) to try when the shape of demand
      changes.

    Classification is deliberately a small rule base rather than a statistical model: it
    is inspectable, it explains itself in the UI, and with one parameter (the lobby share
    of traffic) the textbook lift patterns separate cleanly.

    PEAS
      - **P** accuracy of its rate estimates and of its pattern classification; the
        fleet-wide wait time its advice produces.
      - **E** the stream of passenger arrivals across every floor.
      - **A** INFORM messages to the dispatcher carrying weights and a parking policy;
        the learned demand and cost weights published on the fleet status board.
      - **S** observed arrivals per floor and direction, per tick.
    """

    agent_type = "Learning agent"
    peas = {
        "performance": "rate-estimate accuracy, classification accuracy, resulting AWT",
        "environment": "the arrival stream across all floors",
        "actuators": "INFORM the dispatcher of new cost weights and parking policy",
        "sensors": "observed arrivals per floor and direction",
    }

    def __init__(self, model: Any, alpha: float = 0.02) -> None:
        super().__init__(model, address="monitor")
        self.alpha = alpha
        floors = model.config.building.floors
        # Learned state: one rate estimate per floor, plus directional totals.
        self.rate: dict[int, float] = dict.fromkeys(range(floors), 0.0)
        self.up_rate: dict[int, float] = dict.fromkeys(range(floors), 0.0)
        self.down_rate: dict[int, float] = dict.fromkeys(range(floors), 0.0)
        self.lobby_share: float = 0.0
        self.to_lobby_share: float = 0.0
        self.pattern: TrafficPattern = "interfloor"
        self.classification_reason: str = "insufficient data"
        self.observed: int = 0
        self._window: list[tuple[int, int]] = []  # recent (origin, destination) pairs
        self._last_advice: tuple[str, str | None] | None = None

    # ------------------------------------------------------------------ learning

    def observe_arrival(self, origin: int, destination: int) -> None:
        """Record one arrival: the feedback signal the learning element consumes."""
        self.observed += 1
        self._window.append((origin, destination))
        if len(self._window) > 200:
            del self._window[:-200]

    def sense(self) -> None:
        """Update the EWMA rate estimates and reclassify the traffic pattern.

        Every floor's estimate decays each tick, whether or not anything arrived there —
        that is what makes it a rate (arrivals per second) rather than a running total,
        and it is what lets the estimate fall again when a rush ends.
        """
        self.collect_mail()
        arrivals = self.model.arrivals_this_tick
        counts: dict[int, int] = {}
        for passenger in arrivals:
            counts[passenger.origin] = counts.get(passenger.origin, 0) + 1
            self.observe_arrival(passenger.origin, passenger.destination)
            if passenger.direction is Direction.UP:
                self.up_rate[passenger.origin] = (1 - self.alpha) * self.up_rate[
                    passenger.origin
                ] + self.alpha
            else:
                self.down_rate[passenger.origin] = (1 - self.alpha) * self.down_rate[
                    passenger.origin
                ] + self.alpha

        for floor in self.rate:
            observed = counts.get(floor, 0)
            self.rate[floor] = (1 - self.alpha) * self.rate[floor] + self.alpha * observed

        self._classify()

    def _classify(self) -> None:
        """Rule-based traffic-pattern classification (forward chaining, in effect).

        R1: most trips start at the lobby                  -> up peak
        R2: most trips end at the lobby                    -> down peak
        R3: lobby dominates overall but in both directions  -> two-way (lunch)
        R4: total estimated demand below 0.08/s            -> light
        R5: otherwise                                      -> inter-floor
        """
        if len(self._window) < 20:
            self.classification_reason = "insufficient data (<20 observations)"
            return

        lobby = self.model.lobby
        from_lobby = sum(1 for o, _d in self._window if o == lobby)
        to_lobby = sum(1 for _o, d in self._window if d == lobby)
        total = len(self._window)
        self.lobby_share = from_lobby / total
        self.to_lobby_share = to_lobby / total
        demand = sum(self.rate.values())

        # Hysteresis on the light/busy boundary. Without it the classification flips
        # between "light" and the peak pattern every few ticks as the EWMA jitters around
        # a single threshold, and each flip retunes the fleet's cost weights and parking
        # policy — the fleet would spend the tail of a rush reconfiguring itself instead
        # of clearing the queue. It must now fall below 0.07/s to become light, and climb
        # back above 0.10/s to stop being light.
        light_threshold = 0.10 if self.pattern == "light" else 0.07
        if demand < light_threshold:
            pattern: TrafficPattern = "light"
            reason = f"total estimated demand {demand:.3f}/s is very low"
        elif self.lobby_share > 0.55 and self.to_lobby_share < 0.3:
            pattern = "up_peak"
            reason = f"{self.lobby_share:.0%} of trips start at the lobby"
        elif self.to_lobby_share > 0.55 and self.lobby_share < 0.3:
            pattern = "down_peak"
            reason = f"{self.to_lobby_share:.0%} of trips end at the lobby"
        elif (
            self.lobby_share + self.to_lobby_share > 0.6
            and min(self.lobby_share, self.to_lobby_share) > 0.15
        ):
            pattern = "two_way"
            reason = (
                f"lobby traffic in both directions "
                f"({self.lobby_share:.0%} out, {self.to_lobby_share:.0%} in)"
            )
        else:
            pattern = "interfloor"
            reason = f"lobby involved in only {self.lobby_share + self.to_lobby_share:.0%} of trips"

        self.pattern = pattern
        self.classification_reason = reason

    # ------------------------------------------------------------- communication

    def communicate(self) -> None:
        """Tell the dispatcher when its advice changes.

        Only on a *change*: re-sending an identical recommendation every tick would flood
        the message log the examiner reads and would wreck the messages-per-call metric,
        while telling the dispatcher nothing it did not already know.
        """
        weights = PATTERN_WEIGHTS[self.pattern]
        # Publish what it has learned on the status board: the demand estimate always,
        # and the pattern's cost weights when the strategy lets the fleet adapt. Every
        # car bids with the weights it reads there.
        self.model.board.publish_policy(
            self.address,
            self.model.tick,
            pattern=self.pattern,
            weights=weights if self.model.strategy.adapts_weights else None,
            demand=self.rate,
        )
        policy = self.recommended_parking()
        advice = (self.pattern, policy)
        if advice == self._last_advice:
            return
        self._last_advice = advice
        self.send(
            Performative.INFORM,
            "dispatcher",
            self.model.new_conversation_id(),
            {
                "pattern": self.pattern,
                "reason": self.classification_reason,
                "weights": weights.model_dump(),
                "parking_policy": policy,
            },
        )

    def recommended_parking(self) -> str | None:
        """Which parking policy suits the current pattern.

        Up peak is the one case with an unambiguous answer — everyone appears in the
        lobby, so park there. Otherwise demand is spread and it is worth searching.
        """
        if self.pattern == "up_peak":
            # Everyone appears in the lobby, so there is nothing to search for.
            return "lobby"
        if self.pattern in ("down_peak", "two_way"):
            # Counter-intuitive, and measured rather than assumed (see docs/TESTING.md):
            # when demand is heavy and spread, repositioning idle cars makes things worse.
            # A car is rarely idle for long, and every repositioning move is a journey it
            # then has to undo, so the fleet is better off leaving cars where their last
            # trip left them — already distributed by the traffic itself.
            return None
        # Inter-floor and light traffic leave cars genuinely idle, so it is worth placing
        # them: minimax bounds the worst case, which is what a sparse pattern suffers from.
        return "minimax"

    # ------------------------------------------------------------------ reporting

    def demand_estimate(self) -> dict[int, float]:
        """The learned per-floor arrival rates, used by the parking searches."""
        return dict(self.rate)

    def describe(self) -> dict[str, Any]:
        """Agent-inspector payload."""
        base = super().describe()
        base.update(
            {
                "pattern": self.pattern,
                "reason": self.classification_reason,
                "observed": self.observed,
                "lobby_share": round(self.lobby_share, 3),
                "to_lobby_share": round(self.to_lobby_share, 3),
                "weights": PATTERN_WEIGHTS[self.pattern].model_dump(),
                "parking_policy": self.recommended_parking(),
                "top_floors": sorted(
                    ({"floor": f, "rate": round(r, 4)} for f, r in self.rate.items() if r > 0),
                    key=lambda d: -float(d["rate"]),
                )[:5],
            }
        )
        return base

    def step(self) -> None:
        """Mesa's default hook; this agent is driven by the model's staged activation."""
