"""The dispatcher: the coordinator and auctioneer of the fleet."""

from __future__ import annotations

from typing import Any

from elevator_mas.agents.base import CommunicatingAgent
from elevator_mas.comms import DecisionEvent, Message, Order, Performative, copy_policy
from elevator_mas.domain import AuctionRound, Bid, Direction, HallCall
from elevator_mas.optimization import (
    MinimaxResult,
    hill_climbing_parking,
    minimax_parking,
    simulated_annealing,
)
from elevator_mas.optimization.local_search import Assignment, LocalSearchResult, hill_climbing


class DispatcherAgent(CommunicatingAgent):
    """Runs the Contract Net protocol, then improves the result with local search.

    **AIMA agent type: utility-based coordinator agent** — a *cooperative* multi-agent
    coordinator, not a central controller. It never commands a car: it asks, the cars
    answer with their own costs, and it awards. That distinction is the point of the
    case study, and it is why a car going out of service degrades the fleet gracefully
    instead of breaking a central plan.

    Three mechanisms, at three different time scales:

    1. **Contract Net per hall call** (immediately, AIMA §2.4 / distributed AI). Floor
       REQUESTs, dispatcher CFPs every car, cars PROPOSE a marginal cost or REFUSE, the
       lowest bid gets ACCEPT_PROPOSAL and the rest REJECT_PROPOSAL. Greedy, but fast and
       fully distributed.
    2. **Periodic global reassignment** (every `reassign_interval` seconds, AIMA ch. 4).
       Because step 1 commits one call at a time it drifts from the global optimum, so
       simulated annealing re-optimises the assignment of calls nobody has picked up yet.
       A hysteresis threshold stops it churning assignments for a trivial gain.
    3. **Idle-car parking** (whenever cars fall idle, AIMA ch. 4 & 5). Either the lobby
       baseline, hill climbing with random restarts on expected response time, or minimax
       with alpha-beta for a worst-case guarantee.

    **What it can see, and how it acts.** Only its own inbox and the public fleet status
    board. It never calls a car's methods or reads a car's private state: a bid arrives
    as a number in a PROPOSE, a pickup as an INFORM, a breakdown as a FAILURE, and every
    decision leaves as a message the car carries out on its own turn.

    PEAS
      - **P** fleet-wide average and 95th-percentile wait, % of long waits, fairness,
        energy, and recovery from failures.
      - **E** every floor, every car, the hall calls, and the traffic pattern.
      - **A** CFP / ACCEPT / REJECT / CANCEL messages, parking advice.
      - **S** REQUESTs from floors, PROPOSEs and REFUSEs from cars, FAILURE and pickup
        reports, traffic-pattern advice from the monitor, the fleet status board.
    """

    agent_type = "Utility-based coordinator (auctioneer)"
    peas = {
        "performance": "fleet AWT/P95, % long waits, fairness, energy, fault recovery",
        "environment": "all floors, all cars, outstanding hall calls, traffic pattern",
        "actuators": "CFP/ACCEPT/REJECT/CANCEL messages, parking advice",
        "sensors": "floor REQUESTs, car PROPOSEs/REFUSEs, FAILUREs, pickups, status board",
    }

    def __init__(self, model: Any) -> None:
        super().__init__(model, address="dispatcher")
        self.assignments: Assignment = {}
        self.pending_calls: dict[HallCall, dict[str, Any]] = {}
        self.picked_up: set[HallCall] = set()
        self.hall_calls_blocked = False
        self.last_auction: AuctionRound | None = None
        self.auction_history: list[AuctionRound] = []
        self.last_reassignment: LocalSearchResult | None = None
        self.last_minimax: MinimaxResult | None = None
        self.last_parking: dict[int, int] = {}
        self.advised_parking_policy: str | None = None
        self.monitor_advice_received: bool = False
        self._next_reassign = model.config.planner.reassign_interval
        self.unassigned: dict[HallCall, dict[str, Any]] = {}
        self._mailbox: list[Message] = []
        self._open_round: tuple[HallCall, dict[str, Any]] | None = None
        self._round_view: tuple[tuple[Any, ...], Any] | None = None
        #: DAgger mixing statistics: auctions awarded by the shadow teacher vs the learner.
        self.teacher_awards: int = 0
        self.learner_awards: int = 0
        self._moved_this_tick: dict[HallCall, tuple[int | None, int]] = {}

    # ------------------------------------------------------------------ sensing

    def sense(self) -> None:
        """Read the mail that arrived since the last tick, then tidy finished calls.

        REQUESTs (new or escalated calls), the monitor's advice, cars' FAILURE reports
        and pickup notices are all handled here, at the start of the tick, so every
        decision this tick is made from the same picture of the world.
        """
        self.inbox = []
        for message in self._take_mail():
            self._handle(message)

        # Forget assignments whose calls no longer exist (everyone boarded).
        for call in list(self.assignments):
            if not self.model.has_waiting(call.floor, call.direction):
                self.assignments.pop(call, None)
                self.picked_up.discard(call)

    def _take_mail(self, keep: Any = None) -> list[Message]:
        """Fetch new mail into the mailbox and take the messages `keep` selects.

        Messages that are not taken stay in the mailbox for a later stage. This is how
        the auctioneer reads only the proposals for the round it has open, leaving a
        floor's fresh REQUEST for the next tick exactly as a real dispatcher would.
        """
        self._mailbox.extend(self.receive())
        if keep is None:
            taken, self._mailbox = self._mailbox, []
            return taken
        taken = [m for m in self._mailbox if keep(m)]
        self._mailbox = [m for m in self._mailbox if not keep(m)]
        return taken

    def _handle(self, message: Message) -> None:
        """React to one message outside an auction round."""
        content = message.content
        performative = message.performative
        if performative is Performative.REQUEST:
            order = content.get("order")
            if isinstance(order, Order):
                self._obey(order)
            elif "call" in content and not self.hall_calls_blocked:
                call = content["call"]
                # An escalation supersedes the earlier request for the same call.
                self.pending_calls[call] = {
                    "call": call,
                    "urgency": int(content.get("urgency", 0)),
                    "conversation_id": message.conversation_id,
                    "waiting": content.get("waiting", 1),
                    "since": content.get("since", self.model.tick),
                }
        elif performative is Performative.INFORM:
            if "parking_policy" in content:
                # The learning agent's advice. Acting on it is what makes the monitor an
                # agent rather than a read-out: it changes the parking policy the
                # dispatcher actually uses on the next cycle.
                self.advised_parking_policy = content["parking_policy"]
                self.monitor_advice_received = True
            if "picked_up" in content:
                self.picked_up.update(c for c in content["picked_up"] if isinstance(c, HallCall))
        elif performative is Performative.FAILURE:
            self._recover_from_failure(message)

    def _recover_from_failure(self, message: Message) -> None:
        """A car has failed or handed work back: re-auction what it can no longer serve.

        Only calls that are still this car's responsibility are re-auctioned; one that a
        reassignment has already moved elsewhere is left with its new owner.
        """
        car_id = message.content.get("car_id")
        for call in message.content.get("released", []):
            if not isinstance(call, HallCall):
                continue
            owner = self.assignments.get(call)
            if owner is not None and owner != car_id:
                continue
            self.assignments.pop(call, None)
            if not self.model.has_waiting(call.floor, call.direction):
                continue
            self.pending_calls[call] = {
                "call": call,
                "urgency": 1,  # a stranded caller has already waited: bump the urgency
                "conversation_id": self.model.new_conversation_id(),
                "waiting": self.model.waiting_count(call.floor, call.direction),
                "since": self.model.tick,
            }
        self.assignments = {c: v for c, v in self.assignments.items() if v != car_id}

    def _obey(self, order: Order) -> None:
        """Carry out a safety order addressed to the dispatcher."""
        if order is Order.BLOCK_HALL_CALLS:
            self.block_hall_calls()
        elif order is Order.RESTORE_SERVICE:
            self.unblock_hall_calls()

    # ------------------------------------------------------- Contract Net round

    @property
    def round_open(self) -> bool:
        """Whether a Contract Net round is open (announced but not yet awarded)."""
        return self._open_round is not None

    def announce(self) -> bool:
        """Negotiation sub-stage 1: broadcast a CFP for the next queued call.

        Returns False (and leaves no round open) when there is nothing left to auction
        this tick. Calls are auctioned one at a time, so each car bids knowing every
        award made before it — exactly the information a sequential Contract Net gives
        its bidders.
        """
        if self.hall_calls_blocked:
            self.pending_calls.clear()
            return False
        if not self.pending_calls:
            return False
        call, record = next(iter(self.pending_calls.items()))
        self.pending_calls.pop(call)
        self._open_round = (call, record)
        self._round_view = (tuple(self.model.board.cars()), copy_policy(self.model.board.policy))
        self.send(
            Performative.CFP,
            None,  # broadcast to every car
            record["conversation_id"],
            {"call": call, "urgency": record["urgency"], "waiting": record["waiting"]},
        )
        return True

    def award(self) -> None:
        """Negotiation sub-stage 3: read the proposals for the open round and award it.

        The dispatcher sees only what the cars sent: a number and its breakdown in each
        PROPOSE, or a reason in each REFUSE. It never inspects a car to check.
        """
        if self._open_round is None:
            return
        call, record = self._open_round
        self._open_round = None
        round_view = self._round_view
        self._round_view = None
        conversation_id = record["conversation_id"]
        replies = self._take_mail(
            lambda m: (
                m.conversation_id == conversation_id
                and m.performative in (Performative.PROPOSE, Performative.REFUSE)
            )
        )

        bids: list[Bid] = []
        shadows: list[Bid] = []
        for reply in replies:
            car_id = int(reply.content["car_id"])
            if reply.performative is Performative.PROPOSE:
                bids.append(reply.content["bid"])
            else:
                reason = str(reply.content.get("reason", "refused"))
                bids.append(Bid(car_id=car_id, total=float("inf"), refused=True, reason=reason))
            shadow = reply.content.get("shadow")
            if isinstance(shadow, Bid):
                shadows.append(shadow)
        bids.sort(key=lambda b: b.car_id)
        shadows.sort(key=lambda b: b.car_id)
        shadow_bids = tuple(shadows) if shadows and len(shadows) == len(bids) else None
        award_bids = self._dagger_mix(bids, shadow_bids)

        viable = [b for b in award_bids if not b.refused]
        round_record = AuctionRound(
            tick=self.model.tick, conversation_id=conversation_id, call=call, bids=bids
        )

        if viable:
            # Tie-break on car id so the award is reproducible from the seed.
            ranked = sorted(viable, key=lambda b: (b.total, b.car_id))
            if self.model.award_hook is not None:
                chosen = self.model.award_hook(call, record, round_view, viable)
                ranked.sort(key=lambda b: b.car_id != chosen)  # stable: chosen first
            winner = ranked[0]
            round_record.winner = winner.car_id
            round_record.reason = explain_award(call, winner, ranked, award_bids)
            if self.model.strategy.uses_learned_bidder:
                round_record.reason += liftzero_trace(bids, shadow_bids, award_bids is not bids)
            self.assignments[call] = winner.car_id
            for bid in viable:
                performative = (
                    Performative.ACCEPT_PROPOSAL
                    if bid.car_id == winner.car_id
                    else Performative.REJECT_PROPOSAL
                )
                self.send(
                    performative,
                    f"car-{bid.car_id}",
                    conversation_id,
                    {"call": call, "cost": bid.total},
                )
            self.unassigned.pop(call, None)
        else:
            # Everyone refused (all full, or the fleet is in fire mode). Keep the call so
            # the floor's aging escalation brings it back rather than losing it.
            round_record.reason = "no car could take it: " + ", ".join(
                f"car {b.car_id} {b.reason}" for b in bids
            )
            self.unassigned[call] = record

        self.last_auction = round_record
        self.auction_history.append(round_record)
        if len(self.auction_history) > 50:
            del self.auction_history[:-50]
        self.model.collector.total_calls += 1

        if self.model.decision_hooks and round_view is not None:
            view_cars, view_policy = round_view
            event = DecisionEvent(
                tick=self.model.tick,
                call=call,
                urgency=record["urgency"],
                waiting=record["waiting"],
                statuses=view_cars,
                policy=view_policy,
                bids=tuple(bids),
                winner=winner.car_id if viable else None,
                building=self.model.config.building,
                seed=self.model.seed_value,
                strategy=self.model.strategy.name,
                bidder=self.model.strategy.bidder,
                shadow_bids=shadow_bids,
            )
            for hook in self.model.decision_hooks:
                hook(event)

    def _dagger_mix(self, bids: list[Bid], shadow_bids: tuple[Bid, ...] | None) -> list[Bid]:
        """The bids the award is decided on: usually the cars' own, but under DAgger mixing
        (``lift.beta`` > 0) the shadow teacher's with probability beta.

        Mixing keeps DAgger exploration safe: the learner drives most auctions (so the
        data shows the states *it* induces) while the teacher still steers a fraction.
        """
        beta = self.model.config.lift.beta
        if shadow_bids is None or beta <= 0.0 or self.model.dagger_rng is None:
            return bids
        if self.model.dagger_rng.random() < beta:
            self.teacher_awards += 1
            return list(shadow_bids)
        self.learner_awards += 1
        return bids

    # ------------------------------------------ periodic global reassignment (SA)

    def decide(self) -> None:
        """Obey any safety order, retry refused calls, then periodically re-optimise.

        Only safety orders are read here; a floor's REQUEST that arrived during this
        tick waits for tomorrow's `sense`, so every auction is run from one consistent
        snapshot of the building.
        """
        self._moved_this_tick = {}
        for message in self._take_mail(
            lambda m: (
                m.performative is Performative.REQUEST and isinstance(m.content.get("order"), Order)
            )
        ):
            self._obey(message.content["order"])
        if self.hall_calls_blocked:
            return

        board = self.model.board
        for call, record in list(self.unassigned.items()):
            if self.model.has_waiting(call.floor, call.direction):
                if board.available_car_ids(with_space=True):
                    self.unassigned.pop(call)
                    self.pending_calls[call] = record
            else:
                self.unassigned.pop(call)

        if self.model.tick >= self._next_reassign:
            self._next_reassign = self.model.tick + self.model.config.planner.reassign_interval
            if self.model.strategy.uses_reassignment:
                self.reassign()
            if self.model.strategy.uses_smart_parking:
                self.park_idle_cars()

    def reassignable(self) -> Assignment:
        """The assignments that may still be moved: calls nobody has picked up yet.

        Moving a call whose passengers are already aboard would be meaningless, and
        moving one the car is about to reach only churns the hall lantern — so this is
        deliberately restricted to open calls.
        """
        return {
            call: car_id
            for call, car_id in self.assignments.items()
            if call not in self.picked_up and self.model.has_waiting(call.floor, call.direction)
        }

    def assignment_cost(self, assignment: Assignment) -> float:
        """Objective for the local search: cheap insertion estimate, summed per car.

        This is deliberately an *estimate* rather than a full replan of every car: the
        search evaluates hundreds of candidate assignments per call, so it must be cheap.
        The quadratic load term is what spreads calls across the fleet instead of piling
        them onto whichever car happens to be closest.
        """
        per_car: dict[int, list[HallCall]] = {}
        for call, car_id in assignment.items():
            per_car.setdefault(car_id, []).append(call)

        total = 0.0
        seconds_per_floor = float(self.model.config.timing.seconds_per_floor)
        dwell = float(self.model.config.timing.door_open + self.model.config.timing.door_close)

        for car_id, calls in per_car.items():
            status = self.model.board.car(car_id)
            if status is None or not status.available:
                total += 1e6 * len(calls)  # an unusable car must never look attractive
                continue

            # Start from the work the car is already committed to, as it published it on
            # the status board. Ignoring that work was the mistake that made an earlier
            # version of this objective actively harmful: SA would pile new calls onto a
            # car that looked "close" while it still had a full load to deliver,
            # improving this estimate but worsening real waits.
            position = status.floor
            elapsed = status.plan_end_eta if status.plan_end_floor is not None else 0.0
            if status.plan_end_floor is not None:
                position = status.plan_end_floor

            # Order the calls the way the car will actually serve them: a directional
            # sweep from where it finishes its current plan, which is what its own A*
            # approximates. Agreeing with the car's real planner is what makes this
            # estimate a usable proxy at all.
            ordered = sorted(calls, key=lambda c: (abs(c.floor - position), c.floor))
            for call in ordered:
                elapsed += abs(call.floor - position) * seconds_per_floor + dwell
                weight = max(1.0, self.model.waiting_weight(call.floor, call.direction))
                age = max(0, self.model.tick - self.assignment_age(call))
                # Ageing calls are weighted up, so the optimiser also fights starvation.
                total += (elapsed + age) * weight
                position = call.floor

            # Penalise stacking calls on one car, counting the load it already carries.
            committed = len(status.car_calls) + len(calls)
            total += 3.0 * committed**2
        return total

    def assignment_age(self, call: HallCall) -> int:
        """When this call was first raised, for the ageing term in the objective."""
        return self.model.oldest_arrival(call.floor, call.direction)

    def reassign(self) -> None:
        """Re-optimise open assignments with simulated annealing.

        The new assignment is only adopted if it beats the current one by more than the
        hysteresis threshold. Without that margin the fleet would reshuffle constantly
        for fractions of a second of gain, and passengers would watch their hall lantern
        flicker between cars — which looks (and is) worse than a slightly stale plan.
        """
        current = self.reassignable()
        if len(current) < 2:
            return
        car_ids = self.model.board.available_car_ids(with_space=True)
        if len(car_ids) < 2:
            return

        planner = self.model.config.planner
        searcher = hill_climbing if self.model.strategy.reassignment == "hill_climbing" else None
        if searcher is hill_climbing:
            result = hill_climbing(
                current, car_ids, self.assignment_cost, self.model.random, iterations=80
            )
        else:
            result = simulated_annealing(
                current,
                car_ids,
                self.assignment_cost,
                self.model.random,
                iterations=planner.sa_iterations,
                initial_temp=planner.sa_initial_temp,
                cooling=planner.sa_cooling,
            )
        self.last_reassignment = result

        if result.improvement <= planner.hysteresis:
            return

        # Adopt it by message: CANCEL to the car that loses a call, ACCEPT_PROPOSAL to the
        # car that gains it. Each car carries the change out itself when it reads its
        # inbox later this tick, and the gaining car lights the new hall lantern.
        conversation_id = self.model.new_conversation_id()
        for call, new_car in result.assignment.items():
            old_car = current.get(call)
            if old_car == new_car:
                continue
            self.assignments[call] = new_car
            self._moved_this_tick[call] = (old_car, new_car)
            if old_car is not None:
                self.send(
                    Performative.CANCEL,
                    f"car-{old_car}",
                    conversation_id,
                    {"call": call, "reason": "global reassignment"},
                )
            self.send(
                Performative.ACCEPT_PROPOSAL,
                f"car-{new_car}",
                conversation_id,
                {"call": call, "reason": "global reassignment"},
            )

    # ------------------------------------------------------------ idle-car parking

    def park_idle_cars(self) -> None:
        """Advise idle cars where to wait, chosen by hill climbing or by minimax.

        Idleness is read from the status board, corrected for the reassignments this
        dispatcher has just ordered (it knows what it asked for, even though the cars
        have not read their mail yet). Advice goes out as INFORM messages; a car that no
        longer needs a parking spot is told so explicitly.
        """
        idle = [status for status in self.model.board.cars() if self._is_idle(status)]
        if not idle:
            self.last_parking = {}
            return

        # The monitor's advice wins when the strategy lets it learn, because it is derived
        # from the demand actually observed; the strategy's own setting is the fallback.
        policy = self.model.strategy.parking_policy or self.model.config.planner.parking_policy
        if self.model.strategy.adapts_weights and self.monitor_advice_received:
            policy = self.advised_parking_policy
        if policy is None:
            # The monitor has concluded that repositioning would not pay in this pattern.
            self.last_parking = {}
            for status in idle:
                self._advise_parking(status, None, "none")
            return
        floors = self.model.config.building.floors
        demand = self.model.board.demand()

        if policy == "lobby":
            targets = {status.car_id: self.model.lobby for status in idle}
        elif policy == "minimax":
            candidates = self._candidate_spots(floors, demand)
            likely = self._likely_call_floors(demand)
            result = minimax_parking(len(idle), floors, candidates, likely)
            self.last_minimax = result
            targets = {
                status.car_id: spot for status, spot in zip(idle, result.best_parking, strict=False)
            }
        else:
            positions, _cost, _curve = hill_climbing_parking(
                [status.car_id for status in idle], floors, demand, self.model.random
            )
            targets = positions

        self.last_parking = targets
        for status in idle:
            target = targets.get(status.car_id)
            self._advise_parking(
                status, target if target is not None and target != status.floor else None, policy
            )

    def _is_idle(self, status: Any) -> bool:
        """In service and without work, counting reassignments ordered this tick."""
        if not status.available or status.riders:
            return False
        calls = set(status.assigned_calls)
        for call, (old_car, new_car) in self._moved_this_tick.items():
            if old_car == status.car_id:
                calls.discard(call)
            if new_car == status.car_id:
                calls.add(call)
        return not calls

    def _advise_parking(self, status: Any, target: int | None, policy: str) -> None:
        """Tell one idle car where to park, or that it should stop heading anywhere."""
        if target is None and status.park_target is None:
            return  # nothing to change, so nothing to say
        self.send(
            Performative.INFORM,
            status.address,
            self.model.new_conversation_id(),
            {"park_at": target, "policy": policy},
        )

    def _candidate_spots(self, floors: int, demand: dict[int, float]) -> list[int]:
        """A small, spread-out set of parking floors for the minimax game.

        The game is exponential in the number of candidate spots, so this keeps the
        branching factor bounded — bounded rationality again, and it is documented as
        such in the design notes.
        """
        limit = self.model.config.planner.minimax_candidate_floors
        busiest = sorted(demand, key=lambda f: -demand[f])[: limit - 2]
        spread = {0, floors - 1, floors // 2, *busiest}
        return sorted(f for f in spread if 0 <= f < floors)[:limit]

    def _likely_call_floors(self, demand: dict[int, float]) -> list[int]:
        """The floors nature may choose in the parking game: the busiest ones."""
        limit = self.model.config.planner.minimax_candidate_floors
        ranked = sorted(demand, key=lambda f: -demand[f])
        likely = [f for f in ranked if demand[f] > 0][:limit]
        return likely or [0, self.model.config.building.floors - 1]

    # ------------------------------------------------------- safety-rule effects

    def block_hall_calls(self) -> None:
        """Fire mode: stop taking hall calls and forget the outstanding ones."""
        self.hall_calls_blocked = True
        self.pending_calls.clear()
        self.unassigned.clear()
        self.assignments.clear()

    def unblock_hall_calls(self) -> None:
        """Alarm cleared: accept hall calls again."""
        self.hall_calls_blocked = False

    # ------------------------------------------------------------------ reporting

    def describe(self) -> dict[str, Any]:
        """Agent-inspector payload."""
        base = super().describe()
        base.update(
            {
                "assignments": {
                    f"{c.floor}{c.direction.name[0]}": v for c, v in self.assignments.items()
                },
                "open_calls": len(self.reassignable()),
                "hall_calls_blocked": self.hall_calls_blocked,
                "last_reassignment": (
                    self.last_reassignment.as_dict() if self.last_reassignment else None
                ),
                "last_minimax": self.last_minimax.as_dict() if self.last_minimax else None,
                "parking": self.last_parking,
            }
        )
        return base

    def step(self) -> None:
        """Mesa's default hook; this agent is driven by the model's staged activation."""


def explain_award(call: HallCall, winner: Bid, ranked: list[Bid], bids: list[Bid]) -> str:
    """One line saying why the winner won: its cost, its biggest component, the margin.

    This is the decision trace the dashboard shows beside each auction, and the record
    later phases learn from, so it names numbers rather than adjectives.
    """
    parts = {
        "wait": winner.wait,
        "ride": winner.ride,
        "crowding": winner.crowding,
        "energy": winner.energy,
    }
    main = max(parts, key=lambda k: parts[k])
    text = (
        f"car {winner.car_id} wins {call.floor}{call.direction.name[0]} at cost "
        f"{winner.total:.1f} (mostly {main} {parts[main]:.1f}, ETA {winner.eta:.0f}s)"
    )
    if len(ranked) > 1:
        runner_up = ranked[1]
        text += f"; next best car {runner_up.car_id} at {runner_up.total:.1f}"
        text += f" (+{runner_up.total - winner.total:.1f})"
    refused = [b for b in bids if b.refused]
    if refused:
        text += "; " + ", ".join(f"car {b.car_id} refused ({b.reason})" for b in refused)
    return text


def liftzero_trace(
    bids: list[Bid], shadow_bids: tuple[Bid, ...] | None, teacher_awarded: bool
) -> str:
    """Decision-trace suffix for a learned auction: the net's bids and the teacher's view."""
    viable = sorted((b for b in bids if not b.refused), key=lambda b: (b.total, b.car_id))
    text = " [LiftZero] net bids: " + ", ".join(f"car {b.car_id}={b.total:.1f}" for b in viable[:3])
    if shadow_bids is not None:
        teacher = sorted(
            (b for b in shadow_bids if not b.refused), key=lambda b: (b.total, b.car_id)
        )
        if viable and teacher:
            if teacher[0].car_id == viable[0].car_id:
                text += "; teacher agrees"
            else:
                text += f"; teacher would pick car {teacher[0].car_id}"
    if teacher_awarded:
        text += " (awarded by teacher: DAgger mixing)"
    return text


def direction_of(call: HallCall) -> Direction:
    """The direction of a hall call (helper used by the API layer)."""
    return call.direction
