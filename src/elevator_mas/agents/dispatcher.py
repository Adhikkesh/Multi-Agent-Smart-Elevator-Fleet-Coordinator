"""The dispatcher: the coordinator and auctioneer of the fleet."""

from __future__ import annotations

from typing import Any

from elevator_mas.agents.base import CommunicatingAgent
from elevator_mas.comms import Performative
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

    PEAS
      - **P** fleet-wide average and 95th-percentile wait, % of long waits, fairness,
        energy, and recovery from failures.
      - **E** every floor, every car, the hall calls, and the traffic pattern.
      - **A** CFP / ACCEPT / REJECT / CANCEL messages, parking orders.
      - **S** REQUESTs from floors, PROPOSEs and REFUSEs from cars, FAILURE reports,
        traffic-pattern advice from the monitor.
    """

    agent_type = "Utility-based coordinator (auctioneer)"
    peas = {
        "performance": "fleet AWT/P95, % long waits, fairness, energy, fault recovery",
        "environment": "all floors, all cars, outstanding hall calls, traffic pattern",
        "actuators": "CFP/ACCEPT/REJECT/CANCEL messages, parking orders",
        "sensors": "floor REQUESTs, car PROPOSEs/REFUSEs, FAILUREs, monitor advice",
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

    # ------------------------------------------------------------------ sensing

    def sense(self) -> None:
        """Collect REQUESTs and FAILUREs, and drop calls that have been served."""
        self.collect_mail()
        for message in self.mail_of(Performative.REQUEST):
            call = message.content["call"]
            if self.hall_calls_blocked:
                continue
            record = {
                "call": call,
                "urgency": int(message.content.get("urgency", 0)),
                "conversation_id": message.conversation_id,
                "waiting": message.content.get("waiting", 1),
                "since": message.content.get("since", self.model.tick),
            }
            # An escalation supersedes the earlier request for the same call.
            self.pending_calls[call] = record

        for message in self.mail_of(Performative.INFORM):
            # The learning agent's advice. Acting on it is what makes the monitor an
            # agent rather than a read-out: it changes the parking policy the dispatcher
            # actually uses on the next cycle.
            if "parking_policy" in message.content:
                self.advised_parking_policy = message.content["parking_policy"]
                self.monitor_advice_received = True

        for message in self.mail_of(Performative.FAILURE):
            car_id = message.content.get("car_id")
            released = message.content.get("released", [])
            for call in released:
                if isinstance(call, HallCall):
                    self.assignments.pop(call, None)
                    self.pending_calls[call] = {
                        "call": call,
                        "urgency": 1,
                        "conversation_id": self.model.new_conversation_id(),
                        "waiting": self.model.waiting_count(call.floor, call.direction),
                        "since": self.model.tick,
                    }
            self.assignments = {c: v for c, v in self.assignments.items() if v != car_id}

        # Forget assignments whose calls no longer exist (everyone boarded).
        for call in list(self.assignments):
            if not self.model.has_waiting(call.floor, call.direction):
                self.assignments.pop(call, None)
                self.picked_up.discard(call)

    # ------------------------------------------------------- Contract Net round

    def communicate(self) -> None:
        """Run one Contract Net round for every outstanding call."""
        if self.hall_calls_blocked:
            self.pending_calls.clear()
            return
        for call, record in list(self.pending_calls.items()):
            self._run_auction(call, record)
            self.pending_calls.pop(call, None)

    def _run_auction(self, call: HallCall, record: dict[str, Any]) -> None:
        """One CFP → PROPOSE/REFUSE → ACCEPT/REJECT cycle.

        The bids are gathered synchronously here rather than across ticks. Every message
        is still created and logged, so the protocol the dashboard shows is real; doing
        it in one tick simply means a caller is never left waiting several seconds for
        the fleet to finish deliberating.

        Non-auction strategies reach this method too, but score the cars with their own
        simpler rule instead of asking for marginal-cost bids. Routing them through the
        same code keeps the comparison fair: identical protocol accounting, identical
        award and INFORM handling, only the scoring differs.
        """
        conversation_id = record["conversation_id"]
        urgency = record["urgency"]
        cars = self.model.cars
        assignment_rule = self.model.strategy.assignment

        self.send(
            Performative.CFP,
            None,  # broadcast to every car
            conversation_id,
            {"call": call, "urgency": urgency, "waiting": record["waiting"]},
        )

        bids: list[Bid] = []
        for car in cars:
            if assignment_rule == "nearest":
                bid = self._nearest_car_score(car, call)
            elif assignment_rule == "look":
                bid = self._collective_score(car, call)
            else:
                bid = car.marginal_cost(call, urgency)
            bids.append(bid)
            performative = Performative.REFUSE if bid.refused else Performative.PROPOSE
            content: dict[str, Any] = {"call": call, "car_id": car.car_id}
            if bid.refused:
                content["reason"] = bid.reason
            else:
                content["bid"] = bid
            car.send(performative, self.address, conversation_id, content)

        viable = [b for b in bids if not b.refused]
        round_record = AuctionRound(
            tick=self.model.tick, conversation_id=conversation_id, call=call, bids=bids
        )

        if viable:
            # Tie-break on car id so the award is reproducible from the seed.
            winner = min(viable, key=lambda b: (b.total, b.car_id))
            round_record.winner = winner.car_id
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
            car = self.model.car(winner.car_id)
            if car is not None:
                car.accept_call(call)
                car.send(
                    Performative.INFORM,
                    f"floor-{call.floor}",
                    conversation_id,
                    {
                        "car_id": car.car_id,
                        "direction": call.direction,
                        "eta": winner.eta,
                    },
                )
            self.unassigned.pop(call, None)
        else:
            # Everyone refused (all full, or the fleet is in fire mode). Keep the call so
            # the floor's aging escalation brings it back rather than losing it.
            self.unassigned[call] = record

        self.last_auction = round_record
        self.auction_history.append(round_record)
        if len(self.auction_history) > 50:
            del self.auction_history[:-50]
        self.model.collector.total_calls += 1

    def _nearest_car_score(self, car: Any, call: HallCall) -> Bid:
        """Nearest-car reflex score: distance only, with idle cars preferred.

        The classic baseline. It ignores what the car is already committed to, which is
        exactly its weakness: a car three floors away with eight stops still queued beats
        an empty car five floors away.
        """
        if not car.available:
            reason = "out of service" if car.out_of_service else "fire mode"
            return Bid(car_id=car.car_id, total=float("inf"), refused=True, reason=reason)
        if car.space <= 0:
            return Bid(car_id=car.car_id, total=float("inf"), refused=True, reason="full")
        distance = abs(car.floor - call.floor)
        busy_penalty = 0.0 if not car.assigned_calls and not car.riders else 8.0
        total = float(distance) + busy_penalty
        return Bid(
            car_id=car.car_id,
            total=total,
            wait=float(distance),
            crowding=busy_penalty,
            eta=distance * float(self.model.config.timing.seconds_per_floor),
        )

    def _collective_score(self, car: Any, call: HallCall) -> Bid:
        """Collective/LOOK score: strongly prefer a car already sweeping towards the call.

        This is how conventional lift control behaves — a call is picked up by a car that
        is coming that way anyway — and it is a much better baseline than nearest-car
        because it avoids sending cars backwards.
        """
        if not car.available:
            reason = "out of service" if car.out_of_service else "fire mode"
            return Bid(car_id=car.car_id, total=float("inf"), refused=True, reason=reason)
        if car.space <= 0:
            return Bid(car_id=car.car_id, total=float("inf"), refused=True, reason="full")

        distance = abs(car.floor - call.floor)
        seconds_per_floor = float(self.model.config.timing.seconds_per_floor)
        approaching = car.direction is Direction.IDLE or (
            car.direction is call.direction and (call.floor - car.floor) * car.direction.sign >= 0
        )
        # A car that must finish its sweep and come back pays for the whole detour.
        detour = 0.0 if approaching else 2.0 * self.model.config.building.floors
        queue_penalty = 2.0 * (len(car.assigned_calls) + len(car.car_calls))
        total = distance * seconds_per_floor + detour + queue_penalty
        return Bid(
            car_id=car.car_id,
            total=total,
            wait=distance * seconds_per_floor,
            ride=detour,
            crowding=queue_penalty,
            eta=(distance + detour / seconds_per_floor) * seconds_per_floor,
        )

    # ------------------------------------------ periodic global reassignment (SA)

    def decide(self) -> None:
        """Retry refused calls, then periodically re-optimise the whole assignment."""
        if self.hall_calls_blocked:
            return

        for call, record in list(self.unassigned.items()):
            if self.model.has_waiting(call.floor, call.direction):
                if any(car.available and car.space > 0 for car in self.model.cars):
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
            car = self.model.car(car_id)
            if car is None or not car.available:
                total += 1e6 * len(calls)  # an unusable car must never look attractive
                continue

            # Start from the work the car is already committed to. Ignoring it was the
            # mistake that made an earlier version of this objective actively harmful:
            # SA would pile new calls onto a car that looked "close" while it still had a
            # full load to deliver, improving this estimate but worsening real waits.
            position = car.floor
            elapsed = float(car.eta_to(car.planned_stops[-1].floor)) if car.planned_stops else 0.0
            if car.planned_stops:
                position = car.planned_stops[-1].floor

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
            committed = len(car.car_calls) + len(calls)
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
        car_ids = [car.car_id for car in self.model.cars if car.available and car.space > 0]
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

        conversation_id = self.model.new_conversation_id()
        for call, new_car in result.assignment.items():
            old_car = current.get(call)
            if old_car == new_car:
                continue
            self.assignments[call] = new_car
            old = self.model.car(old_car) if old_car is not None else None
            if old is not None:
                self.send(
                    Performative.CANCEL,
                    old.address,
                    conversation_id,
                    {"call": call, "reason": "global reassignment"},
                )
                old.drop_call(call)
            new = self.model.car(new_car)
            if new is not None:
                self.send(
                    Performative.ACCEPT_PROPOSAL,
                    new.address,
                    conversation_id,
                    {"call": call, "reason": "global reassignment"},
                )
                new.accept_call(call)
                new.send(
                    Performative.INFORM,
                    f"floor-{call.floor}",
                    conversation_id,
                    {
                        "car_id": new.car_id,
                        "direction": call.direction,
                        "eta": new.eta_to(call.floor),
                    },
                )

    # ------------------------------------------------------------ idle-car parking

    def park_idle_cars(self) -> None:
        """Send idle cars to floors chosen by hill climbing or by minimax."""
        idle = [
            car
            for car in self.model.cars
            if car.available and not car.assigned_calls and not car.riders
        ]
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
            for car in idle:
                car.park_target = None
            return
        floors = self.model.config.building.floors
        demand = self.model.monitor.demand_estimate()

        if policy == "lobby":
            targets = {car.car_id: self.model.lobby for car in idle}
        elif policy == "minimax":
            candidates = self._candidate_spots(floors, demand)
            likely = self._likely_call_floors(demand)
            result = minimax_parking(len(idle), floors, candidates, likely)
            self.last_minimax = result
            targets = {
                car.car_id: spot for car, spot in zip(idle, result.best_parking, strict=False)
            }
        else:
            positions, _cost, _curve = hill_climbing_parking(
                [car.car_id for car in idle], floors, demand, self.model.random
            )
            targets = positions

        self.last_parking = targets
        for car in idle:
            target = targets.get(car.car_id)
            if target is not None and target != car.floor:
                car.park_target = target
                self.send(
                    Performative.INFORM,
                    car.address,
                    self.model.new_conversation_id(),
                    {"park_at": target, "policy": policy},
                )
            else:
                car.park_target = None

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

    def reauction(self, calls: list[HallCall]) -> int:
        """Queue a failed car's calls for the next Contract Net round."""
        queued = 0
        for call in calls:
            if not self.model.has_waiting(call.floor, call.direction):
                continue
            self.assignments.pop(call, None)
            self.pending_calls[call] = {
                "call": call,
                "urgency": 1,  # a stranded caller has already waited: bump the urgency
                "conversation_id": self.model.new_conversation_id(),
                "waiting": self.model.waiting_count(call.floor, call.direction),
                "since": self.model.tick,
            }
            queued += 1
        return queued

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


def direction_of(call: HallCall) -> Direction:
    """The direction of a hall call (helper used by the API layer)."""
    return call.direction
