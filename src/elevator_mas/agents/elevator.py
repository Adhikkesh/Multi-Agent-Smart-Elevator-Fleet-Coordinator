"""The elevator car: a goal-based and utility-based agent."""

from __future__ import annotations

from typing import Any

from elevator_mas.agents.base import CommunicatingAgent
from elevator_mas.comms import Performative
from elevator_mas.domain import (
    Bid,
    CarState,
    Direction,
    DoorState,
    HallCall,
    Stop,
    StopKind,
)
from elevator_mas.planning.routing import RoutingCosts, plan_route


class ElevatorAgent(CommunicatingAgent):
    """One car: its own physics, its own plan, and its own bids.

    **AIMA agent type: goal-based *and* utility-based agent** (§2.4.4-2.4.5).

    *Goal-based*, because it does not react to the current percept: it holds a set of
    stops it must reach and searches (A*, in `planning/routing.py`) for a sequence of
    actions that achieves them. Change the goals and the behaviour follows without any
    rule being rewritten.

    *Utility-based*, because "reach the goals" does not say *which* plan to prefer, and
    the plans differ in kind, not just in degree: one is quicker for the people aboard,
    another kinder to the crowd waiting downstairs, another cheaper in electricity. So
    the car scores outcomes with a scalar utility — the weighted sum of wait, ride,
    crowding and energy — and that same function is what it bids with. Its bid is the
    *marginal* cost of inserting a call into its existing plan, which is what makes the
    Contract Net auction meaningful: each car reports its true opportunity cost.

    PEAS
      - **P** wait and ride time of the people it serves, energy used, never exceeding
        capacity, never moving with its doors open.
      - **E** its shaft, the floors, its passengers, and the other cars it competes with.
      - **A** motor up/down/stop, doors open/close, car-button lamps, messages.
      - **S** position encoder, load sensor, door sensor, car buttons, messages.
    """

    agent_type = "Goal-based + utility-based agent"
    peas = {
        "performance": "wait/ride time served, energy, capacity and door safety",
        "environment": "its shaft, floors, its passengers, the other cars",
        "actuators": "motor up/down/stop, doors, car-button lamps, messages",
        "sensors": "position encoder, load sensor, door sensor, car buttons, messages",
    }

    def __init__(self, model: Any, car_id: int) -> None:
        super().__init__(model, address=f"car-{car_id}")
        self.car_id = car_id
        config = model.config
        self.capacity = config.building.capacity
        self.timing = config.timing

        # --- internal model (this is what makes it more than a reflex agent) ---
        self.floor: int = config.building.lobby
        self.direction: Direction = Direction.IDLE
        self.door_state: DoorState = DoorState.CLOSED
        self.state: CarState = CarState.IDLE
        self.assigned_calls: set[HallCall] = set()
        self.car_calls: set[int] = set()
        self.riders: list[Any] = []
        self.planned_stops: list[Stop] = []
        self.out_of_service: bool = False
        self.fire_mode: bool = False
        self.boarding_refused: bool = False
        self.door_blocked_ticks: int = 0
        self.park_target: int | None = None

        # --- physics timers, in ticks (1 tick == 1 simulated second) ---
        self._move_timer: int = 0
        self._door_timer: int = 0
        self._dwell_remaining: int = 0
        self._target_floor: int | None = None

        # --- bookkeeping for the dashboard and the metrics ---
        self.last_plan_nodes: int = 0
        self.last_plan_cost: float = 0.0
        self.replans: int = 0
        self.needs_replan: bool = True
        self.last_bid: Bid | None = None
        self.floors_travelled: int = 0
        self.stops_made: int = 0
        self.reversals: int = 0

    # ---------------------------------------------------------------- properties

    @property
    def load(self) -> int:
        """How many people are aboard."""
        return len(self.riders)

    @property
    def rider_weight(self) -> float:
        """Summed weight aboard, so a VIP counts for more in the cost function."""
        return sum(r.weight for r in self.riders)

    @property
    def space(self) -> int:
        """Remaining places, zero when boarding is refused."""
        if self.boarding_refused or self.out_of_service or self.fire_mode:
            return 0
        return max(0, self.capacity - self.load)

    @property
    def moving(self) -> bool:
        """True while travelling between floors."""
        return self._target_floor is not None and self._move_timer > 0

    @property
    def available(self) -> bool:
        """Whether this car may take on new hall calls at all."""
        return not (self.out_of_service or self.fire_mode)

    @property
    def search_limit(self) -> int:
        """Stop count above which routing falls back to the LOOK sweep.

        A strategy whose `routing` is "look" sets this to zero, so every plan uses the
        sweep and the A* search never runs — that is what makes the benchmark's
        A*-versus-LOOK comparison a real comparison rather than a relabelling.
        """
        if self.model.strategy.routing == "look":
            return 0
        return self.model.config.planner.max_stops_for_search

    @property
    def routing_costs(self) -> RoutingCosts:
        """Timing and energy constants for the routing search."""
        planner = self.model.config.planner
        return RoutingCosts(
            seconds_per_floor=float(self.timing.seconds_per_floor),
            dwell=float(self.timing.door_open + self.timing.door_close),
            energy_per_floor=planner.energy_per_floor,
            energy_per_stop=planner.energy_per_stop,
        )

    # ------------------------------------------------------------------- goals

    def prune_stale_calls(self) -> None:
        """Forget assigned pickups that nobody is waiting for any more.

        A call can go stale because another car picked its passengers up, or because they
        boarded this car in the opposite direction. Keeping it would make the car stop for
        nobody and inflate its own bids, so it is dropped here every tick.
        """
        stale = {
            call
            for call in self.assigned_calls
            if not self.model.has_waiting(call.floor, call.direction)
        }
        if stale:
            self.assigned_calls -= stale
            self.needs_replan = True

    def pending_stops(self) -> list[Stop]:
        """The car's current goal set: its assigned pickups and its riders' drop-offs.

        Drop-off weight is the number of riders bound for that floor, and pickup weight
        is the crowd waiting — so the routing cost naturally prioritises the stops that
        relieve the most people.
        """
        stops: list[Stop] = []
        for floor in sorted(self.car_calls):
            riders_for = sum(r.weight for r in self.riders if r.destination == floor)
            stops.append(
                Stop(
                    floor=floor,
                    kind=StopKind.DROPOFF,
                    direction=Direction.IDLE,
                    weight=max(1.0, riders_for),
                )
            )
        for call in sorted(self.assigned_calls, key=lambda c: (c.floor, c.direction.value)):
            weight = self.model.waiting_weight(call.floor, call.direction)
            stops.append(
                Stop(
                    floor=call.floor,
                    kind=StopKind.PICKUP,
                    direction=call.direction,
                    weight=max(1.0, weight),
                )
            )
        return stops

    def replan(self) -> None:
        """Re-run the routing search and cache the stop sequence it returns.

        Replanning is event-driven, not per-tick: `needs_replan` is set when the goal set
        actually changes (a call is won or cancelled, someone boards or alights, a fault
        occurs). Searching every tick would be pure waste, and the `stress_scale`
        scenario's time budget depends on this.
        """
        stops = self.pending_stops()
        planner = self.model.config.planner
        route, result = plan_route(
            self.floor,
            self.direction,
            stops,
            self.routing_costs,
            algorithm=planner.algorithm,
            max_stops_for_search=self.search_limit,
            boarding_per_passenger=float(self.timing.boarding_per_passenger),
        )
        self.planned_stops = route
        self.needs_replan = False
        self.replans += 1
        if result is not None and result.found:
            self.last_plan_nodes = result.nodes_expanded
            self.last_plan_cost = result.cost
            self.model.collector.total_nodes_expanded += result.nodes_expanded
        self.model.collector.total_replans += 1

    def marginal_cost(self, call: HallCall, urgency: int = 0) -> Bid:
        """Bid = the *extra* utility cost of inserting `call` into the current plan.

        Planning with and without the call and taking the difference is what makes this a
        true marginal (opportunity) cost: a car already passing the floor bids nearly
        nothing, while a car that would have to reverse bids a lot. That is precisely the
        information the Contract Net auction needs in order to pick well.

        The four components are reported separately so the dashboard can show *why* a car
        won, and they are weighted by W1-W4, which the TrafficMonitorAgent retunes as the
        traffic pattern changes.
        """
        weights = self.model.weights
        if not self.available:
            reason = "out of service" if self.out_of_service else "fire mode"
            return Bid(car_id=self.car_id, total=float("inf"), refused=True, reason=reason)
        if self.space <= 0:
            return Bid(car_id=self.car_id, total=float("inf"), refused=True, reason="full")

        costs = self.routing_costs
        current_stops = self.pending_stops()
        planner = self.model.config.planner

        base_route, base_result = plan_route(
            self.floor,
            self.direction,
            current_stops,
            costs,
            algorithm=planner.algorithm,
            max_stops_for_search=self.search_limit,
            boarding_per_passenger=float(self.timing.boarding_per_passenger),
        )
        base_cost = (
            base_result.cost
            if base_result and base_result.found
            else self._route_cost(base_route, costs)
        )

        waiting_weight = max(1.0, self.model.waiting_weight(call.floor, call.direction))
        candidate_stops = [
            *current_stops,
            Stop(
                floor=call.floor,
                kind=StopKind.PICKUP,
                direction=call.direction,
                weight=waiting_weight,
            ),
        ]
        new_route, new_result = plan_route(
            self.floor,
            self.direction,
            candidate_stops,
            costs,
            algorithm=planner.algorithm,
            max_stops_for_search=self.search_limit,
            boarding_per_passenger=float(self.timing.boarding_per_passenger),
        )
        new_cost = (
            new_result.cost
            if new_result and new_result.found
            else self._route_cost(new_route, costs)
        )
        if new_result is not None:
            self.model.collector.total_nodes_expanded += new_result.nodes_expanded

        eta = self._eta_to(new_route, call.floor, costs)
        delta = max(0.0, new_cost - base_cost)

        wait_term = eta * waiting_weight
        ride_term = max(0.0, delta - wait_term) if self.riders else 0.0
        crowding_term = (self.load / self.capacity) * 10.0 if self.capacity else 0.0
        energy_term = costs.energy_per_floor * abs(call.floor - self.floor) + costs.energy_per_stop

        total = (
            weights.wait * wait_term
            + weights.ride * ride_term
            + weights.crowding * crowding_term
            + weights.energy * energy_term
        )
        # Aging discount: an escalated call becomes cheaper for everyone to take, so a
        # starving floor eventually outbids fresher, closer calls.
        total = max(0.0, total - urgency * self.model.config.fairness.escalation_bonus)

        bid = Bid(
            car_id=self.car_id,
            total=total,
            wait=weights.wait * wait_term,
            ride=weights.ride * ride_term,
            crowding=weights.crowding * crowding_term,
            energy=weights.energy * energy_term,
            eta=eta,
        )
        self.last_bid = bid
        return bid

    def _route_cost(self, route: list[Stop], costs: RoutingCosts) -> float:
        """Cost of a route produced by the LOOK fallback, scored the same way as a plan."""
        total = 0.0
        position = self.floor
        elapsed = 0.0
        remaining = sum(s.weight for s in route)
        for stop in route:
            duration = costs.travel(position, stop.floor) + costs.dwell
            elapsed += duration
            total += (
                duration * remaining
                + costs.energy_per_floor * abs(stop.floor - position)
                + costs.energy_per_stop
            )
            remaining -= stop.weight
            position = stop.floor
        return total

    def eta_to(self, floor: int, route: list[Stop] | None = None) -> float:
        """Seconds until the car reaches `floor`, along `route` or its current plan."""
        return self._eta_to(
            self.planned_stops if route is None else route, floor, self.routing_costs
        )

    def _eta_to(self, route: list[Stop], floor: int, costs: RoutingCosts) -> float:
        """Seconds until the car reaches `floor` along `route`."""
        elapsed = float(self._move_timer + self._dwell_remaining)
        position = self.floor
        for stop in route:
            elapsed += costs.travel(position, stop.floor)
            position = stop.floor
            if stop.floor == floor:
                return elapsed
            elapsed += costs.dwell
        return elapsed + costs.travel(position, floor)

    # ------------------------------------------------------------ communication

    def sense(self) -> None:
        """Read the inbox and drop goals that reality has already satisfied."""
        self.collect_mail()
        self.prune_stale_calls()

    def communicate(self) -> None:
        """Answer every call for proposals, and honour awards and cancellations."""
        for message in self.inbox:
            performative = message.performative
            content = message.content
            if performative is Performative.CFP:
                # The dispatcher polls each car within its own auction round and posts
                # that car's PROPOSE/REFUSE on its behalf, so the reply already exists by
                # the time the broadcast CFP is drained here. Answering it again would put
                # two proposals per car into the log and double the messages-per-call
                # metric, so this branch deliberately does nothing.
                continue
            if performative is Performative.ACCEPT_PROPOSAL:
                call = content["call"]
                self.accept_call(call)
                self.send(
                    Performative.INFORM,
                    f"floor-{call.floor}",
                    message.conversation_id,
                    {
                        "car_id": self.car_id,
                        "direction": call.direction,
                        "eta": self.eta_to(call.floor),
                    },
                )
            elif performative is Performative.CANCEL:
                call = content.get("call")
                if isinstance(call, HallCall):
                    self.drop_call(call)

    def accept_call(self, call: HallCall) -> None:
        """Take responsibility for a hall call and replan."""
        self.assigned_calls.add(call)
        self.needs_replan = True
        self.replan()

    def drop_call(self, call: HallCall) -> None:
        """Give up a hall call (lost in a reassignment, or blocked by fire mode)."""
        if call in self.assigned_calls:
            self.assigned_calls.discard(call)
            self.needs_replan = True

    # --------------------------------------------------------------- decide/act

    def decide(self) -> None:
        """Choose the next target floor from the plan, or a parking floor when idle."""
        if self.out_of_service:
            self.state = CarState.OUT_OF_SERVICE
            return
        if self.needs_replan:
            self.replan()

        if self.fire_mode:
            self.state = CarState.FIRE_RECALL
            if self.floor != self.model.lobby and not self.moving:
                self._target_floor = self.model.lobby
            return

        if self._dwell_remaining > 0 or self.moving:
            return

        next_stop = self.planned_stops[0] if self.planned_stops else None
        if next_stop is not None:
            self._target_floor = next_stop.floor
            if next_stop.floor != self.floor:
                self.direction = Direction.UP if next_stop.floor > self.floor else Direction.DOWN
        elif self.park_target is not None and self.park_target != self.floor:
            self._target_floor = self.park_target
            self.direction = Direction.UP if self.park_target > self.floor else Direction.DOWN
        else:
            self._target_floor = None
            self.direction = Direction.IDLE

    def act(self) -> None:
        """Advance the physics by one tick: doors, then motion, then boarding.

        The ordering here is what enforces the safety invariant the tests assert: a car
        only ever moves in the `CLOSED` branch, so the doors can never be open while it
        is in motion.
        """
        if self.out_of_service:
            self.state = CarState.OUT_OF_SERVICE
            self.direction = Direction.IDLE
            return

        if self.door_state is not DoorState.CLOSED:
            self._advance_doors()
            return

        if self._target_floor is not None and self._target_floor != self.floor:
            self._advance_motion()
            return

        # Standing at a floor with the doors shut: open them if this is a real stop.
        if self._should_serve_here():
            self._begin_door_cycle()
        else:
            self.state = CarState.IDLE
            if not self.planned_stops:
                self.direction = Direction.IDLE

    def _should_serve_here(self) -> bool:
        """Whether this floor is a stop the car must open its doors for."""
        if self.fire_mode:
            return self.floor == self.model.lobby
        if self.planned_stops and self.planned_stops[0].floor == self.floor:
            return True
        return self.floor in self.car_calls

    def _begin_door_cycle(self) -> None:
        """Start opening the doors and stop counting this floor as travel."""
        self.door_state = DoorState.OPENING
        self._door_timer = self.timing.door_open
        self.state = CarState.DOORS
        self.stops_made += 1
        self.model.collector.energy.stops += 1

    def _advance_doors(self) -> None:
        """Run the door state machine one tick, handling alighting and boarding."""
        self.state = CarState.FIRE_RECALL if self.fire_mode else CarState.DOORS
        self._door_timer -= 1
        if self._door_timer > 0:
            return

        if self.door_state is DoorState.OPENING:
            self.door_state = DoorState.OPEN
            self._alight()
            dwell = self._board()
            # At least one tick, always: a zero-length dwell would leave the timer at 0
            # and the car parked in OPEN forever, which is how an earlier version of this
            # method deadlocked a drained building.
            self._dwell_remaining = max(1, self.timing.dwell_min, dwell)
            self._door_timer = self._dwell_remaining
            self._clear_served_stop()
            return

        if self.door_state is DoorState.OPEN:
            if self.fire_mode and self.floor == self.model.lobby:
                self._door_timer = 1  # stay open for evacuation
                return
            # Someone arrived while the doors were open: take them too.
            extra = self._board()
            if extra > 0:
                self._door_timer = extra
                return
            self.door_state = DoorState.CLOSING
            self._door_timer = max(1, self.timing.door_close)
            return

        if self.door_state is DoorState.CLOSING:
            self.door_state = DoorState.CLOSED
            self._dwell_remaining = 0
            self.boarding_refused = False
            self.needs_replan = True

    def _advance_motion(self) -> None:
        """Move one tick towards the target floor."""
        target = self._target_floor
        if target is None:
            return
        new_direction = Direction.UP if target > self.floor else Direction.DOWN
        if self.direction is not Direction.IDLE and new_direction is not self.direction:
            self.reversals += 1
            self.model.collector.energy.reversals += 1
        self.direction = new_direction
        self.state = CarState.MOVING_UP if new_direction is Direction.UP else CarState.MOVING_DOWN

        if self._move_timer <= 0:
            self._move_timer = self.timing.seconds_per_floor
        self._move_timer -= 1
        if self._move_timer <= 0:
            self.floor += new_direction.sign
            self.floors_travelled += 1
            self.model.collector.energy.floors_travelled += 1
            if self.floor == target:
                self._target_floor = None

    def _alight(self) -> None:
        """Let everyone bound for this floor out."""
        leaving = [r for r in self.riders if r.destination == self.floor]
        for rider in leaving:
            self.riders.remove(rider)
            rider.alight(self.model.tick)
            self.model.on_alight(rider)
        self.car_calls.discard(self.floor)
        if leaving:
            self.needs_replan = True

    def _board(self) -> int:
        """Board whoever is waiting for this car's direction, and press their buttons.

        This is where partial observability is resolved: only now does the car learn the
        destinations, so it must replan with strictly more information than it bid with.
        """
        if self.fire_mode or self.out_of_service or self.boarding_refused:
            return 0
        boarded = 0
        directions = self._serving_directions()
        for passenger in self.model.waiting_at(self.floor):
            if self.space <= 0:
                break
            if passenger.direction not in directions:
                continue
            passenger.board(self.car_id, self.model.tick)
            self.riders.append(passenger)
            self.car_calls.add(passenger.destination)  # the car button
            self.model.on_board(passenger, self.car_id)
            boarded += 1
        if boarded:
            self.needs_replan = True
            for direction in directions:
                self.assigned_calls.discard(HallCall(self.floor, direction))
        return boarded * self.timing.boarding_per_passenger

    def _serving_directions(self) -> set[Direction]:
        """Which hall calls this car is announcing at the current floor.

        Its doors are open anyway, so refusing someone standing there who wants to travel
        the way the car is already going costs them a whole extra round trip and saves the
        car nothing. Real collective control picks them up, and so does this.

        The subtle case is a car that has stopped purely to let riders out: it has no
        pickup planned here and its direction may momentarily read IDLE because it just
        reached its target. Falling through to "no directions" there was a real bug — in
        down-peak traffic, where nearly every stop is a drop-off, it made the fleet refuse
        most of the people waiting and lose to the reflex baseline.
        """
        planned = {
            stop.direction
            for stop in self.planned_stops
            if stop.floor == self.floor and stop.is_pickup
        }
        assigned = {c.direction for c in self.assigned_calls if c.floor == self.floor}
        directions = planned | assigned
        if directions:
            return directions

        # Nothing was scheduled here, but the doors are open. Take whoever fits.
        if not self.riders:
            # Empty car: it is free to serve either direction from here.
            return {Direction.UP, Direction.DOWN}

        # Riders aboard: only take people going the way the car must continue, so nobody
        # already on board is carried backwards. Infer that from the remaining plan when
        # the direction flag has been cleared.
        onward = self._onward_direction()
        return {onward} if onward is not Direction.IDLE else set()

    def _onward_direction(self) -> Direction:
        """The direction the car will travel next, from its plan if the flag is cleared."""
        if self.direction is not Direction.IDLE:
            return self.direction
        for stop in self.planned_stops:
            if stop.floor > self.floor:
                return Direction.UP
            if stop.floor < self.floor:
                return Direction.DOWN
        return Direction.IDLE

    def _clear_served_stop(self) -> None:
        """Drop the stop the car has just served from the cached plan."""
        if self.planned_stops and self.planned_stops[0].floor == self.floor:
            self.planned_stops.pop(0)

    # ------------------------------------------------------- safety-rule effects

    def enter_fire_mode(self, lobby: int) -> None:
        """Fire recall: abandon every hall call and run to the lobby."""
        self.fire_mode = True
        self.state = CarState.FIRE_RECALL
        self.assigned_calls.clear()
        self.planned_stops = []
        self.park_target = None
        self._target_floor = lobby if self.floor != lobby else None
        self.needs_replan = False

    def hold_doors_open(self) -> None:
        """Keep the doors open (fire mode at the lobby)."""
        if self.door_state is not DoorState.OPEN:
            self.door_state = DoorState.OPEN
        self._door_timer = max(self._door_timer, 1)
        self.state = CarState.FIRE_RECALL
        self._alight_all_at_floor()

    def _alight_all_at_floor(self) -> None:
        """Let everyone out during an evacuation, wherever they were going."""
        for rider in list(self.riders):
            self.riders.remove(rider)
            rider.record.alight_tick = self.model.tick
            rider.boarded = False
            rider.current_floor = self.floor
            self.model.on_alight(rider)
        self.car_calls.clear()

    def refuse_boarding(self) -> None:
        """Overload: hold the doors and stop letting anyone else in."""
        self.boarding_refused = True
        if self.door_state is DoorState.CLOSING:
            self.door_state = DoorState.OPEN
            self._door_timer = 1

    def reopen_doors(self) -> None:
        """Door obstruction: restart the door cycle."""
        self.door_state = DoorState.OPENING
        self._door_timer = self.timing.door_open
        self.door_blocked_ticks = 0

    def go_out_of_service(self) -> list[HallCall]:
        """Fail: stop at the next floor, turn the riders out, release the calls.

        Returns the calls that must be re-auctioned, which is what the dispatcher needs
        in order to recover the service rather than silently losing those passengers.
        """
        self.out_of_service = True
        self.state = CarState.OUT_OF_SERVICE
        released = sorted(self.assigned_calls, key=lambda c: (c.floor, c.direction.value))
        self.assigned_calls.clear()
        self.planned_stops = []
        self._target_floor = None
        self._move_timer = 0
        self.direction = Direction.IDLE
        self.door_state = DoorState.OPEN
        self._door_timer = 1
        for rider in list(self.riders):
            self.riders.remove(rider)
            rider.requeue(self.floor, self.model.tick)
            self.model.on_requeue(rider)
        self.car_calls.clear()
        return released

    def return_to_service(self) -> None:
        """Repair: come back as an empty, idle, closed car."""
        self.out_of_service = False
        self.fire_mode = False
        self.boarding_refused = False
        self.state = CarState.IDLE
        self.door_state = DoorState.CLOSED
        self.direction = Direction.IDLE
        self._door_timer = 0
        self._dwell_remaining = 0
        self.needs_replan = True

    def restore_normal_service(self) -> None:
        """Leave fire mode and resume ordinary dispatch."""
        self.fire_mode = False
        self.state = CarState.IDLE
        self.door_state = DoorState.CLOSED
        self._door_timer = 0
        self.needs_replan = True

    # ------------------------------------------------------------------ reporting

    def snapshot(self) -> dict[str, Any]:
        """What the dashboard draws for this car."""
        return {
            "car_id": self.car_id,
            "floor": self.floor,
            "direction": self.direction.name,
            "door": self.door_state.value,
            "state": self.state.value,
            "load": self.load,
            "capacity": self.capacity,
            "out_of_service": self.out_of_service,
            "fire_mode": self.fire_mode,
            "route": [
                {"floor": s.floor, "kind": s.kind.value, "direction": s.direction.name}
                for s in self.planned_stops
            ],
            "car_calls": sorted(self.car_calls),
            "assigned": [
                {"floor": c.floor, "direction": c.direction.name}
                for c in sorted(self.assigned_calls, key=lambda c: (c.floor, c.direction.value))
            ],
            "progress": self._progress(),
            "plan_cost": round(self.last_plan_cost, 2),
            "plan_nodes": self.last_plan_nodes,
        }

    def _progress(self) -> float:
        """Fraction of the way to the next floor, so the UI can interpolate smoothly."""
        if self._target_floor is None or self._move_timer <= 0:
            return 0.0
        total = max(1, self.timing.seconds_per_floor)
        return max(0.0, min(1.0, (total - self._move_timer) / total))

    def describe(self) -> dict[str, Any]:
        """Agent-inspector payload: type, PEAS, internal state and planned route."""
        base = super().describe()
        base.update(
            {
                "car_id": self.car_id,
                "internal_state": {
                    "floor": self.floor,
                    "direction": self.direction.name,
                    "door_state": self.door_state.value,
                    "load": f"{self.load}/{self.capacity}",
                    "riders_to": sorted(self.car_calls),
                    "assigned_calls": [
                        f"{c.floor}{c.direction.name[0]}" for c in self.assigned_calls
                    ],
                    "out_of_service": self.out_of_service,
                    "fire_mode": self.fire_mode,
                },
                "planned_route": [f"{s.floor}:{s.kind.value}" for s in self.planned_stops],
                "plan_cost": round(self.last_plan_cost, 2),
                "plan_nodes_expanded": self.last_plan_nodes,
                "replans": self.replans,
                "last_bid": self.last_bid.as_dict() if self.last_bid else None,
                "energy": {
                    "floors_travelled": self.floors_travelled,
                    "stops": self.stops_made,
                    "reversals": self.reversals,
                },
            }
        )
        return base

    def step(self) -> None:
        """Mesa's default hook; this agent is driven by the model's staged activation."""
