"""The Mesa model: the environment, the agent population, and the tick loop.

Mesa 3 API only — `Agent.__init__(self, model, ...)`, no schedulers, staged activation
through `model.agents_by_type[Cls].do(stage)`, `agent.remove()`, and every random draw
from `model.random`, so a run is reproducible from its seed alone.

**Environment classification** (justified in `docs/DESIGN.md` and shown in the UI):

- *Partially observable* — a passenger's destination is private until they board, and each
  agent sees only its own percepts and inbox.
- *Multi-agent, cooperative* — the cars compete in auctions but share one objective.
- *Stochastic* — Poisson arrivals with a hidden, time-varying rate, plus injected faults.
- *Sequential* — an assignment now changes what is optimal for the next ten minutes.
- *Dynamic* — demand keeps arriving while the agents deliberate.
- *Discrete* — floors and one-second ticks, discretised from continuous motion.
- *Known physics, unknown demand* — the agents know how lifts move; the arrival process
  has to be learned online.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

from mesa import Model
from mesa.datacollection import DataCollector

from elevator_mas.agents import (
    DispatcherAgent,
    ElevatorAgent,
    FloorAgent,
    PassengerAgent,
    SafetyAgent,
    TrafficMonitorAgent,
)
from elevator_mas.comms import MessageBus, StatusBoard
from elevator_mas.config import CostWeights, ScenarioConfig
from elevator_mas.domain import CarState, Direction
from elevator_mas.metrics import Metrics, MetricsCollector
from elevator_mas.strategies import DispatchStrategy, get_strategy
from elevator_mas.traffic import ArrivalGenerator

#: The fixed order of the tick. Every agent of one type completes a stage before the next
#: stage begins, so within a tick nobody acts on a world another agent has already
#: changed — which is what makes the run deterministic given the seed.
#:
#: ``negotiate`` is the Contract Net, expanded into its own sub-stages for each queued
#: call: the dispatcher *announces* (CFP), every car *bids* from its own inbox
#: (PROPOSE/REFUSE), the dispatcher *awards* (ACCEPT/REJECT), and every car *commits*
#: (the winner takes the call and INFORMs the landing). In the course's vocabulary:
#: sense → reason (communicate + negotiate) → decide → act → learn.
STAGES: tuple[str, ...] = ("sense", "communicate", "negotiate", "decide", "act", "learn")

#: Within a stage, agent types run in this order. Safety leads so that its orders bind
#: everyone later in the same stage; cars run last so they read every message addressed
#: to them in that stage.
AGENT_ORDER = (SafetyAgent, TrafficMonitorAgent, FloorAgent, DispatcherAgent, ElevatorAgent)


class ElevatorModel(Model):
    """The building, its fleet, and the agents coordinating it."""

    def __init__(self, config: ScenarioConfig | None = None, seed: int | None = None) -> None:
        config = config or ScenarioConfig(name="default")
        # Mesa 3.5 takes `rng=`; it seeds both `self.random` and `self.rng`.
        super().__init__(rng=seed if seed is not None else config.seed)
        self.config = config
        self.seed_value = seed if seed is not None else config.seed
        self.strategy: DispatchStrategy = get_strategy(config.strategy)

        self.tick: int = 0
        self.lobby = config.building.lobby
        self.floors_count = config.building.floors

        self.bus = MessageBus()
        #: The public blackboard: cars publish their status, the monitor its policy.
        self.board = StatusBoard(config.weights)
        self.negotiation_rounds: int = 0
        self.collector = MetricsCollector(long_wait_threshold=config.fairness.long_wait_threshold)
        self.generator = ArrivalGenerator(config, self.random)

        self._conversation_ids = itertools.count(1)
        self.arrivals_this_tick: list[PassengerAgent] = []
        self.events_fired: list[dict[str, Any]] = []
        self._pending_events = sorted(config.events, key=lambda e: e.tick)
        self.compute_ms_last_tick: float = 0.0
        self.latest_metrics: Metrics = Metrics()

        # --- agent population ---
        self.floors: list[FloorAgent] = [
            FloorAgent(self, floor) for floor in range(self.floors_count)
        ]
        self.cars: list[ElevatorAgent] = [
            ElevatorAgent(self, car_id) for car_id in range(config.building.cars)
        ]
        self.monitor = TrafficMonitorAgent(self)
        self.dispatcher = DispatcherAgent(self)
        self.safety = SafetyAgent(self)

        self.passengers: list[PassengerAgent] = []
        for car in self.cars:
            car.publish_status()

        self.datacollector = DataCollector(
            model_reporters={
                "tick": lambda m: m.tick,
                "avg_wait": lambda m: m.latest_metrics.avg_wait,
                "p95_wait": lambda m: m.latest_metrics.p95_wait,
                "max_wait": lambda m: m.latest_metrics.max_wait,
                "avg_ride": lambda m: m.latest_metrics.avg_ride,
                "avg_system": lambda m: m.latest_metrics.avg_system,
                "long_wait_pct": lambda m: m.latest_metrics.long_wait_pct,
                "delivered": lambda m: m.latest_metrics.delivered,
                "waiting": lambda m: m.latest_metrics.waiting,
                "riding": lambda m: m.latest_metrics.riding,
                "throughput": lambda m: m.latest_metrics.throughput,
                "energy": lambda m: m.latest_metrics.energy,
                "messages": lambda m: m.bus.total_sent,
                "nodes_expanded": lambda m: m.collector.total_nodes_expanded,
                "rules_fired": lambda m: m.collector.total_rules_fired,
                "pattern": lambda m: m.monitor.pattern,
                "compute_ms": lambda m: m.compute_ms_last_tick,
            }
        )

    # ------------------------------------------------------------------- helpers

    @property
    def weights(self) -> CostWeights:
        """The cost weights the fleet currently bids with, as published on the board."""
        return self.board.weights

    @weights.setter
    def weights(self, value: CostWeights) -> None:
        self.board.publish_policy("model", self.tick, weights=value)

    def new_conversation_id(self) -> str:
        """A fresh conversation id, threading one protocol round together."""
        return f"c{next(self._conversation_ids)}"

    def car(self, car_id: int | None) -> ElevatorAgent | None:
        """A car by id, or None."""
        if car_id is None:
            return None
        for car in self.cars:
            if car.car_id == car_id:
                return car
        return None

    def waiting_at(self, floor: int) -> list[PassengerAgent]:
        """Everyone currently waiting on a landing."""
        return [p for p in self.passengers if p.waiting and p.current_floor == floor]

    def waiting_count(self, floor: int, direction: Direction) -> int:
        """How many people are waiting on a floor for a direction."""
        return sum(1 for p in self.waiting_at(floor) if p.direction is direction)

    def waiting_weight(self, floor: int, direction: Direction) -> float:
        """Summed weight waiting on a floor for a direction (VIPs count for more)."""
        return sum(p.weight for p in self.waiting_at(floor) if p.direction is direction)

    def has_waiting(self, floor: int, direction: Direction) -> bool:
        """Whether anybody is still waiting for this hall call."""
        return self.waiting_count(floor, direction) > 0

    def oldest_arrival(self, floor: int, direction: Direction) -> int:
        """The arrival tick of the longest-waiting person for this call."""
        waits = [p.record.arrival_tick for p in self.waiting_at(floor) if p.direction is direction]
        return min(waits) if waits else self.tick

    # ------------------------------------------------------------ event callbacks

    def on_board(self, passenger: PassengerAgent, car_id: int) -> None:
        """A passenger boarded (the car itself tells the dispatcher, by INFORM)."""
        del passenger, car_id

    def on_alight(self, passenger: PassengerAgent) -> None:
        """A passenger was delivered."""
        del passenger

    def on_requeue(self, passenger: PassengerAgent) -> None:
        """A passenger was turned out of a failed car and is waiting again."""
        del passenger

    # ------------------------------------------------------------------ arrivals

    def spawn_passenger(
        self, origin: int, destination: int, weight: float = 1.0, priority: bool = False
    ) -> PassengerAgent:
        """Create one passenger and register them with the metrics collector."""
        passenger = PassengerAgent(self, origin, destination, weight, priority)
        self.passengers.append(passenger)
        self.collector.register(passenger.record)
        self.arrivals_this_tick.append(passenger)
        return passenger

    def _generate_arrivals(self) -> None:
        """Draw this tick's Poisson arrivals."""
        self.arrivals_this_tick = []
        for origin, destination, weight, priority in self.generator.arrivals_for_tick(self.tick):
            self.spawn_passenger(origin, destination, weight, priority)

    # -------------------------------------------------------------------- events

    def _apply_events(self) -> None:
        """Fire any scripted scenario events due at this tick."""
        while self._pending_events and self._pending_events[0].tick <= self.tick:
            event = self._pending_events.pop(0)
            self.inject(event.kind, car=event.car, floor=event.floor, count=event.count)

    def inject(
        self,
        kind: str,
        car: int | None = None,
        floor: int | None = None,
        count: int = 10,
    ) -> dict[str, Any]:
        """Inject a disturbance, from a scenario script or from the dashboard."""
        detail: dict[str, Any] = {"tick": self.tick, "kind": kind}
        if kind == "car_fault":
            car_id = car if car is not None else self.random.randrange(len(self.cars))
            if self.car(car_id) is not None:
                # The fault line: a hardware signal into the safety supervisor, whose
                # rule R4 then orders the car out of service.
                self.safety.report_fault(car_id)
            detail["car"] = car_id
        elif kind == "car_repair":
            car_id = car if car is not None else 0
            self.safety.clear_fault(car_id)
            detail["car"] = car_id
        elif kind == "fire_alarm":
            self.safety.trigger_fire_alarm()
        elif kind == "fire_clear":
            self.safety.clear_fire_alarm()
        elif kind == "rush":
            target_floor = floor if floor is not None else self.lobby
            for origin, destination, weight, priority in self.generator.rush(
                target_floor, count, self.tick
            ):
                self.spawn_passenger(origin, destination, weight, priority)
            detail["floor"] = target_floor
            detail["count"] = count
        else:
            raise ValueError(f"unknown event kind {kind!r}")

        self.events_fired.append(detail)
        if len(self.events_fired) > 50:
            del self.events_fired[:-50]
        return detail

    # ----------------------------------------------------------------- tick loop

    def step(self) -> None:
        """Advance the simulation by one tick through the four fixed stages.

        The SafetyAgent runs its whole cycle *before* the dispatcher's, because safety
        overrides coordination: if an alarm is raised this tick, hall calls are already
        blocked by the time any auction could run.
        """
        started = time.perf_counter()
        self.tick += 1
        self._apply_events()
        self._generate_arrivals()

        # Stage order is fixed; the safety agent leads so its conclusions bind everyone.
        for stage in STAGES:
            if stage == "negotiate":
                self._negotiate()
                continue
            for agent_type in AGENT_ORDER:
                self.agents_by_type[agent_type].do(stage)

        self._retire_delivered()

        self.collector.total_messages = self.bus.total_sent
        self.collector.ticks = self.tick
        self.compute_ms_last_tick = (time.perf_counter() - started) * 1000.0
        self.collector.compute_ms += self.compute_ms_last_tick
        self.latest_metrics = self.collector.snapshot(self.tick)
        self.datacollector.collect(self)

    def _negotiate(self) -> None:
        """Run one Contract Net round per queued call, entirely by message.

        The model only schedules the turns — announce, bid, award, commit — exactly as
        Mesa's staged activation schedules any other stage. Every piece of information
        that passes between the dispatcher and the cars travels as a message on the
        bus, read by its recipient on its own turn. Rounds run one call at a time, so
        each car bids knowing every award already made this tick.
        """
        dispatchers = self.agents_by_type[DispatcherAgent]
        cars = self.agents_by_type[ElevatorAgent]
        rounds = 0
        while True:
            dispatchers.do("announce")
            if not self.dispatcher.round_open:
                break
            cars.do("bid")
            dispatchers.do("award")
            cars.do("commit")
            rounds += 1
        self.negotiation_rounds = rounds

    def _retire_delivered(self) -> None:
        """Remove delivered passengers from the population.

        Their `PassengerRecord` stays with the metrics collector, so the statistics are
        complete — only the agent object goes, which keeps a long run's memory and
        per-tick scan costs flat.
        """
        if len(self.passengers) < 200:
            return
        remaining: list[PassengerAgent] = []
        for passenger in self.passengers:
            if passenger.record.delivered:
                passenger.remove()  # Mesa 3: agent.remove(), not a scheduler call
            else:
                remaining.append(passenger)
        self.passengers = remaining

    def run(self, ticks: int | None = None) -> Metrics:
        """Run for `ticks` (default: the scenario's duration) and return the metrics."""
        horizon = ticks if ticks is not None else self.config.duration
        for _ in range(horizon):
            self.step()
        return self.latest_metrics

    def drain(self, max_ticks: int = 3000) -> Metrics:
        """Stop new arrivals and run until everyone already in the system is delivered.

        The tests use this to assert that no passenger is ever starved: after the drain
        every single arrival must have been delivered.
        """
        for phase in self.config.traffic.phases:
            phase.rate = 0.0
        for _ in range(max_ticks):
            if all(r.delivered for r in self.collector.records.values()):
                break
            self.step()
        return self.latest_metrics

    # ------------------------------------------------------------------ snapshots

    def snapshot(self) -> dict[str, Any]:
        """The full world state the dashboard renders each frame."""
        return {
            "tick": self.tick,
            "scenario": self.config.name,
            "strategy": self.strategy.name,
            "seed": self.seed_value,
            "building": {
                "floors": self.floors_count,
                "cars": len(self.cars),
                "capacity": self.config.building.capacity,
                "lobby": self.lobby,
            },
            "cars": [car.snapshot() for car in self.cars],
            "floors": [floor.snapshot() for floor in self.floors],
            "metrics": self.latest_metrics.as_dict(),
            "auction": self.dispatcher.last_auction.as_dict()
            if self.dispatcher.last_auction
            else None,
            "messages": [m.as_dict() for m in self.bus.recent(40)],
            "rules": self.safety.recent_rules(12),
            "traffic": {
                "pattern": self.monitor.pattern,
                "reason": self.monitor.classification_reason,
                "observed": self.monitor.observed,
                "true_pattern": self.config.traffic.pattern_at(self.tick),
                "weights": self.weights.model_dump(),
            },
            "events": self.events_fired[-5:],
            "fire_alarm": self.safety.fire_alarm,
            "reassignment": (
                self.dispatcher.last_reassignment.as_dict()
                if self.dispatcher.last_reassignment
                else None
            ),
            "parking": self.dispatcher.last_parking,
        }

    def violations(self) -> list[str]:
        """Check every safety invariant now; the tests assert this stays empty.

        These are the properties that must hold at *every* tick of *every* run, so they
        are written once here and checked both by the tests and (cheaply) by the
        dashboard.
        """
        problems: list[str] = []
        for car in self.cars:
            if car.load > car.capacity:
                problems.append(f"car {car.car_id} over capacity: {car.load}>{car.capacity}")
            if car.moving and car.door_state.value != "closed":
                problems.append(f"car {car.car_id} moving with doors {car.door_state.value}")
            if not 0 <= car.floor < self.floors_count:
                problems.append(f"car {car.car_id} out of bounds at floor {car.floor}")
            if car.state is CarState.OUT_OF_SERVICE and car.riders:
                problems.append(f"car {car.car_id} out of service still carrying riders")
        return problems
