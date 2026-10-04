"""Fast twin simulator state, physical step loop, and semi-MDP execution.

Implements the physical elevator world with LOOK sweep stop sequencing,
exact door state cycles, passenger lifecycle, and decision stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from elevator_mas.config import ScenarioConfig
from elevator_mas.learning.schema import MAX_CARS
from elevator_mas.learning.twin.kernels import look_route, route_eta
from elevator_mas.learning.twin.traffic import TwinTrafficGenerator
from elevator_mas.learning.view import DecisionContext, FleetView


@dataclass
class TwinPassenger:
    """Passenger record in the twin simulator."""

    origin: int
    destination: int
    direction: int  # +1 UP, -1 DOWN
    weight: float
    is_priority: bool
    arrived_tick: int
    boarded_tick: int = -1
    alighted_tick: int = -1


@dataclass
class TwinCar:
    """State of a single elevator car in the twin simulator."""

    car_id: int
    floor: int = 0
    direction: int = 0  # +1 UP, -1 DOWN, 0 IDLE
    door_state: int = 0  # 0 CLOSED, 1 OPENING, 2 OPEN, 3 CLOSING
    door_timer: int = 0
    dwell_remaining: int = 0
    move_timer: int = 0
    target_floor: int | None = None
    available: bool = True
    out_of_service: bool = False
    fire_mode: bool = False
    door_blocked_ticks: int = 0

    car_calls: set[int] = field(default_factory=set)
    assigned_calls: set[tuple[int, int]] = field(default_factory=set)
    riders: list[TwinPassenger] = field(default_factory=list)

    floors_travelled: int = 0
    stops_made: int = 0
    reversals: int = 0
    park_target: int | None = None


class TwinSimulator:
    """Fast twin simulator of the elevator building and assignment semi-MDP."""

    def __init__(
        self,
        floors: int = 15,
        cars: int = 4,
        capacity: int = 10,
        lobby: int = 0,
        duration: int = 900,
        seconds_per_floor: int = 2,
        door_open: int = 2,
        door_close: int = 2,
        boarding_per_passenger: int = 1,
        dwell_min: int = 1,
        seed: int = 42,
        rate: float = 0.20,
        pattern: str = "up_peak",
        priority_prob: float = 0.0,
        config: ScenarioConfig | None = None,
    ) -> None:
        if config is not None:
            self.floors = config.building.floors
            self.n_cars = config.building.cars
            self.capacity = config.building.capacity
            self.lobby = config.building.lobby
            self.duration = config.duration
            self.seconds_per_floor = config.timing.seconds_per_floor
            self.door_open = config.timing.door_open
            self.door_close = config.timing.door_close
            self.boarding_per_passenger = config.timing.boarding_per_passenger
            self.dwell_min = config.timing.dwell_min
            self.seed = seed if seed is not None else config.seed
            self.scenario_config = config
        else:
            self.floors = floors
            self.n_cars = cars
            self.capacity = capacity
            self.lobby = lobby
            self.duration = duration
            self.seconds_per_floor = seconds_per_floor
            self.door_open = door_open
            self.door_close = door_close
            self.boarding_per_passenger = boarding_per_passenger
            self.dwell_min = dwell_min
            self.seed = seed
            self.scenario_config = None

        self.dwell_default = float(self.door_open + self.door_close)
        self.default_rate = rate
        self.default_pattern = pattern
        self.priority_prob = priority_prob

        self.rng = np.random.default_rng(self.seed)
        self.traffic_gen = TwinTrafficGenerator(
            floors=self.floors,
            lobby=self.lobby,
            priority_prob=self.priority_prob,
            rng=self.rng,
        )

        self.cars: list[TwinCar] = []
        self.waiting: dict[tuple[int, int], list[TwinPassenger]] = {}
        self.unassigned_calls: list[tuple[int, int]] = []
        self.call_first_tick: dict[tuple[int, int], int] = {}
        self.delivered_passengers: list[TwinPassenger] = []
        self.all_passengers: list[TwinPassenger] = []

        self.ewma_demand: dict[int, float] = {}
        self.tick: int = 0
        self.hall_calls_blocked: bool = False
        self.decisions_count: int = 0

        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the simulator to tick 0."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.traffic_gen.rng = self.rng

        self.tick = 0
        self.decisions_count = 0
        self.hall_calls_blocked = False

        self.cars = [TwinCar(car_id=i, floor=self.lobby) for i in range(self.n_cars)]
        self.waiting = {
            (f, d): [] for f in range(self.floors) for d in (1, -1)
        }
        self.unassigned_calls = []
        self.call_first_tick = {}
        self.delivered_passengers = []
        self.all_passengers = []
        self.ewma_demand = {f: 0.0 for f in range(self.floors)}

    def current_rate(self) -> float:
        if self.scenario_config is not None:
            return self.scenario_config.traffic.rate_at(self.tick)
        return self.default_rate

    def current_pattern(self) -> str:
        if self.scenario_config is not None:
            return str(self.scenario_config.traffic.pattern_at(self.tick))
        return self.default_pattern

    # ---------------------------------------------------------------- Semi-MDP stepping

    def advance(self) -> tuple[DecisionContext | None, bool]:
        """Advance time until a hall call requires an assignment, or duration reached.

        Returns (context, terminated). If terminated, context is None.
        """
        while not self.unassigned_calls and self.tick < self.duration:
            self._step_tick()

        if self.unassigned_calls:
            call = self.unassigned_calls[0]
            ctx = self._build_context(call)
            return ctx, False

        return None, True

    def step_action(self, car_idx: int) -> tuple[DecisionContext | None, bool]:
        """Assign the currently pending call to car_idx, then advance to next decision."""
        if self.unassigned_calls:
            call = self.unassigned_calls.pop(0)
            target_car = max(0, min(car_idx, self.n_cars - 1))
            self.cars[target_car].assigned_calls.add(call)
            self.decisions_count += 1

        return self.advance()

    # ---------------------------------------------------------------- Internal tick step

    def _step_tick(self) -> None:
        self.tick += 1

        # 1. Apply events if from ScenarioConfig
        if self.scenario_config is not None:
            for ev in self.scenario_config.events:
                if ev.tick == self.tick:
                    self._apply_scenario_event(ev)

        # 2. Arrivals
        rate = self.current_rate()
        pat = self.current_pattern()
        arrivals = self.traffic_gen.generate_arrivals(rate, pat)

        # EWMA demand update (alpha=0.02)
        alpha = 0.02
        floor_counts = {f: 0 for f in range(self.floors)}
        for orig, dest, weight, prio in arrivals:
            floor_counts[orig] += 1
            direction = 1 if dest > orig else -1
            p = TwinPassenger(
                origin=orig,
                destination=dest,
                direction=direction,
                weight=weight,
                is_priority=prio,
                arrived_tick=self.tick,
            )
            self.waiting[(orig, direction)].append(p)
            self.all_passengers.append(p)

            # Check if hall call should be pressed
            if not self.hall_calls_blocked:
                call = (orig, direction)
                if call not in self.call_first_tick:
                    self.call_first_tick[call] = self.tick
                # If no car has this call assigned and not already in unassigned list
                is_assigned = any(call in c.assigned_calls for c in self.cars)
                if not is_assigned and call not in self.unassigned_calls:
                    self.unassigned_calls.append(call)

        for f in range(self.floors):
            self.ewma_demand[f] = (1.0 - alpha) * self.ewma_demand[f] + alpha * floor_counts[f]

        # 3. Advance physics for each car
        for car in self.cars:
            self._advance_car(car)

    def _apply_scenario_event(self, ev: Any) -> None:
        t = getattr(ev, "type", "")
        car_id = getattr(ev, "car_id", 0)
        if t == "car_fault":
            self.inject_fault(car_id)
        elif t == "car_repair":
            self.repair_car(car_id)
        elif t == "fire_alarm":
            self.inject_fire_alarm()
        elif t == "fire_clear":
            self.clear_fire_alarm()
        elif t == "rush":
            floor = getattr(ev, "floor", 0)
            count = getattr(ev, "count", 10)
            for _ in range(count):
                choices = [f for f in range(self.floors) if f != floor]
                dest = int(self.rng.choice(choices))
                d = 1 if dest > floor else -1
                p = TwinPassenger(
                    origin=floor,
                    destination=dest,
                    direction=d,
                    weight=1.0,
                    is_priority=False,
                    arrived_tick=self.tick,
                )
                self.waiting[(floor, d)].append(p)
                self.all_passengers.append(p)
                call = (floor, d)
                if not self.hall_calls_blocked:
                    if call not in self.call_first_tick:
                        self.call_first_tick[call] = self.tick
                    is_assigned = any(call in c.assigned_calls for c in self.cars)
                    if not is_assigned and call not in self.unassigned_calls:
                        self.unassigned_calls.append(call)

    def inject_fault(self, car_id: int) -> None:
        if 0 <= car_id < self.n_cars:
            car = self.cars[car_id]
            car.out_of_service = True
            car.available = False
            # Eject riders to waiting queue at car's floor
            for r in car.riders:
                d = 1 if r.destination > car.floor else -1
                self.waiting[(car.floor, d)].append(r)
                call = (car.floor, d)
                if not self.hall_calls_blocked and call not in self.unassigned_calls:
                    self.unassigned_calls.append(call)
            car.riders.clear()
            car.car_calls.clear()
            # Release assigned calls
            for c in car.assigned_calls:
                if not self.hall_calls_blocked and c not in self.unassigned_calls:
                    self.unassigned_calls.append(c)
            car.assigned_calls.clear()

    def repair_car(self, car_id: int) -> None:
        if 0 <= car_id < self.n_cars:
            car = self.cars[car_id]
            car.out_of_service = False
            car.available = True
            car.door_state = 0

    def inject_fire_alarm(self) -> None:
        self.hall_calls_blocked = True
        self.unassigned_calls.clear()
        for car in self.cars:
            car.fire_mode = True
            car.assigned_calls.clear()
            car.car_calls.clear()

    def clear_fire_alarm(self) -> None:
        self.hall_calls_blocked = False
        for car in self.cars:
            car.fire_mode = False
        # Re-announce all waiting calls
        for (f, d), q in self.waiting.items():
            if q and (f, d) not in self.unassigned_calls:
                self.unassigned_calls.append((f, d))

    def _advance_car(self, car: TwinCar) -> None:
        if car.out_of_service:
            car.direction = 0
            return

        if car.fire_mode:
            # Fire recall: recall to lobby
            if car.floor != self.lobby:
                if car.door_state == 0:
                    car.direction = 1 if self.lobby > car.floor else -1
                    if car.move_timer <= 0:
                        car.move_timer = self.seconds_per_floor
                    car.move_timer -= 1
                    if car.move_timer <= 0:
                        car.floor += car.direction
                        car.floors_travelled += 1
            else:
                car.door_state = 2  # stay open at lobby
                car.direction = 0
            return

        # Normal operation
        if car.door_state != 0:
            self._advance_car_doors(car)
            return

        # Car doors are closed. Check if we should serve current floor.
        if self._should_serve_floor(car, car.floor):
            self._begin_door_cycle(car)
            return

        # Choose target by LOOK sweep
        all_stops = list(car.car_calls | {call[0] for call in car.assigned_calls})
        if car.park_target is not None and not all_stops:
            all_stops.append(car.park_target)

        if not all_stops:
            route = np.empty(0, dtype=np.int32)
        else:
            route = look_route(car.floor, car.direction, np.array(all_stops, dtype=np.int32))
        if len(route) > 0:
            target = int(route[0])
            if target != car.floor:
                car.target_floor = target
                new_dir = 1 if target > car.floor else -1
                if car.direction != 0 and new_dir != car.direction:
                    car.reversals += 1
                car.direction = new_dir

                if car.move_timer <= 0:
                    car.move_timer = self.seconds_per_floor
                car.move_timer -= 1
                if car.move_timer <= 0:
                    car.floor += car.direction
                    car.floors_travelled += 1
                    if car.floor == target:
                        car.target_floor = None
            else:
                self._begin_door_cycle(car)
        else:
            car.direction = 0
            car.target_floor = None

    def _should_serve_floor(self, car: TwinCar, floor: int) -> bool:
        if floor in car.car_calls:
            return True
        for (c_fl, c_dir) in car.assigned_calls:
            if c_fl == floor:
                return True
        return False

    def _begin_door_cycle(self, car: TwinCar) -> None:
        car.door_state = 1  # OPENING
        car.door_timer = self.door_open
        car.stops_made += 1
        car.target_floor = None
        car.move_timer = 0

    def _advance_car_doors(self, car: TwinCar) -> None:
        car.door_timer -= 1
        if car.door_timer > 0:
            return

        if car.door_state == 1:
            # OPENING -> OPEN
            car.door_state = 2
            # 1. Alight
            leaving = [r for r in car.riders if r.destination == car.floor]
            for r in leaving:
                car.riders.remove(r)
                r.alighted_tick = self.tick
                self.delivered_passengers.append(r)
            car.car_calls.discard(car.floor)

            # 2. Board
            boarded = 0
            serving_dirs = self._get_serving_directions(car)
            for d in serving_dirs:
                q = self.waiting.get((car.floor, d), [])
                to_board: list[TwinPassenger] = []
                for p in q:
                    if len(car.riders) + boarded >= self.capacity:
                        break
                    to_board.append(p)
                    boarded += 1

                for p in to_board:
                    q.remove(p)
                    p.boarded_tick = self.tick
                    car.riders.append(p)
                    car.car_calls.add(p.destination)

                # Clear assigned call for this direction
                car.assigned_calls.discard((car.floor, d))
                if not q:
                    self.call_first_tick.pop((car.floor, d), None)
                elif (car.floor, d) not in self.unassigned_calls:
                    # People left behind: re-request call!
                    self.unassigned_calls.append((car.floor, d))

            dwell = max(1, self.dwell_min, boarded * self.boarding_per_passenger)
            car.dwell_remaining = dwell
            car.door_timer = dwell
            return

        if car.door_state == 2:
            # OPEN -> CLOSING
            car.door_state = 3
            car.door_timer = max(1, self.door_close)
            return

        if car.door_state == 3:
            # CLOSING -> CLOSED
            car.door_state = 0
            car.dwell_remaining = 0
            car.target_floor = None
            car.move_timer = 0

    def _get_serving_directions(self, car: TwinCar) -> list[int]:
        assigned_dirs = [d for (f, d) in car.assigned_calls if f == car.floor]
        if assigned_dirs:
            car.direction = assigned_dirs[0]
            return assigned_dirs
        if not car.riders:
            return [1, -1]
        if car.direction != 0:
            return [car.direction]
        all_stops = list(car.car_calls | {c[0] for c in car.assigned_calls})
        if any(s > car.floor for s in all_stops):
            car.direction = 1
            return [1]
        if any(s < car.floor for s in all_stops):
            car.direction = -1
            return [-1]
        return [1, -1]

    # ---------------------------------------------------------------- Feature Context Builder

    def _build_context(self, call: tuple[int, int]) -> DecisionContext:
        call_floor, call_dir = call
        waiting_q = self.waiting.get((call_floor, call_dir), [])
        waiting_count = len(waiting_q)
        first_tick = self.call_first_tick.get(call, self.tick)
        urgency = min(4, max(0, (self.tick - first_tick) // 60))

        c_floors = np.zeros(self.n_cars, dtype=np.int16)
        c_dirs = np.zeros(self.n_cars, dtype=np.int8)
        c_loads = np.zeros(self.n_cars, dtype=np.int16)
        c_caps = np.full(self.n_cars, self.capacity, dtype=np.int16)
        c_avails = np.zeros(self.n_cars, dtype=bool)
        c_oos = np.zeros(self.n_cars, dtype=bool)
        c_fire = np.zeros(self.n_cars, dtype=bool)
        c_doors = np.zeros(self.n_cars, dtype=np.int8)
        c_blocked = np.zeros(self.n_cars, dtype=np.int16)
        c_n_assigned = np.zeros(self.n_cars, dtype=np.int16)
        c_n_car_calls = np.zeros(self.n_cars, dtype=np.int16)
        c_riders = np.zeros(self.n_cars, dtype=np.int16)
        c_planned_stops = np.zeros(self.n_cars, dtype=np.int16)
        c_plan_end_floors = np.zeros(self.n_cars, dtype=np.int16)
        c_plan_end_etas = np.zeros(self.n_cars, dtype=np.float32)
        c_has_same = np.zeros(self.n_cars, dtype=bool)
        c_park_targets = np.full(self.n_cars, -1, dtype=np.int16)

        for i, car in enumerate(self.cars):
            c_floors[i] = car.floor
            c_dirs[i] = car.direction
            c_loads[i] = len(car.riders)
            c_avails[i] = car.available and not car.out_of_service and not car.fire_mode
            c_oos[i] = car.out_of_service
            c_fire[i] = car.fire_mode
            c_doors[i] = car.door_state
            c_blocked[i] = car.door_blocked_ticks
            c_n_assigned[i] = len(car.assigned_calls)
            c_n_car_calls[i] = len(car.car_calls)
            c_riders[i] = len(car.riders)

            # Planned stops & LOOK route ETA
            all_stops = list(car.car_calls | {c[0] for c in car.assigned_calls})
            if not all_stops:
                c_planned_stops[i] = 0
                c_plan_end_floors[i] = car.floor
                c_plan_end_etas[i] = 0.0
            else:
                route = look_route(car.floor, car.direction, np.array(all_stops, dtype=np.int32))
                c_planned_stops[i] = len(route)
                end_fl, eta = route_eta(
                    car.floor,
                    route,
                    self.seconds_per_floor,
                    self.dwell_default,
                    car.move_timer,
                    car.dwell_remaining,
                )
                c_plan_end_floors[i] = end_fl
                c_plan_end_etas[i] = eta
            c_has_same[i] = call in car.assigned_calls
            c_park_targets[i] = car.park_target if car.park_target is not None else -1

        fleet = FleetView(
            n_cars=self.n_cars,
            floors=c_floors,
            directions=c_dirs,
            loads=c_loads,
            capacities=c_caps,
            availables=c_avails,
            out_of_service=c_oos,
            fire_mode=c_fire,
            doors=c_doors,
            door_blocked_ticks=c_blocked,
            n_assigned=c_n_assigned,
            n_car_calls=c_n_car_calls,
            riders=c_riders,
            planned_stops=c_planned_stops,
            plan_end_floors=c_plan_end_floors,
            plan_end_etas=c_plan_end_etas,
            has_same_call=c_has_same,
            park_targets=c_park_targets,
        )

        return DecisionContext(
            call_floor=call_floor,
            call_direction=call_dir,
            call_waiting=waiting_count,
            call_urgency=urgency,
            floors=self.floors,
            lobby=self.lobby,
            capacity=self.capacity,
            pattern=self.current_pattern(),
            demand=dict(self.ewma_demand),
            tick=self.tick,
            open_calls_count=len(self.unassigned_calls),
            fleet=fleet,
        )

    # ---------------------------------------------------------------- Metrics

    def get_metrics(self) -> dict[str, float]:
        """Compute summary metrics matching MetricsCollector."""
        waits: list[float] = []
        rides: list[float] = []

        for p in self.all_passengers:
            if p.boarded_tick >= 0:
                waits.append(float(p.boarded_tick - p.arrived_tick))
            elif self.tick > p.arrived_tick:
                # Still waiting at end of run
                waits.append(float(self.tick - p.arrived_tick))

            if p.alighted_tick >= 0:
                rides.append(float(p.alighted_tick - p.boarded_tick))

        n_passengers = len(waits)
        avg_wait = float(np.mean(waits)) if waits else 0.0
        p95_wait = float(np.percentile(waits, 95)) if waits else 0.0
        max_wait = float(np.max(waits)) if waits else 0.0
        avg_ride = float(np.mean(rides)) if rides else 0.0

        long_waits = sum(1 for w in waits if w > 60.0)
        long_wait_pct = (long_waits / n_passengers * 100.0) if n_passengers else 0.0

        tot_floors = sum(c.floors_travelled for c in self.cars)
        tot_stops = sum(c.stops_made for c in self.cars)
        energy = 0.5 * tot_floors + 1.0 * tot_stops

        throughput = float(len(self.delivered_passengers)) / max(float(self.tick), 1.0)

        return {
            "arrived": len(self.all_passengers),
            "delivered": len(self.delivered_passengers),
            "waiting": sum(len(q) for q in self.waiting.values()),
            "riding": sum(len(c.riders) for c in self.cars),
            "avg_wait": avg_wait,
            "p95_wait": p95_wait,
            "max_wait": max_wait,
            "avg_ride": avg_ride,
            "long_wait_pct": long_wait_pct,
            "throughput": throughput,
            "energy": energy,
            "floors_travelled": tot_floors,
            "stops": tot_stops,
            "decisions": self.decisions_count,
        }
