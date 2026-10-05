"""The search world model: a twin built from the dispatcher's *public* view of the building.

The dispatcher knows every car's published status (floor, direction, doors, load, the
destination buttons pressed inside it, its assigned hall calls), the fleet policy (traffic
pattern, demand estimate), the static building and timing, and the hall calls with the
waiting counts the floors report. It does **not** know where waiting people want to go, how
riders are spread over the pressed buttons, or who will arrive next. Those are *sampled* per
simulation — determinisation, i.e. perfect-information Monte Carlo over a belief state, the
standard answer to a partially observable environment.

Common random numbers: simulation ``s`` fixes one sampled hidden world and one future arrival
stream (``seed = (root_seed, s)``); every candidate car evaluated in simulation ``s`` faces the
same world, so differences between candidates are not drowned in sampling noise.

The twin's per-tick cost is the Phase 5 team reward: waiting people + ½ riding people per
second (/100), 0.02 per floor travelled, 0.5 per new 60-second wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from elevator_mas.learning.twin.state import TwinPassenger, TwinSimulator
from elevator_mas.learning.twin.traffic import TWIN_PROFILES

_DOOR = {"closed": 0, "opening": 1, "open": 2, "closing": 3}


@dataclass(frozen=True)
class PublicView:
    """Everything the dispatcher may legitimately use for look-ahead."""

    statuses: tuple[Any, ...]  # CarStatus of every car (announce-time snapshot)
    pattern: str
    demand_rate: float  # estimated arrivals per second for the whole building
    floors: int
    lobby: int
    capacity: int
    seconds_per_floor: int
    door_open: int
    door_close: int
    boarding_per_passenger: int
    tick: int
    #: (floor, +1/-1) -> (waiting count reported by the floor, first tick of the call)
    calls: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)


class SearchTwin(TwinSimulator):
    """A twin that accumulates the team cost tick by tick (exact SMDP rewards)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cost = 0.0
        self._crossed: set[int] = set()

    def _step_tick(self) -> None:
        floors_before = sum(c.floors_travelled for c in self.cars)
        super()._step_tick()
        waiting = riding = crossings = 0
        for q in self.waiting.values():
            for p in q:
                waiting += 1
                if self.tick - p.arrived_tick >= 60 and id(p) not in self._crossed:
                    self._crossed.add(id(p))
                    crossings += 1
        for c in self.cars:
            riding += len(c.riders)
        moved = sum(c.floors_travelled for c in self.cars) - floors_before
        self.cost += (waiting + 0.5 * riding) / 100.0 + 0.02 * moved + 0.5 * crossings


def _sample_destination(
    rng: np.random.Generator, origin: int, direction: int, view: PublicView
) -> int:
    prof = TWIN_PROFILES.get(view.pattern, TWIN_PROFILES["interfloor"])
    if direction > 0:
        choices = list(range(origin + 1, view.floors))
    else:
        if view.lobby < origin and rng.random() < prof.lobby_destination:
            return view.lobby
        choices = list(range(0, origin))
    if not choices:
        return view.lobby if origin != view.lobby else min(origin + 1, view.floors - 1)
    return int(rng.choice(choices))


def build_world(
    view: PublicView, root_call: tuple[int, int], seed: Any, horizon: int
) -> SearchTwin:
    """A sampled, fully specified twin consistent with ``view`` (``root_call`` decides first)."""
    sim = SearchTwin(
        floors=view.floors,
        cars=len(view.statuses),
        capacity=view.capacity,
        lobby=view.lobby,
        duration=view.tick + horizon,
        seconds_per_floor=view.seconds_per_floor,
        door_open=view.door_open,
        door_close=view.door_close,
        boarding_per_passenger=view.boarding_per_passenger,
        seed=0,
        rate=max(view.demand_rate, 0.01),
        pattern=view.pattern,
    )
    rng = np.random.default_rng(seed)
    sim.rng = rng
    sim.traffic_gen.rng = rng
    sim.tick = view.tick
    assigned: set[tuple[int, int]] = set()
    for s, car in zip(view.statuses, sim.cars, strict=True):
        car.floor = int(s.floor)
        d = getattr(s.direction, "sign", 0)
        car.direction = int(d)
        car.door_state = _DOOR.get(str(s.door).lower(), 0)
        car.door_timer = 1 if car.door_state else 0
        car.available = bool(s.available)
        car.out_of_service = bool(s.out_of_service)
        car.fire_mode = bool(s.fire_mode)
        car.car_calls = {int(f) for f in s.car_calls}
        car.assigned_calls = {(int(c.floor), int(c.direction.sign)) for c in s.assigned_calls}
        assigned |= car.assigned_calls
        car.park_target = s.park_target
        # Riders: at least one per pressed button, the rest spread uniformly over them.
        buttons = sorted(car.car_calls)
        n = int(s.load)
        dests = buttons[:n]
        while len(dests) < n:
            dests.append(int(rng.choice(buttons)) if buttons else view.lobby)
        car.riders = [
            TwinPassenger(
                origin=car.floor,
                destination=dst,
                direction=1 if dst > car.floor else -1,
                weight=1.0,
                is_priority=False,
                arrived_tick=view.tick,
                boarded_tick=view.tick,
            )
            for dst in dests
        ]
    sim.unassigned_calls = []
    for (f, d), (count, first) in sorted(view.calls.items()):
        q = sim.waiting.setdefault((f, d), [])
        for _ in range(max(count, 1)):
            dst = _sample_destination(rng, f, d, view)
            q.append(
                TwinPassenger(
                    origin=f,
                    destination=dst,
                    direction=d,
                    weight=1.0,
                    is_priority=False,
                    arrived_tick=first,
                )
            )
        sim.all_passengers.extend(q)
        sim.call_first_tick[(f, d)] = first
        if (f, d) not in assigned and (f, d) != root_call:
            sim.unassigned_calls.append((f, d))
    sim.unassigned_calls.insert(0, root_call)
    return sim
