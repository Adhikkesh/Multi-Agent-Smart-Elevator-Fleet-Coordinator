"""The simulation session that the dashboard drives.

Holds one `ElevatorModel` plus the play/pause/speed state, and owns the background task
that steps it. The engine itself knows nothing about this: the session only calls
`model.step()` and reads `model.snapshot()`, which is what keeps the headless engine and
the UI genuinely separate.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from elevator_mas.comms.message import jsonable
from elevator_mas.config import ScenarioConfig, available_scenarios
from elevator_mas.model import ElevatorModel
from elevator_mas.planning import ALGORITHMS
from elevator_mas.planning.routing import CarRoutingProblem, RoutingCosts
from elevator_mas.strategies import STRATEGIES


def _stop_label(stop: Any) -> str:
    """Compact label for a stop in the Search Lab's route column, e.g. "7^P"."""
    arrow = {"UP": "^", "DOWN": "v"}.get(stop.direction.name, "")
    return f"{stop.floor}{arrow}{'P' if stop.is_pickup else 'D'}"


#: The wall-clock seconds one tick takes at 1x speed.
BASE_TICK_SECONDS = 1.0


class SimulationSession:
    """One live simulation, shared by every connected dashboard client."""

    def __init__(self, scenario: str = "demo_story") -> None:
        self.scenario_name = scenario
        self.model = ElevatorModel(ScenarioConfig.load(scenario))
        self.running = False
        self.speed = 10.0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    # ------------------------------------------------------------------ lifecycle

    def reset(
        self,
        scenario: str | None = None,
        strategy: str | None = None,
        seed: int | None = None,
        floors: int | None = None,
        cars: int | None = None,
    ) -> None:
        """Rebuild the model, optionally with different settings.

        Everything the dashboard can change goes through the config, so no UI control ever
        needs a code change to work.
        """
        name = scenario or self.scenario_name
        if name not in available_scenarios():
            raise ValueError(f"unknown scenario {name!r}")
        config = ScenarioConfig.load(name)
        updates: dict[str, Any] = {}
        if strategy is not None:
            if strategy not in STRATEGIES:
                raise ValueError(f"unknown strategy {strategy!r}")
            updates["strategy"] = strategy
        if seed is not None:
            updates["seed"] = seed
        if floors is not None or cars is not None:
            building = config.building.model_copy(
                update={k: v for k, v in (("floors", floors), ("cars", cars)) if v is not None}
            )
            updates["building"] = building
        if updates:
            config = config.model_copy(update=updates)
        self.scenario_name = name
        self.model = ElevatorModel(config)

    async def start(self) -> None:
        """Begin (or resume) automatic stepping."""
        self.running = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Pause automatic stepping; the model state is kept."""
        self.running = False

    async def shutdown(self) -> None:
        """Cancel the background task on server shutdown."""
        self.running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        """Step the model while running, pacing to the requested speed."""
        while True:
            if not self.running:
                await asyncio.sleep(0.05)
                continue
            if self.model.tick >= self.model.config.duration:
                self.running = False
                await self.broadcast()
                continue
            async with self._lock:
                self.model.step()
            await self.broadcast()
            await asyncio.sleep(max(0.01, BASE_TICK_SECONDS / max(0.1, self.speed)))

    async def step_once(self, count: int = 1) -> None:
        """Advance exactly `count` ticks while paused."""
        async with self._lock:
            for _ in range(max(1, count)):
                self.model.step()
        await self.broadcast()

    # ---------------------------------------------------------------- subscribers

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a WebSocket client and get its frame queue."""
        # Bounded: a slow client drops frames rather than growing the queue without limit.
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a disconnected client."""
        self._subscribers.discard(queue)

    async def broadcast(self) -> None:
        """Push the current snapshot to every subscriber."""
        if not self._subscribers:
            return
        frame = self.snapshot()
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:
                # The client is behind; skip this frame for it and keep the sim smooth.
                continue

    # ------------------------------------------------------------------- snapshot

    def snapshot(self) -> dict[str, Any]:
        """The world state plus the session's own controls."""
        payload: dict[str, Any] = jsonable(self.model.snapshot())
        payload["session"] = {
            "running": self.running,
            "speed": self.speed,
            "duration": self.model.config.duration,
        }
        return payload

    # ------------------------------------------------------------- search lab

    def snapshot_routing_problem(self, car_id: int | None = None) -> dict[str, Any]:
        """Take a live car's routing problem and run every algorithm on it.

        This is the Search Lab: the *same* problem instance goes to BFS, UCS, Greedy and
        A*, so the comparison is exact rather than illustrative. The problem is a real
        one lifted out of the running simulation, which is what makes the lab convincing.
        """
        car = self.model.car(car_id) if car_id is not None else None
        if car is None:
            # Prefer a car with an interesting plan over an idle one.
            candidates = sorted(self.model.cars, key=lambda c: -len(c.pending_stops()))
            car = candidates[0] if candidates else None
        if car is None:
            return {"error": "no cars"}

        stops = car.pending_stops()
        if not stops:
            stops = self._random_stops()
            source = "randomly generated (no car had pending stops)"
        else:
            source = f"live snapshot of car {car.car_id}"

        problem = CarRoutingProblem(
            car.floor,
            stops,
            RoutingCosts(
                seconds_per_floor=float(self.model.config.timing.seconds_per_floor),
                dwell=float(
                    self.model.config.timing.door_open + self.model.config.timing.door_close
                ),
                energy_per_floor=self.model.config.planner.energy_per_floor,
                energy_per_stop=self.model.config.planner.energy_per_stop,
            ),
            boarding_per_passenger=float(self.model.config.timing.boarding_per_passenger),
        )

        results = []
        for name in ("bfs", "ucs", "greedy", "astar"):
            result = ALGORITHMS[name](problem)
            payload = result.as_dict()
            payload["route"] = [_stop_label(s) for s in result.actions]
            results.append(payload)

        optimal = min((r["cost"] for r in results if r["found"]), default=0.0)
        for payload in results:
            payload["optimal"] = bool(payload["found"] and abs(payload["cost"] - optimal) < 1e-6)

        return {
            "source": source,
            "car_id": car.car_id,
            "current_floor": car.floor,
            "stops": [
                {
                    "floor": s.floor,
                    "kind": s.kind.value,
                    "direction": s.direction.name,
                    "weight": s.weight,
                }
                for s in problem.pending
            ],
            "h_at_start": round(problem.h(problem.initial_state()), 3),
            "results": results,
        }

    def _random_stops(self) -> list[Any]:
        """A random but legal set of stops, for when no car is busy."""
        from elevator_mas.domain import Direction, Stop, StopKind

        floors = self.model.floors_count
        rng = self.model.random
        stops: list[Stop] = []
        for _ in range(rng.randint(3, 5)):
            floor = rng.randrange(floors)
            if rng.random() < 0.5:
                stops.append(
                    Stop(
                        floor,
                        StopKind.PICKUP,
                        Direction.UP if rng.random() < 0.5 else Direction.DOWN,
                        float(rng.randint(1, 3)),
                    )
                )
            else:
                stops.append(
                    Stop(floor, StopKind.DROPOFF, Direction.IDLE, float(rng.randint(1, 3)))
                )
        return stops
