"""The FastAPI application: REST control, WebSocket streaming, and the static dashboard."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from elevator_mas.agents import (
    DispatcherAgent,
    ElevatorAgent,
    FloorAgent,
    PassengerAgent,
    SafetyAgent,
    TrafficMonitorAgent,
)
from elevator_mas.api.schemas import (
    BenchmarkRequest,
    InjectRequest,
    PassengerRequest,
    ResetRequest,
    SpeedRequest,
    StepRequest,
)
from elevator_mas.api.session import SimulationSession
from elevator_mas.config import available_scenarios
from elevator_mas.domain import Direction, HallCall
from elevator_mas.optimization import minimax_parking
from elevator_mas.optimization.local_search import hill_climbing, simulated_annealing
from elevator_mas.rules import SAFETY_RULES
from elevator_mas.strategies import STRATEGIES

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: The agent classes whose PEAS the Theory tab documents, in presentation order.
AGENT_CLASSES = (
    PassengerAgent,
    FloorAgent,
    ElevatorAgent,
    DispatcherAgent,
    TrafficMonitorAgent,
    SafetyAgent,
)

#: The environment classification, with the justification shown beside each property.
ENVIRONMENT_PROPERTIES: list[dict[str, str]] = [
    {
        "property": "Partially observable",
        "value": "Partially observable",
        "justification": (
            "A passenger's destination is private until they board and press a car button, "
            "so the dispatcher must commit a car to a call before it knows where that "
            "call is going. Each agent also sees only its own sensors and inbox."
        ),
    },
    {
        "property": "Agents",
        "value": "Multi-agent (cooperative)",
        "justification": (
            "Cars bid against each other in Contract Net auctions, but they are scored on "
            "one shared fleet objective, so the competition is a mechanism for allocating "
            "work rather than a conflict of interest."
        ),
    },
    {
        "property": "Determinism",
        "value": "Stochastic",
        "justification": (
            "Arrivals follow a Poisson process with a time-varying rate, and faults are "
            "injected. Given a seed the whole run replays exactly, which makes the "
            "stochasticity reproducible rather than uncontrolled."
        ),
    },
    {
        "property": "Episodic vs sequential",
        "value": "Sequential",
        "justification": (
            "Assigning a call now changes which car is well placed for the next ten "
            "minutes of calls, so actions cannot be evaluated in isolation."
        ),
    },
    {
        "property": "Static vs dynamic",
        "value": "Dynamic",
        "justification": (
            "Passengers keep arriving while the agents deliberate, so a plan can be stale "
            "before it is executed. This is why replanning is event-driven and closed-loop."
        ),
    },
    {
        "property": "Discrete vs continuous",
        "value": "Discrete (discretised from continuous)",
        "justification": (
            "Floors and one-second ticks are discrete, abstracted from genuinely "
            "continuous motion. The frontend interpolates between ticks so the animation "
            "looks continuous without the engine having to be."
        ),
    },
    {
        "property": "Known vs unknown",
        "value": "Known physics, unknown demand",
        "justification": (
            "The agents know exactly how long a car takes to move a floor or cycle its "
            "doors. They do not know the arrival rates, which the TrafficMonitorAgent has "
            "to estimate online — so the environment is only partly known."
        ),
    },
]

#: The system-level PEAS, refined per agent in each agent's docstring and in the UI.
SYSTEM_PEAS: dict[str, str] = {
    "performance": (
        "Average and 95th-percentile wait, ride time and system time; percentage of waits "
        "over 60 s; throughput; energy (floors travelled, stops, reversals); zero safety "
        "violations; fairness (no starved call)."
    ),
    "environment": (
        "A multi-storey building: floors with hall buttons and occupancy sensors, shafts, "
        "the cars, the passengers, and the other cars each car competes with."
    ),
    "actuators": (
        "Car motors (up/down/stop), doors (open/close/hold), hall lanterns and car "
        "displays, and FIPA-ACL messages on the bus."
    ),
    "sensors": (
        "Position encoders, load and door sensors, car and hall buttons, per-floor "
        "occupancy sensors, and the message inbox."
    ),
}


def create_app(scenario: str = "demo_story") -> FastAPI:
    """Build the FastAPI application around one simulation session."""
    session = SimulationSession(scenario)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Start the stepping task on boot and cancel it cleanly on shutdown."""
        await session.start()
        session.running = False  # start paused, so the demo opens on a still building
        try:
            yield
        finally:
            await session.shutdown()

    app = FastAPI(
        title="Smart Elevator Fleet Coordinator",
        description="Multi-agent elevator simulation — Fundamentals of AI case study.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.session = session

    # ---------------------------------------------------------------- state

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        """The current world snapshot."""
        return session.snapshot()

    @app.get("/api/meta")
    async def get_meta() -> dict[str, Any]:
        """Scenarios, strategies, PEAS tables and the environment classification."""
        return {
            "scenarios": available_scenarios(),
            "strategies": [s.as_dict() for s in STRATEGIES.values()],
            "agents": [
                {
                    "name": cls.__name__,
                    "agent_type": cls.agent_type,
                    "peas": cls.peas,
                    "docstring": " ".join((cls.__doc__ or "").split())[:600],
                }
                for cls in AGENT_CLASSES
            ],
            "environment": ENVIRONMENT_PROPERTIES,
            "system_peas": SYSTEM_PEAS,
            "rules": [
                {"name": r.name, "salience": r.salience, "description": r.description}
                for r in sorted(SAFETY_RULES, key=lambda r: -r.salience)
            ],
        }

    # -------------------------------------------------------------- controls

    @app.post("/api/play")
    async def play() -> dict[str, Any]:
        """Resume automatic stepping."""
        await session.start()
        session.running = True
        return {"running": True}

    @app.post("/api/pause")
    async def pause() -> dict[str, Any]:
        """Pause automatic stepping."""
        await session.stop()
        return {"running": False}

    @app.post("/api/step")
    async def step(request: StepRequest) -> dict[str, Any]:
        """Advance a fixed number of ticks while paused."""
        session.running = False
        await session.step_once(request.ticks)
        return {"tick": session.model.tick}

    @app.post("/api/speed")
    async def set_speed(request: SpeedRequest) -> dict[str, Any]:
        """Set the playback multiplier (1x to 50x)."""
        session.speed = request.speed
        return {"speed": session.speed}

    @app.post("/api/reset")
    async def reset(request: ResetRequest) -> dict[str, Any]:
        """Rebuild the simulation, optionally with a new scenario/strategy/seed/size."""
        session.running = False
        try:
            session.reset(
                scenario=request.scenario,
                strategy=request.strategy,
                seed=request.seed,
                floors=request.floors,
                cars=request.cars,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        await session.broadcast()
        return session.snapshot()

    # ------------------------------------------------------------ interaction

    @app.post("/api/passenger")
    async def add_passenger(request: PassengerRequest) -> dict[str, Any]:
        """Add one passenger, from clicking a floor in the dashboard."""
        model = session.model
        if not 0 <= request.origin < model.floors_count:
            raise HTTPException(status_code=400, detail="origin floor out of range")
        destination = request.destination
        if destination is None or destination == request.origin:
            choices = [f for f in range(model.floors_count) if f != request.origin]
            destination = model.random.choice(choices)
        if not 0 <= destination < model.floors_count:
            raise HTTPException(status_code=400, detail="destination floor out of range")
        weight = model.config.traffic.priority_weight if request.priority else 1.0
        passenger = model.spawn_passenger(request.origin, destination, weight, request.priority)
        await session.broadcast()
        return {
            "passenger_id": passenger.unique_id,
            "origin": passenger.origin,
            "destination": passenger.destination,
            "priority": passenger.priority,
        }

    @app.post("/api/inject")
    async def inject(request: InjectRequest) -> dict[str, Any]:
        """Inject a car fault, a fire alarm, an alarm reset, a repair or a rush."""
        try:
            detail = session.model.inject(
                request.kind, car=request.car, floor=request.floor, count=request.count
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        await session.broadcast()
        return detail

    # ---------------------------------------------------------------- inspect

    @app.get("/api/agent/{address}")
    async def inspect_agent(address: str) -> dict[str, Any]:
        """The agent inspector: type, PEAS, internal state and planned route."""
        model = session.model
        lookup: dict[str, Any] = {
            "dispatcher": model.dispatcher,
            "monitor": model.monitor,
            "safety": model.safety,
        }
        for car in model.cars:
            lookup[car.address] = car
        for floor_agent in model.floors:
            lookup[floor_agent.address] = floor_agent
        agent = lookup.get(address)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"no agent at {address!r}")
        return agent.describe()

    @app.get("/api/messages")
    async def get_messages(limit: int = 120) -> dict[str, Any]:
        """The message log, newest last."""
        return {
            "messages": [m.as_dict() for m in session.model.bus.recent(min(limit, 500))],
            "total": session.model.bus.total_sent,
        }

    @app.get("/api/auctions")
    async def get_auctions(limit: int = 20) -> dict[str, Any]:
        """Recent Contract Net rounds, for the auction panel."""
        history = session.model.dispatcher.auction_history[-limit:]
        return {"auctions": [a.as_dict() for a in history]}

    # -------------------------------------------------------------- search lab

    @app.get("/api/search-lab")
    async def search_lab(car: int | None = None) -> dict[str, Any]:
        """Snapshot a car's routing problem and run BFS/UCS/Greedy/A* on it."""
        return session.snapshot_routing_problem(car)

    @app.get("/api/search-lab/annealing")
    async def annealing_lab() -> dict[str, Any]:
        """Run SA and hill climbing on the live assignment and return both curves."""
        dispatcher = session.model.dispatcher
        current = dispatcher.reassignable()
        car_ids = [c.car_id for c in session.model.cars if c.available]
        synthetic = False

        if len(current) < 2 or len(car_ids) < 2:
            # Not enough live traffic to optimise. Rather than showing an empty panel,
            # build a representative instance from the building's own geometry, and say
            # clearly that it is constructed — the algorithms and the objective are the
            # real ones either way.
            synthetic = True
            floors = session.model.floors_count
            car_ids = [c.car_id for c in session.model.cars] or [0, 1]
            spread = [1, floors // 3, floors // 2, (2 * floors) // 3, floors - 2]
            current = {
                HallCall(
                    floor=min(max(f, 0), floors - 1),
                    direction=Direction.UP if i % 2 == 0 else Direction.DOWN,
                ): car_ids[0]
                for i, f in enumerate(sorted(set(spread)))
            }
            if len(current) < 2 or len(car_ids) < 2:
                return {
                    "available": False,
                    "reason": "The building needs at least two floors and two cars.",
                }
        planner = session.model.config.planner
        sa = simulated_annealing(
            current,
            car_ids,
            dispatcher.assignment_cost,
            session.model.random,
            iterations=planner.sa_iterations,
            initial_temp=planner.sa_initial_temp,
            cooling=planner.sa_cooling,
        )
        hc = hill_climbing(
            current, car_ids, dispatcher.assignment_cost, session.model.random, iterations=80
        )
        return {
            "available": True,
            "synthetic": synthetic,
            "calls": len(current),
            "cars": len(car_ids),
            "simulated_annealing": sa.as_dict(),
            "hill_climbing": hc.as_dict(),
        }

    @app.get("/api/search-lab/minimax")
    async def minimax_lab() -> dict[str, Any]:
        """Solve the parking game and report minimax vs alpha-beta node counts."""
        model = session.model
        idle = [c for c in model.cars if c.available and not c.assigned_calls and not c.riders]
        car_count = max(1, min(3, len(idle) or 2))
        demand = model.monitor.demand_estimate()
        floors = model.floors_count
        ranked = sorted(demand, key=lambda f: -demand[f])
        likely = [f for f in ranked if demand[f] > 0][:6] or [0, floors // 2, floors - 1]
        candidates = sorted({0, floors // 2, floors - 1, *ranked[:3]})
        result = minimax_parking(car_count, floors, candidates, likely)
        payload = result.as_dict()
        payload.update(
            {
                "idle_cars": len(idle),
                "cars_placed": car_count,
                "candidate_spots": candidates,
                "likely_floors": likely,
            }
        )
        return payload

    # --------------------------------------------------------------- benchmark

    @app.post("/api/benchmark")
    async def benchmark(request: BenchmarkRequest) -> dict[str, Any]:
        """Run a benchmark headlessly and return the tables.

        Executed in a worker thread: it is CPU-bound and would otherwise block the event
        loop, freezing the live animation for every connected client.
        """
        from elevator_mas.sim.benchmark import (
            DEFAULT_SCENARIOS,
            DEFAULT_STRATEGIES,
            run_benchmark,
        )

        scenarios = request.scenarios or list(DEFAULT_SCENARIOS)
        strategies = request.strategies or list(DEFAULT_STRATEGIES)
        for name in scenarios:
            if name not in available_scenarios():
                raise HTTPException(status_code=400, detail=f"unknown scenario {name!r}")
        for name in strategies:
            if name not in STRATEGIES:
                raise HTTPException(status_code=400, detail=f"unknown strategy {name!r}")

        seeds = list(range(1, request.seeds + 1))
        result = await asyncio.to_thread(
            run_benchmark, scenarios, strategies, seeds, request.ticks, False
        )
        aggregate = result.aggregate()
        comparison = result.comparison()
        return {
            "runs": len(result.runs),
            "scenarios": scenarios,
            "strategies": strategies,
            "seeds": seeds,
            "ticks": request.ticks,
            "summary": aggregate.to_dict(orient="records"),
            "comparison": comparison.to_dict(orient="records"),
            "wins": result.wins(),
        }

    # --------------------------------------------------------------- websocket

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Stream a snapshot per tick to one dashboard client."""
        await websocket.accept()
        queue = session.subscribe()
        try:
            await websocket.send_json(session.snapshot())
            while True:
                frame = await queue.get()
                await websocket.send_json(frame)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except RuntimeError:
            # The socket closed mid-send; nothing to recover, just stop streaming.
            pass
        finally:
            session.unsubscribe(queue)

    # ------------------------------------------------------------------ static

    @app.get("/")
    async def index() -> FileResponse:
        """Serve the single-page dashboard."""
        path = WEB_DIR / "index.html"
        if not path.exists():
            return JSONResponse({"detail": "dashboard not built"}, status_code=404)
        return FileResponse(path)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Liveness probe, also used by the smoke tests."""
        return {
            "status": "ok",
            "tick": session.model.tick,
            "scenario": session.scenario_name,
            "strategy": session.model.strategy.name,
        }

    static_dir = WEB_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


#: Module-level app for `uvicorn elevator_mas.api.server:app --reload`.
app = create_app()
