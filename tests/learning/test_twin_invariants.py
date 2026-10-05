"""Property-style invariant tests for the fast twin simulator."""

from __future__ import annotations

import pytest

from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.twin.policies import CollectivePolicy, NearestPolicy
from elevator_mas.learning.twin.state import TwinSimulator


@pytest.mark.parametrize("seed", [10, 42, 99])
@pytest.mark.parametrize("pattern", ["up_peak", "down_peak", "two_way"])
def test_twin_conservation_and_physical_invariants(seed: int, pattern: str) -> None:
    sim = TwinSimulator(
        floors=10,
        cars=3,
        capacity=8,
        duration=300,
        seed=seed,
        rate=0.15,
        pattern=pattern,
    )
    policy = NearestPolicy()

    ctx, term = sim.advance()
    while not term:
        # Check invariants at every decision
        for car in sim.cars:
            assert len(car.riders) <= sim.capacity, f"Car {car.car_id} exceeded capacity"
            assert 0 <= car.floor < sim.floors, f"Car {car.car_id} out of bounds"
            if car.move_timer > 0 or car.target_floor is not None:
                # If in motion, door must be closed (0)
                assert car.door_state == 0, f"Car {car.car_id} moved with open doors"

        enc = encode_decision(ctx)
        act = policy.act({"cars": enc.cars, "eligible": enc.eligible})
        ctx, term = sim.step_action(int(act))

    # Conservation check at end of run
    arrived = len(sim.all_passengers)
    waiting = sum(len(q) for q in sim.waiting.values())
    riding = sum(len(c.riders) for c in sim.cars)
    delivered = len(sim.delivered_passengers)

    assert arrived == waiting + riding + delivered, (
        f"Conservation broken: arrived ({arrived}) != "
        f"waiting ({waiting}) + riding ({riding}) + delivered ({delivered})"
    )


def test_twin_seed_determinism() -> None:
    """Same seed and policy must produce identical trajectories and metrics."""
    sim1 = TwinSimulator(floors=12, cars=3, duration=200, seed=1234, rate=0.20)
    sim2 = TwinSimulator(floors=12, cars=3, duration=200, seed=1234, rate=0.20)
    pol = CollectivePolicy()

    # Run sim 1
    ctx1, term1 = sim1.advance()
    while not term1:
        enc1 = encode_decision(ctx1)
        act1 = pol.act({"cars": enc1.cars, "eligible": enc1.eligible})
        ctx1, term1 = sim1.step_action(int(act1))

    # Run sim 2
    ctx2, term2 = sim2.advance()
    while not term2:
        enc2 = encode_decision(ctx2)
        act2 = pol.act({"cars": enc2.cars, "eligible": enc2.eligible})
        ctx2, term2 = sim2.step_action(int(act2))

    m1 = sim1.get_metrics()
    m2 = sim2.get_metrics()
    assert m1 == m2
    assert [c.floors_travelled for c in sim1.cars] == [c.floors_travelled for c in sim2.cars]


def test_twin_drain_delivers_everyone() -> None:
    """When arrival rate drops to zero (drain), all passengers are eventually delivered."""
    sim = TwinSimulator(floors=8, cars=2, capacity=6, duration=200, seed=77, rate=0.25)
    pol = NearestPolicy()

    ctx, term = sim.advance()
    while not term:
        enc = encode_decision(ctx)
        act = pol.act({"cars": enc.cars, "eligible": enc.eligible})
        ctx, term = sim.step_action(int(act))

    # Drain phase: rate = 0, extend duration until empty
    sim.default_rate = 0.0
    sim.duration = 1000
    ctx, term = sim.advance()
    while not term and (sim.all_passengers != sim.delivered_passengers):
        enc = encode_decision(ctx)
        act = pol.act({"cars": enc.cars, "eligible": enc.eligible})
        ctx, term = sim.step_action(int(act))

    # Run remaining ticks to let all aboard alight
    def _still_active() -> bool:
        has_waiting = sum(len(q) for q in sim.waiting.values()) > 0
        has_riding = sum(len(c.riders) for c in sim.cars) > 0
        return has_waiting or has_riding

    while _still_active() and sim.tick < 2000:
        sim._step_tick()

    assert len(sim.delivered_passengers) == len(sim.all_passengers)


def test_twin_fault_ejects_riders_and_redecides() -> None:
    """A car fault empties car riders to waiting and re-releases assigned calls."""
    sim = TwinSimulator(floors=10, cars=2, duration=200, seed=55, rate=0.2)
    # Step until some riders board or calls assigned
    for _ in range(50):
        sim._step_tick()

    # Put a rider in car 0
    sim.cars[0].riders.append(sim.all_passengers[0])
    sim.cars[0].assigned_calls.add((5, 1))

    sim.inject_fault(0)
    assert sim.cars[0].out_of_service is True
    assert sim.cars[0].available is False
    assert len(sim.cars[0].riders) == 0
    assert len(sim.cars[0].assigned_calls) == 0
    # Released call must now be in unassigned_calls
    assert (5, 1) in sim.unassigned_calls

    sim.repair_car(0)
    assert sim.cars[0].out_of_service is False
    assert sim.cars[0].available is True


def test_twin_fire_recall() -> None:
    """Fire alarm recalls cars to lobby and blocks hall calls."""
    sim = TwinSimulator(floors=10, cars=2, duration=200, seed=88, rate=0.2)
    for _ in range(30):
        sim._step_tick()

    sim.inject_fire_alarm()
    assert sim.hall_calls_blocked is True
    assert len(sim.unassigned_calls) == 0

    # Step ticks until cars reach lobby
    for _ in range(100):
        sim._step_tick()

    for car in sim.cars:
        assert car.floor == sim.lobby
        assert car.door_state == 2  # Open at lobby

    sim.clear_fire_alarm()
    assert sim.hall_calls_blocked is False
    for car in sim.cars:
        assert car.fire_mode is False


def test_scenario_events_fire_in_the_twin() -> None:
    """Regression: the twin read `ev.type`/`ev.car_id` and never applied any event."""
    from elevator_mas.config import ScenarioConfig
    from elevator_mas.learning.twin.state import TwinSimulator

    cfg = ScenarioConfig.load("fire_emergency")
    fire = next(e for e in cfg.events if e.kind == "fire_alarm")
    clear = next(e for e in cfg.events if e.kind == "fire_clear")
    sim = TwinSimulator(config=cfg, seed=1)
    while sim.tick < fire.tick:
        sim._step_tick()
    assert sim.hall_calls_blocked
    assert all(c.fire_mode for c in sim.cars)
    while sim.tick < clear.tick:
        sim._step_tick()
    assert not sim.hall_calls_blocked
