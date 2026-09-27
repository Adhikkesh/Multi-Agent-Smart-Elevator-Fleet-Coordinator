"""The forward-chaining safety rules, and the behaviour they produce."""

from __future__ import annotations

from elevator_mas.config import ScenarioConfig
from elevator_mas.domain import CarState, DoorState
from elevator_mas.model import ElevatorModel
from elevator_mas.rules import ForwardChainingEngine, Rule


class TestEngine:
    """The inference engine itself, independent of elevators."""

    def test_fires_a_rule_whose_premises_hold(self) -> None:
        """The basic production cycle."""
        fired: list[str] = []
        rule = Rule(
            name="R",
            salience=1,
            condition=lambda wm, _w: [{}] if ("p", True) in wm else [],
            act=lambda _b, _w: fired.append("yes") or "done",
        )
        engine = ForwardChainingEngine([rule])
        engine.assert_fact(("p", True))
        engine.run(None, tick=1)
        assert fired == ["yes"]

    def test_does_not_fire_without_its_premises(self) -> None:
        """No premise, no conclusion."""
        engine = ForwardChainingEngine(
            [Rule(name="R", salience=1, condition=lambda wm, _w: [{}] if ("p", True) in wm else [])]
        )
        assert engine.run(None, tick=1) == []

    def test_chains_derived_facts_to_a_fixed_point(self) -> None:
        """A conclusion must be able to satisfy another rule's premise."""
        engine = ForwardChainingEngine(
            [
                Rule(
                    name="A",
                    salience=2,
                    condition=lambda wm, _w: [{}] if ("a", True) in wm else [],
                    conclude=lambda _b, _w: [("b", True)],
                ),
                Rule(
                    name="B",
                    salience=1,
                    condition=lambda wm, _w: [{}] if ("b", True) in wm else [],
                    conclude=lambda _b, _w: [("c", True)],
                ),
            ]
        )
        engine.assert_fact(("a", True))
        fired = engine.run(None, tick=1)
        assert {f.rule for f in fired} == {"A", "B"}
        assert engine.holds(("c", True))

    def test_salience_orders_the_firings(self) -> None:
        """Higher salience must fire first; fire recall outranks everything."""
        order: list[str] = []
        rules = [
            Rule(
                name="low",
                salience=1,
                condition=lambda _wm, _w: [{}],
                act=lambda _b, _w: order.append("low") or "",
            ),
            Rule(
                name="high",
                salience=99,
                condition=lambda _wm, _w: [{}],
                act=lambda _b, _w: order.append("high") or "",
            ),
        ]
        ForwardChainingEngine(rules).run(None, tick=1)
        assert order == ["high", "low"]

    def test_a_retracted_fact_stops_entailing(self) -> None:
        """Clearing an alarm must really change the conclusions."""
        engine = ForwardChainingEngine(
            [Rule(name="R", salience=1, condition=lambda wm, _w: [{}] if ("p", True) in wm else [])]
        )
        engine.assert_fact(("p", True))
        assert engine.run(None, tick=1)
        engine.retract(("p", True))
        assert engine.run(None, tick=2) == []

    def test_a_self_satisfying_rule_terminates(self) -> None:
        """A rule that re-asserts its own premise must not loop forever."""
        engine = ForwardChainingEngine(
            [
                Rule(
                    name="loop",
                    salience=1,
                    condition=lambda _wm, _w: [{}],
                    conclude=lambda _b, _w: [("x", True)],
                )
            ]
        )
        fired = engine.run(None, tick=1)
        assert len(fired) == 1


class TestFireEmergency:
    """Rules R1, R2, R3 and R7."""

    def test_alarm_recalls_every_car_to_the_lobby(self, small_config: ScenarioConfig) -> None:
        """The defining safety behaviour."""
        model = ElevatorModel(small_config)
        model.run(60)
        model.inject("fire_alarm")
        model.run(120)

        rules = {r.rule for r in model.safety.log}
        assert "R1_fire_recall" in rules
        assert "R3_block_hall_calls" in rules
        for car in model.cars:
            if car.out_of_service:
                continue
            assert car.floor == model.lobby, f"car {car.car_id} is at {car.floor}, not the lobby"
            assert car.state is CarState.FIRE_RECALL

    def test_alarm_blocks_hall_calls(self, small_config: ScenarioConfig) -> None:
        """No auction may award a call during an evacuation."""
        model = ElevatorModel(small_config)
        model.run(40)
        model.inject("fire_alarm")
        model.run(20)
        assert model.dispatcher.hall_calls_blocked
        assert not model.dispatcher.assignments

    def test_doors_are_held_open_at_the_lobby(self, small_config: ScenarioConfig) -> None:
        """Occupants must be able to leave."""
        model = ElevatorModel(small_config)
        model.run(40)
        model.inject("fire_alarm")
        model.run(140)
        assert any(car.door_state is DoorState.OPEN for car in model.cars if not car.out_of_service)
        assert "R2_fire_doors_open" in {r.rule for r in model.safety.log}

    def test_clearing_the_alarm_restores_service(self, small_config: ScenarioConfig) -> None:
        """Rule R7 must undo fire mode across the fleet."""
        model = ElevatorModel(small_config)
        model.run(40)
        model.inject("fire_alarm")
        model.run(80)
        model.inject("fire_clear")
        model.run(30)

        assert "R7_fire_cleared" in {r.rule for r in model.safety.log}
        assert not model.dispatcher.hall_calls_blocked
        assert not any(car.fire_mode for car in model.cars)


class TestCarFault:
    """Rule R4."""

    def test_fault_takes_the_car_out_of_service(self, small_config: ScenarioConfig) -> None:
        """A broken car must stop serving and must not keep its riders."""
        model = ElevatorModel(small_config)
        model.run(80)
        model.inject("car_fault", car=1)
        model.run(10)

        car = model.car(1)
        assert car is not None
        assert car.out_of_service
        assert car.state is CarState.OUT_OF_SERVICE
        assert car.riders == []
        assert car.assigned_calls == set()
        assert "R4_car_fault_out_of_service" in {r.rule for r in model.safety.log}

    def test_stranded_riders_keep_their_original_arrival_time(
        self, small_config: ScenarioConfig
    ) -> None:
        """A breakdown must not flatter the metrics by resetting anyone's clock."""
        model = ElevatorModel(small_config)
        model.run(100)
        car = max(model.cars, key=lambda c: len(c.riders))
        riders = list(car.riders)
        arrivals = {r.unique_id: r.record.arrival_tick for r in riders}
        if not riders:
            return  # nobody aboard in this run; nothing to assert
        model.inject("car_fault", car=car.car_id)
        model.run(5)
        for rider in riders:
            assert rider.record.arrival_tick == arrivals[rider.unique_id]
            assert rider.record.board_tick is None, "a re-queued rider is waiting again"
            assert rider.record.requeued >= 1

    def test_a_failed_car_refuses_new_calls(self, small_config: ScenarioConfig) -> None:
        """It must bid REFUSE, not a large number."""
        model = ElevatorModel(small_config)
        model.run(40)
        model.inject("car_fault", car=0)
        model.run(5)
        from elevator_mas.domain import Direction, HallCall

        bid = model.cars[0].marginal_cost(HallCall(3, Direction.UP))
        assert bid.refused
        assert bid.reason == "out of service"

    def test_repair_returns_the_car_to_service(self, small_config: ScenarioConfig) -> None:
        """Recovery must be complete: empty, idle, doors closed."""
        model = ElevatorModel(small_config)
        model.run(60)
        model.inject("car_fault", car=2)
        model.run(20)
        model.inject("car_repair", car=2)
        model.run(10)

        car = model.car(2)
        assert car is not None
        assert not car.out_of_service
        assert car.available


class TestOverload:
    """Rule R5."""

    def test_overload_refuses_further_boarding(self, small_config: ScenarioConfig) -> None:
        """An overloaded car must stop taking people, whatever the plan says."""
        model = ElevatorModel(small_config)
        model.run(30)
        car = model.cars[0]
        # Force an overload the physics would normally prevent.
        while len(car.riders) <= car.capacity:
            passenger = model.spawn_passenger(car.floor, (car.floor + 3) % model.floors_count)
            car.riders.append(passenger)
        model.step()
        assert "R5_overload_refuse_boarding" in {r.rule for r in model.safety.log}
        assert car.boarding_refused
        assert car.space == 0
