"""The safety supervisor: a knowledge-based agent."""

from __future__ import annotations

from typing import Any

from elevator_mas.agents.base import CommunicatingAgent
from elevator_mas.comms import Performative
from elevator_mas.rules import FiredRule, build_safety_engine


class SafetyAgent(CommunicatingAgent):
    """Enforces the safety rules, and explains every decision it makes.

    **AIMA agent type: knowledge-based (logical) agent** (§7.1, §9.3). Its behaviour is
    not written as procedural code: it asserts what it senses into working memory and
    forward-chains over a declarative rule base (`rules/safety_rules.py`, mirrored as
    Prolog in `docs/safety_rules.pl`) until no further conclusion follows. Adding a new
    safety policy means adding a rule, not editing this class.

    This is the right architecture for safety specifically, for two reasons. Firstly, the
    rules can be reviewed and audited as knowledge, by someone who does not read Python.
    Secondly, every action is traceable to the rule and the bindings that produced it —
    the rules-fired log in the dashboard is a genuine explanation, and the `salience`
    ordering makes the priority between competing rules explicit instead of implicit in
    statement order.

    It deliberately overrides the dispatcher and the cars. Fire recall is not a
    negotiation: the SafetyAgent runs before the dispatcher in the tick and blocks hall
    calls outright, so no auction can award a call during an evacuation.

    PEAS
      - **P** zero safety violations: no overloaded car moves, no car runs with its doors
        open, every car recalls on an alarm, no passenger is stranded in a failed car.
      - **E** every car's load, door and fault state, and the building alarm lines.
      - **A** recall cars, hold or re-open doors, take a car out of service, block hall
        calls, INFORM the dispatcher.
      - **S** load sensors, door sensors, fault lines, the fire-alarm input.
    """

    agent_type = "Knowledge-based agent (forward chaining)"
    peas = {
        "performance": "zero safety violations; correct recall; nobody stranded",
        "environment": "car loads, doors, fault states, building alarm lines",
        "actuators": "recall, hold/re-open doors, out-of-service, block hall calls",
        "sensors": "load sensors, door sensors, fault lines, fire-alarm input",
    }

    def __init__(self, model: Any) -> None:
        super().__init__(model, address="safety")
        self.engine = build_safety_engine()
        self.fired_this_tick: list[FiredRule] = []
        self.log: list[FiredRule] = []
        self.fire_alarm: bool = False

    # ------------------------------------------------------------------- sensing

    def sense(self) -> None:
        """Assert the current percepts into working memory.

        Facts that are no longer true are retracted, not merely left behind: an alarm that
        has been reset must actually stop entailing recall, which is what lets rule R7
        restore normal service.
        """
        self.collect_mail()
        if self.fire_alarm:
            self.engine.assert_fact(("fire_alarm", True))
        else:
            self.engine.retract(("fire_alarm", True))

        for message in self.mail_of(Performative.FAILURE):
            car_id = message.content.get("car_id")
            if car_id is not None:
                self.engine.assert_fact(("car_fault", car_id))

        for car in self.model.cars:
            if car.load > car.capacity:
                self.engine.assert_fact(("car_overload", car.car_id))
            else:
                self.engine.retract(("car_overload", car.car_id))
                self.engine.retract(("overloaded", car.car_id))

    # -------------------------------------------------------------- inference

    def decide(self) -> None:
        """Forward-chain to a fixed point; the rules' actions are their conclusions."""
        self.fired_this_tick = self.engine.run(self, self.model.tick)
        if self.fired_this_tick:
            self.log.extend(self.fired_this_tick)
            if len(self.log) > 300:
                del self.log[:-300]
            self.model.collector.total_rules_fired += len(self.fired_this_tick)

    def communicate(self) -> None:
        """Tell the dispatcher about any safety action that affects dispatching."""
        for record in self.fired_this_tick:
            self.send(
                Performative.INFORM,
                "dispatcher",
                self.model.new_conversation_id(),
                {"rule": record.rule, "effect": record.effect},
            )

    # -------------------------------- the world interface the rules act through
    # The rule functions receive this agent as `world`, so these are the only
    # actuators the rule base can reach. Keeping them here (rather than letting rules
    # touch the model directly) is what bounds what a safety rule is able to do.

    @property
    def cars(self) -> list[Any]:
        """Every car in the fleet."""
        return self.model.cars

    @property
    def lobby(self) -> int:
        """The lobby floor cars recall to."""
        return self.model.lobby

    @property
    def door_obstruction_limit(self) -> int:
        """How long a door may stay blocked before rule R6 re-opens it."""
        return 8

    def car(self, car_id: int) -> Any:
        """One car by id."""
        return self.model.car(car_id)

    def block_hall_calls(self) -> None:
        """Stop the dispatcher accepting hall calls, and clear every landing."""
        self.model.dispatcher.block_hall_calls()
        for floor_agent in self.model.floors:
            floor_agent.clear_calls()

    def take_out_of_service(self, car_id: int) -> int:
        """Fail a car and hand its calls back to the dispatcher for re-auction."""
        car = self.model.car(car_id)
        if car is None:
            return 0
        released = car.go_out_of_service()
        return self.model.dispatcher.reauction(released)

    def restore_normal_service(self) -> None:
        """Leave fire mode across the fleet and accept hall calls again."""
        for car in self.model.cars:
            if car.fire_mode:
                car.restore_normal_service()
        self.model.dispatcher.unblock_hall_calls()
        for fact in [("hall_calls_blocked", True), ("fire_cleared", True)]:
            self.engine.retract(fact)
        for car in self.model.cars:
            self.engine.retract(("recalling", car.car_id))
            self.engine.retract(("doors_held_open", car.car_id))

    # ------------------------------------------------------------------ external

    def trigger_fire_alarm(self) -> None:
        """Raise the alarm (a scenario event, or the dashboard button)."""
        self.fire_alarm = True

    def clear_fire_alarm(self) -> None:
        """Reset the alarm so rule R7 can restore service."""
        self.fire_alarm = False
        self.engine.retract(("fire_alarm", True))
        self.engine.retract(("hall_calls_blocked", True))

    def report_fault(self, car_id: int) -> None:
        """Inject a car fault; rule R4 does the rest."""
        self.engine.assert_fact(("car_fault", car_id))

    def clear_fault(self, car_id: int) -> None:
        """Repair a car and return it to service."""
        self.engine.retract(("car_fault", car_id))
        self.engine.retract(("out_of_service", car_id))
        car = self.model.car(car_id)
        if car is not None:
            car.return_to_service()

    # ------------------------------------------------------------------ reporting

    def recent_rules(self, limit: int = 20) -> list[dict[str, Any]]:
        """The most recent firings, for the dashboard's rules-fired log."""
        return [r.as_dict() for r in self.log[-limit:]]

    def describe(self) -> dict[str, Any]:
        """Agent-inspector payload, including the rule base itself."""
        base = super().describe()
        base.update(
            {
                "fire_alarm": self.fire_alarm,
                "facts": sorted(str(f) for f in self.engine.working_memory),
                "rules": [
                    {"name": r.name, "salience": r.salience, "description": r.description}
                    for r in self.engine.rules
                ],
                "total_fired": self.engine.total_fired,
                "recent": self.recent_rules(10),
            }
        )
        return base

    def step(self) -> None:
        """Mesa's default hook; this agent is driven by the model's staged activation."""
