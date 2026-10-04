"""Phase 1: agents interact only by messages and the public status board.

These tests pin down the rule written on `CommunicatingAgent`: nobody calls another
agent's methods. The dispatcher *asks*, the cars *decide*, and the log proves it.
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest

from elevator_mas.agents.dispatcher import DispatcherAgent
from elevator_mas.agents.elevator import ElevatorAgent
from elevator_mas.agents.safety import SafetyAgent
from elevator_mas.comms import CarStatus, Order, Performative, StatusBoard
from elevator_mas.config import CostWeights, ScenarioConfig
from elevator_mas.domain import Direction
from elevator_mas.model import STAGES, ElevatorModel

#: Car methods that only the car's own code may run.
PRIVATE_CAR_METHODS = (
    "compute_bid",
    "marginal_cost",
    "nearest_car_bid",
    "collective_bid",
    "accept_call",
    "drop_call",
    "go_out_of_service",
)


def _caller_classes(depth: int = 3) -> set[str]:
    """Names of the classes whose methods are on the stack above the spied call."""
    names: set[str] = set()
    for frame_info in inspect.stack()[2 : 2 + depth + 20]:
        owner = frame_info.frame.f_locals.get("self")
        if owner is not None:
            names.add(type(owner).__name__)
    return names


def _spy(monkeypatch: pytest.MonkeyPatch, seen: list[tuple[str, set[str]]]) -> None:
    """Record, for every private car method call, which agent classes were on the stack."""
    for name in PRIVATE_CAR_METHODS:
        original: Callable[..., Any] = getattr(ElevatorAgent, name)

        def wrapper(
            self: ElevatorAgent,
            *args: Any,
            _original: Callable[..., Any] = original,
            _name: str = name,
            **kwargs: Any,
        ) -> Any:
            seen.append((_name, _caller_classes()))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ElevatorAgent, name, wrapper)


class TestNoDirectCalls:
    """The dispatcher and the safety agent never reach into a car."""

    def test_dispatcher_never_invokes_car_methods(
        self, small_config: ScenarioConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, set[str]]] = []
        _spy(monkeypatch, seen)
        model = ElevatorModel(small_config)
        model.run(200)
        assert seen, "the spy never fired, so the test proves nothing"
        offenders = [name for name, callers in seen if "DispatcherAgent" in callers]
        assert not offenders, f"dispatcher called car methods directly: {set(offenders)}"

    def test_safety_agent_never_invokes_car_methods(
        self, small_config: ScenarioConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, set[str]]] = []
        _spy(monkeypatch, seen)
        model = ElevatorModel(small_config)
        model.run(60)
        model.inject("car_fault", car=0)
        model.inject("fire_alarm")
        model.run(40)
        offenders = [name for name, callers in seen if "SafetyAgent" in callers]
        assert not offenders, f"safety agent called car methods directly: {set(offenders)}"

    def test_reassignment_is_done_by_messages(self, small_config: ScenarioConfig) -> None:
        """A global reassignment shows up as CANCEL + ACCEPT_PROPOSAL, not a method call."""
        config = small_config.model_copy(update={"duration": 400})
        model = ElevatorModel(config)
        model.run(400)
        reassigns = [
            m
            for m in model.bus.history
            if m.performative is Performative.ACCEPT_PROPOSAL
            and m.content.get("reason") == "global reassignment"
        ]
        cancels = [m for m in model.bus.history if m.performative is Performative.CANCEL]
        if reassigns:  # the annealer only moves work when it finds a saving
            assert cancels, "a call was moved without telling the car that lost it"
            assert all(m.receiver and m.receiver.startswith("car-") for m in reassigns)


class TestAuctionMessages:
    """Every step of the Contract Net is a message a car reads from its own inbox."""

    def test_one_inform_per_award(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(200)
        awards = Counter(
            m.conversation_id
            for m in model.bus.history
            if m.performative is Performative.ACCEPT_PROPOSAL
            and m.content.get("reason") != "global reassignment"
        )
        informs = Counter(
            m.conversation_id
            for m in model.bus.history
            if m.performative is Performative.INFORM
            and m.sender.startswith("car-")
            and (m.receiver or "").startswith("floor-")
            and "eta" in m.content
        )
        checked = 0
        for conversation, count in awards.items():
            assert informs[conversation] <= count == 1
            checked += 1
        assert checked > 0

    def test_each_car_replies_once_per_cfp(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(150)
        cfps = [m for m in model.bus.history if m.performative is Performative.CFP]
        assert cfps
        for cfp in cfps[:15]:
            senders = Counter(
                m.sender
                for m in model.bus.history
                if m.conversation_id == cfp.conversation_id
                and m.performative in (Performative.PROPOSE, Performative.REFUSE)
            )
            assert len(senders) == len(model.cars)
            assert set(senders.values()) == {1}

    def test_replies_go_to_the_auctioneer(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(100)
        replies = [
            m
            for m in model.bus.history
            if m.performative in (Performative.PROPOSE, Performative.REFUSE)
        ]
        assert replies
        assert all(m.receiver == model.dispatcher.address for m in replies)

    def test_every_award_has_a_decision_trace(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(200)
        awarded = [a for a in model.dispatcher.auction_history if a.winner is not None]
        assert awarded
        for auction in awarded:
            assert auction.reason, "an award without an explanation"
            assert (
                f"car {auction.winner}" in auction.reason.lower()
                or str(auction.winner) in auction.reason
            )
        assert "reason" in awarded[0].as_dict()


class TestStageOrder:
    """The tick has the documented shape."""

    def test_stage_list(self) -> None:
        assert STAGES == ("sense", "communicate", "negotiate", "decide", "act", "learn")

    def test_negotiation_finishes_within_the_tick(self, small_config: ScenarioConfig) -> None:
        """No CFP is left open across ticks: every round is closed before decide()."""
        model = ElevatorModel(small_config)
        for _ in range(120):
            model.step()
            assert not model.dispatcher.round_open


class TestSafetyOrders:
    """The safety agent supervises by REQUEST orders, not by editing cars."""

    def test_fault_becomes_an_out_of_service_order(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(40)
        model.inject("car_fault", car=1)
        model.run(5)
        orders = [
            m
            for m in model.bus.history
            if m.performative is Performative.REQUEST
            and m.content.get("order") is Order.OUT_OF_SERVICE
        ]
        assert orders, "no OUT_OF_SERVICE order was ever sent"
        assert all(m.sender == model.safety.address for m in orders)
        status = model.board.car(1)
        assert status is not None
        assert status.out_of_service and not status.available

    def test_fire_alarm_is_a_broadcast_of_orders(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(30)
        model.inject("fire_alarm")
        model.run(5)
        recalls = {
            m.receiver
            for m in model.bus.history
            if m.performative is Performative.REQUEST
            and m.content.get("order") is Order.FIRE_RECALL
        }
        assert {c.address for c in model.cars} <= recalls
        assert all(s.fire_mode for s in model.board.cars())
        model.inject("fire_clear")
        model.run(5)
        assert not any(s.fire_mode for s in model.board.cars())

    def test_released_calls_are_re_auctioned(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(120)
        status = max(model.board.cars(), key=lambda s: len(s.assigned_calls))
        released = set(status.assigned_calls)
        model.inject("car_fault", car=status.car_id)
        model.run(40)
        for call in released:
            if model.has_waiting(call.floor, call.direction):
                assert model.dispatcher.assignments.get(call) != status.car_id


class TestStatusBoard:
    """The shared blackboard of public state."""

    def _status(self, car_id: int, **kw: Any) -> CarStatus:
        base: dict[str, Any] = {
            "car_id": car_id,
            "address": f"car-{car_id}",
            "tick": 0,
            "floor": 0,
            "direction": Direction.IDLE,
            "load": 0,
            "capacity": 8,
            "space": 8,
            "available": True,
            "out_of_service": False,
            "fire_mode": False,
            "door": "closed",
            "door_blocked_ticks": 0,
        }
        base.update(kw)
        return CarStatus(**base)

    def test_publish_and_read(self) -> None:
        board = StatusBoard(CostWeights())
        board.publish_car(self._status(1, floor=4))
        board.publish_car(self._status(0, floor=2))
        assert [s.car_id for s in board.cars()] == [0, 1]
        assert board.car(1) is not None and board.car(1).floor == 4
        assert board.car(9) is None and board.car(None) is None
        assert board.writes == 2

    def test_entries_are_immutable(self) -> None:
        status = self._status(0)
        with pytest.raises(AttributeError):
            status.floor = 5  # type: ignore[misc]

    def test_available_ids_respect_space_and_service(self) -> None:
        board = StatusBoard(CostWeights())
        board.publish_car(self._status(0))
        board.publish_car(self._status(1, space=0))
        board.publish_car(self._status(2, available=False, out_of_service=True))
        assert board.available_car_ids() == [0]
        assert board.available_car_ids(with_space=False) == [0, 1]

    def test_policy_is_published_by_the_monitor(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        model.run(60)
        assert model.board.policy.published_by == model.monitor.address
        assert model.board.policy.tick > 0
        assert model.weights is model.board.weights

    def test_board_matches_the_cars_after_each_tick(self, small_config: ScenarioConfig) -> None:
        """The public view is never stale: it equals the cars' own state at tick end."""
        model = ElevatorModel(small_config)
        for _ in range(80):
            model.step()
            for car in model.cars:
                status = model.board.car(car.car_id)
                assert status is not None
                assert status.floor == car.floor
                assert status.load == car.load
                assert status.assigned_calls == frozenset(car.assigned_calls)

    def test_agent_types_are_all_present(self, small_config: ScenarioConfig) -> None:
        model = ElevatorModel(small_config)
        assert isinstance(model.dispatcher, DispatcherAgent)
        assert isinstance(model.safety, SafetyAgent)
