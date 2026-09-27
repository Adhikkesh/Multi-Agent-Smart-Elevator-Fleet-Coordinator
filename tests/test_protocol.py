"""The Contract Net protocol and the message bus."""

from __future__ import annotations

from collections import Counter

from elevator_mas.comms import Message, MessageBus, Performative
from elevator_mas.config import ScenarioConfig
from elevator_mas.model import ElevatorModel


class TestMessageBus:
    """Delivery, broadcast and logging."""

    def test_delivers_to_the_named_receiver_only(self) -> None:
        """A directed message must not reach anyone else."""
        bus = MessageBus()
        bus.register("a")
        bus.register("b")
        bus.send(Message(Performative.INFORM, "a", "b", "c1"))
        assert bus.drain("b")
        assert bus.drain("a") == []

    def test_broadcast_reaches_everyone_but_the_sender(self) -> None:
        """The CFP goes to every car, but does not echo back to the dispatcher."""
        bus = MessageBus()
        for address in ("dispatcher", "car-0", "car-1"):
            bus.register(address)
        bus.send(Message(Performative.CFP, "dispatcher", None, "c1"))
        assert len(bus.drain("car-0")) == 1
        assert len(bus.drain("car-1")) == 1
        assert bus.drain("dispatcher") == []

    def test_draining_clears_the_inbox(self) -> None:
        """A message must be consumed exactly once."""
        bus = MessageBus()
        bus.register("a")
        bus.send(Message(Performative.INFORM, "x", "a", "c1"))
        assert len(bus.drain("a")) == 1
        assert bus.drain("a") == []

    def test_history_is_bounded(self) -> None:
        """A long headless run must not grow the log without limit."""
        bus = MessageBus(log_limit=50)
        bus.register("a")
        for _ in range(200):
            bus.send(Message(Performative.INFORM, "x", "a", "c"))
        assert len(bus.history) == 50
        assert bus.total_sent == 200


class TestContractNet:
    """The auction protocol invariants."""

    def test_exactly_one_accept_per_call(self, small_config: ScenarioConfig) -> None:
        """A call must be awarded to exactly one car, or the fleet double-serves it."""
        model = ElevatorModel(small_config)
        model.run(200)

        accepts: Counter[tuple[str, int, str]] = Counter()
        for message in model.bus.history:
            if message.performative is Performative.ACCEPT_PROPOSAL:
                call = message.content.get("call")
                if call is not None and message.content.get("reason") != "global reassignment":
                    accepts[(message.conversation_id, call.floor, call.direction.name)] += 1
        assert accepts, "no call was ever awarded"
        assert all(count == 1 for count in accepts.values()), (
            f"a call was accepted more than once: {[k for k, v in accepts.items() if v > 1]}"
        )

    def test_every_cfp_is_answered_by_every_car(self, small_config: ScenarioConfig) -> None:
        """Each car must PROPOSE or REFUSE — silence would stall the protocol."""
        model = ElevatorModel(small_config)
        model.run(120)

        cfps = [m for m in model.bus.history if m.performative is Performative.CFP]
        assert cfps
        for cfp in cfps[:20]:
            replies = [
                m
                for m in model.bus.history
                if m.conversation_id == cfp.conversation_id
                and m.performative in (Performative.PROPOSE, Performative.REFUSE)
            ]
            assert len(replies) == len(model.cars), (
                f"{len(replies)} replies to a CFP from {len(model.cars)} cars"
            )

    def test_the_lowest_bid_wins(self, small_config: ScenarioConfig) -> None:
        """The award must actually go to the cheapest proposal."""
        model = ElevatorModel(small_config)
        model.run(200)

        checked = 0
        for auction in model.dispatcher.auction_history:
            viable = [b for b in auction.bids if not b.refused]
            if not viable or auction.winner is None:
                continue
            best = min(b.total for b in viable)
            winner = next(b for b in auction.bids if b.car_id == auction.winner)
            assert winner.total == best
            checked += 1
        assert checked > 0, "no completed auction to check"

    def test_out_of_service_cars_refuse(self, small_config: ScenarioConfig) -> None:
        """A broken car must answer REFUSE, so it is never awarded work."""
        model = ElevatorModel(small_config)
        model.run(50)
        model.inject("car_fault", car=0)
        model.run(60)

        refusals = [
            m
            for m in model.bus.history
            if m.performative is Performative.REFUSE and m.sender == "car-0"
        ]
        assert refusals, "the failed car never refused a call"
        assert any("out of service" in str(m.content.get("reason", "")) for m in refusals)

    def test_a_failure_gets_the_calls_re_auctioned(self, small_config: ScenarioConfig) -> None:
        """Nobody may be stranded because their car broke."""
        model = ElevatorModel(small_config)
        model.run(120)
        busiest = max(model.cars, key=lambda c: len(c.assigned_calls))
        released = set(busiest.assigned_calls)
        model.inject("car_fault", car=busiest.car_id)
        model.run(60)

        for call in released:
            if model.has_waiting(call.floor, call.direction):
                assert model.dispatcher.assignments.get(call) != busiest.car_id

    def test_every_message_carries_a_conversation_id(self, small_config: ScenarioConfig) -> None:
        """Threading is what makes the log auditable."""
        model = ElevatorModel(small_config)
        model.run(80)
        assert all(m.conversation_id for m in model.bus.history)

    def test_fire_mode_stops_new_awards(self, small_config: ScenarioConfig) -> None:
        """No ACCEPT may be issued once the alarm has blocked hall calls."""
        model = ElevatorModel(small_config)
        model.run(60)
        model.inject("fire_alarm")
        tick_at_alarm = model.tick
        model.run(60)

        late_accepts = [
            m
            for m in model.bus.history
            if m.performative is Performative.ACCEPT_PROPOSAL and m.tick > tick_at_alarm + 2
        ]
        assert not late_accepts, "a call was awarded during a fire evacuation"


class TestFairness:
    """Aging: an unserved call must escalate rather than starve."""

    def test_a_long_wait_escalates(self, small_config: ScenarioConfig) -> None:
        """The floor re-requests with rising urgency past the threshold."""
        config = small_config.model_copy(
            update={"building": small_config.building.model_copy(update={"cars": 1})}
        )
        model = ElevatorModel(config)
        model.run(400)

        escalated = [f for f in model.floors if max(f.escalations.values(), default=0) > 0]
        requests = [
            m
            for m in model.bus.history
            if m.performative is Performative.REQUEST and m.content.get("urgency", 0) > 0
        ]
        assert escalated or requests, "no call ever escalated despite a single overloaded car"
