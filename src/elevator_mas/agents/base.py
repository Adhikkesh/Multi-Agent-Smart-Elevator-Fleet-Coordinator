"""The shared base class for every agent that talks on the message bus.

Mesa 3 API only: `Agent.__init__(self, model, ...)` calling `super().__init__(model)`,
no schedulers, and all randomness drawn from `model.random`.
"""

from __future__ import annotations

from typing import Any

from mesa import Agent

from elevator_mas.comms import Message, MessageBus, Performative


class CommunicatingAgent(Agent):
    """An agent with a bus address and an inbox.

    Subclasses implement the staged-activation hooks they need — `sense`, `communicate`,
    `decide`, `act` — which the model calls in that fixed order every tick via
    `model.agents_by_type[Cls].do(stage)`. The stages exist so that within one tick every
    agent senses the *same* world state before anyone acts on it; without that, results
    would depend on activation order rather than on the agents' reasoning.
    """

    #: The AIMA agent type, shown in the dashboard's agent inspector.
    agent_type: str = "abstract"
    #: The PEAS description, shown in the inspector and the Theory tab.
    peas: dict[str, str] = {}

    def __init__(self, model: Any, address: str) -> None:
        super().__init__(model)
        self.address = address
        self.bus: MessageBus = model.bus
        self.bus.register(address)
        self.inbox: list[Message] = []

    # ------------------------------------------------------------------ messaging

    def send(
        self,
        performative: Performative,
        receiver: str | None,
        conversation_id: str,
        content: dict[str, Any] | None = None,
    ) -> Message:
        """Put one message on the bus, stamped with the current tick."""
        message = Message(
            performative=performative,
            sender=self.address,
            receiver=receiver,
            conversation_id=conversation_id,
            content=content or {},
            tick=self.model.tick,
        )
        self.bus.send(message)
        return message

    def collect_mail(self) -> list[Message]:
        """Move everything addressed to this agent into `self.inbox`."""
        self.inbox = self.bus.drain(self.address)
        return self.inbox

    def mail_of(self, *performatives: Performative) -> list[Message]:
        """The messages in the current inbox with any of these performatives."""
        wanted = set(performatives)
        return [m for m in self.inbox if m.performative in wanted]

    # -------------------------------------------------------- staged activation

    def sense(self) -> None:
        """Stage 1: read the environment and the inbox. No side effects on others."""

    def communicate(self) -> None:
        """Stage 2: send messages (requests, bids, awards)."""

    def decide(self) -> None:
        """Stage 3: choose this tick's action from the sensed state."""

    def act(self) -> None:
        """Stage 4: apply the chosen action to the environment."""

    def describe(self) -> dict[str, Any]:
        """What the dashboard's agent inspector shows for this agent."""
        return {
            "address": self.address,
            "agent_type": self.agent_type,
            "peas": self.peas,
        }
