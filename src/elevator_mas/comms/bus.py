"""The message bus: delivery plus a complete, inspectable log.

Delivery is deliberately synchronous and queue-based rather than a direct method call.
Agents only ever see their own inbox, which is what keeps the multi-agent interaction
real (and visible in the dashboard) instead of a hidden function call.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator

from elevator_mas.comms.message import Message, Performative


class MessageBus:
    """Routes messages between named agents and keeps a bounded history.

    The history is what the dashboard's message log and the `messages per call` metric
    read from; `log_limit` stops a long headless run growing without bound.
    """

    def __init__(self, log_limit: int = 4000) -> None:
        self._inboxes: dict[str, deque[Message]] = defaultdict(deque)
        self._subscribers: set[str] = set()
        self.history: deque[Message] = deque(maxlen=log_limit)
        self.total_sent: int = 0
        self.per_conversation: defaultdict[str, int] = defaultdict(int)

    def register(self, address: str) -> None:
        """Announce an address so broadcasts can reach it."""
        self._subscribers.add(address)
        self._inboxes.setdefault(address, deque())

    def unregister(self, address: str) -> None:
        """Remove an address; its undelivered mail is dropped."""
        self._subscribers.discard(address)
        self._inboxes.pop(address, None)

    def send(self, message: Message) -> None:
        """Deliver one message, to its receiver or to every subscriber if broadcast."""
        self.total_sent += 1
        self.per_conversation[message.conversation_id] += 1
        self.history.append(message)
        if message.receiver is None:
            for address in self._subscribers:
                if address != message.sender:
                    self._inboxes[address].append(message)
        else:
            self._inboxes[message.receiver].append(message)

    def drain(self, address: str) -> list[Message]:
        """Take and clear everything addressed to `address`."""
        inbox = self._inboxes.get(address)
        if not inbox:
            return []
        messages = list(inbox)
        inbox.clear()
        return messages

    def peek(self, address: str) -> Iterator[Message]:
        """Iterate an inbox without consuming it."""
        return iter(tuple(self._inboxes.get(address, ())))

    def recent(
        self, limit: int = 100, performatives: Iterable[Performative] | None = None
    ) -> list[Message]:
        """The most recent messages, newest last, optionally filtered by performative."""
        wanted = set(performatives) if performatives else None
        out = [m for m in self.history if wanted is None or m.performative in wanted]
        return out[-limit:]

    def messages_in(self, conversation_id: str) -> int:
        """How many messages a given conversation used."""
        return self.per_conversation[conversation_id]

    def reset(self) -> None:
        """Clear all mail, history and counters, keeping registrations."""
        for inbox in self._inboxes.values():
            inbox.clear()
        self.history.clear()
        self.per_conversation.clear()
        self.total_sent = 0
