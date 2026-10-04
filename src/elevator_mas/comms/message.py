"""The message type agents exchange.

Modelled on FIPA-ACL (AIMA 4e §2.4.4 multiagent communication): a performative saying
what the speech act *does*, plus sender, receiver, a conversation id threading one
Contract Net round together, and the content.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Performative(Enum):
    """The FIPA-ACL speech acts this system uses."""

    REQUEST = "REQUEST"
    CFP = "CFP"
    PROPOSE = "PROPOSE"
    REFUSE = "REFUSE"
    ACCEPT_PROPOSAL = "ACCEPT_PROPOSAL"
    REJECT_PROPOSAL = "REJECT_PROPOSAL"
    INFORM = "INFORM"
    CANCEL = "CANCEL"
    FAILURE = "FAILURE"


class Order(Enum):
    """The actions the SafetyAgent may REQUEST of other agents.

    Safety is a supervisor, not a puppeteer: it asks, and each agent carries the order
    out itself when it next reads its inbox. Keeping the vocabulary closed and explicit
    is what makes every safety intervention visible in the message log.
    """

    FIRE_RECALL = "fire_recall"
    HOLD_DOORS_OPEN = "hold_doors_open"
    BLOCK_HALL_CALLS = "block_hall_calls"
    RESTORE_SERVICE = "restore_service"
    OUT_OF_SERVICE = "out_of_service"
    RETURN_TO_SERVICE = "return_to_service"
    REFUSE_BOARDING = "refuse_boarding"
    REOPEN_DOORS = "reopen_doors"


_counter = itertools.count(1)


@dataclass
class Message:
    """One FIPA-ACL-style message.

    `receiver` is None for a broadcast (used for the CFP to every car).
    """

    performative: Performative
    sender: str
    receiver: str | None
    conversation_id: str
    content: dict[str, Any] = field(default_factory=dict)
    tick: int = 0
    seq: int = field(default_factory=lambda: next(_counter))

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the dashboard's message log."""
        return {
            "seq": self.seq,
            "tick": self.tick,
            "performative": self.performative.value,
            "sender": self.sender,
            "receiver": self.receiver or "broadcast",
            "conversation_id": self.conversation_id,
            "content": jsonable(self.content),
        }

    def __str__(self) -> str:
        target = self.receiver or "all"
        return f"[t={self.tick}] {self.sender} -{self.performative.value}-> {target}"


def jsonable(value: Any) -> Any:
    """Coerce enums, dataclasses and containers into strictly JSON-safe values.

    "Strictly" matters: browsers' `JSON.parse` rejects `Infinity` and `NaN`, which Python
    would otherwise happily emit (a refused bid costs infinity), so those become `None`.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    if hasattr(value, "as_dict"):
        return jsonable(value.as_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return value
