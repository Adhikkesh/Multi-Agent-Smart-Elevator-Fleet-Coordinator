"""Privileged state for the centralised critic (CTDE) — never seen by the actor.

The actor (the exported network every car runs) sees only the public board. During training
the critic additionally sees what no car knows: exactly how many people wait on every floor
in each direction, how long the oldest of them has waited, and how many riders are bound for
each floor. This *asymmetric* critic is what makes "centralised training, decentralised
execution" more than a slogan: better advantage estimates, same deployable actor.

Layout (``KPRIV = 5 * MAX_FLOORS``, zero-padded above the building's top floor):
``[waiting_up/20 | waiting_down/20 | oldest_up/120 s | oldest_down/120 s | riders_to/10]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_FLOORS = 48
KPRIV = 5 * MAX_FLOORS


def privileged_features(model: Any) -> np.ndarray:
    """The critic-only state vector of a live ``ElevatorModel`` (float32 [KPRIV])."""
    out = np.zeros(KPRIV, dtype=np.float32)
    f = MAX_FLOORS
    tick = model.tick
    for p in model.passengers:
        if p.boarded:
            dest = p.destination
            if 0 <= dest < f:
                out[4 * f + dest] += 0.1
            continue
        if not p.waiting:
            continue
        fl = p.current_floor
        if not 0 <= fl < f:
            continue
        up = p.destination > fl
        out[(0 if up else f) + fl] += 1.0 / 20.0
        age = (tick - p.record.arrival_tick) / 120.0
        k = (2 * f if up else 3 * f) + fl
        out[k] = max(out[k], age)
    return np.clip(out, 0.0, 1.0)
