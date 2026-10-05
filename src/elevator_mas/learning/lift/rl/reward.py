"""The cooperative team reward, accumulated tick by tick in the real simulator.

Per simulated second (tick)::

    r = -( n_waiting * w_wait + n_riding * w_ride + floors_moved * w_energy
           + new_long_waits * w_threshold )

with the Phase 3 coefficients (``w_wait = 1/100``, ``w_ride = 0.5/100``, ``w_energy = 0.02``,
``w_threshold = 0.5``, a long wait = 60 s). Summed over the ticks between two decisions this
is exactly ``-(Σ_waiting Δt + 0.5 Σ_riding Δt)/100 - 0.02·floors - 0.5·crossings``; the Phase 3
twin approximated the sums with end-of-interval counts, here they are exact. Every car
receives the same reward (a cooperative game): no car is paid for taking a call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients of the team reward (part of the model card)."""

    w_wait: float = 1.0 / 100.0
    w_ride: float = 0.5 / 100.0
    w_energy: float = 0.02
    w_threshold: float = 0.5
    threshold_sec: int = 60

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


class RewardTracker:
    """Computes each tick's team reward from the model's state.

    The reward may use privileged information (who is waiting, for how long): it is the
    training signal, never an input of the actor.
    """

    def __init__(self, model: Any, cfg: RewardConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or RewardConfig()
        self._floors = int(model.collector.energy.floors_travelled)
        self._crossed: set[int] = set()
        self.components = {"wait": 0.0, "ride": 0.0, "energy": 0.0, "threshold": 0.0}

    def tick_reward(self) -> float:
        """Reward of the tick that just ran (call once after every ``model.step()``)."""
        cfg, model = self.cfg, self.model
        tick = model.tick
        waiting = riding = crossings = 0
        for p in model.passengers:
            if p.boarded:
                riding += 1
            elif p.waiting:
                waiting += 1
                pid = p.record.passenger_id
                if tick - p.record.arrival_tick >= cfg.threshold_sec and pid not in self._crossed:
                    self._crossed.add(pid)
                    crossings += 1
        floors = int(model.collector.energy.floors_travelled)
        moved, self._floors = floors - self._floors, floors
        parts = {
            "wait": waiting * cfg.w_wait,
            "ride": riding * cfg.w_ride,
            "energy": moved * cfg.w_energy,
            "threshold": crossings * cfg.w_threshold,
        }
        for k, v in parts.items():
            self.components[k] += v
        return -sum(parts.values())
