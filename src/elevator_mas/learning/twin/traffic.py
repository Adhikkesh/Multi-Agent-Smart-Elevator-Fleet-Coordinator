"""Traffic arrival generation for the fast twin simulator.

Reproduces the exact same Poisson arrival process and 5 OD profiles
as elevator_mas.traffic.generator.ArrivalGenerator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

PATTERNS: Final[tuple[str, ...]] = (
    "up_peak",
    "down_peak",
    "two_way",
    "interfloor",
    "light",
)


@dataclass(frozen=True)
class TwinODProfile:
    lobby_origin: float
    lobby_destination: float
    label: str


TWIN_PROFILES: Final[dict[str, TwinODProfile]] = {
    "up_peak": TwinODProfile(0.90, 0.05, "Up peak"),
    "down_peak": TwinODProfile(0.05, 0.90, "Down peak"),
    "two_way": TwinODProfile(0.45, 0.45, "Two-way"),
    "interfloor": TwinODProfile(0.20, 0.20, "Interfloor"),
    "light": TwinODProfile(0.30, 0.30, "Light"),
}


def sample_poisson(lam: float, rand_val: float) -> int:
    """Knuth Poisson inverse transform matching ArrivalGenerator._poisson."""
    if lam <= 0.0:
        return 0
    cumulative = 0.0
    term = math.exp(-lam)
    k = 0
    while k < 64:
        cumulative += term
        if cumulative >= rand_val:
            return k
        k += 1
        term *= lam / k
    return k


class TwinTrafficGenerator:
    """Fast arrival generator producing identical statistics to Mesa traffic."""

    def __init__(
        self,
        floors: int = 15,
        lobby: int = 0,
        priority_prob: float = 0.0,
        priority_weight: float = 3.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.floors = floors
        self.lobby = lobby
        self.priority_prob = priority_prob
        self.priority_weight = priority_weight
        self.rng = rng or np.random.default_rng()

    def generate_arrivals(
        self,
        rate: float,
        pattern: str,
    ) -> list[tuple[int, int, float, bool]]:
        """Generate (origin, destination, weight, priority) for one tick."""
        if rate <= 0.0:
            return []
        count = sample_poisson(rate, float(self.rng.random()))
        if count == 0:
            return []

        prof = TWIN_PROFILES.get(pattern, TWIN_PROFILES["interfloor"])
        uppers = [f for f in range(self.floors) if f != self.lobby]
        if not uppers:
            return []

        out: list[tuple[int, int, float, bool]] = []
        for _ in range(count):
            r1 = float(self.rng.random())
            if r1 < prof.lobby_origin:
                origin = self.lobby
                dest = int(self.rng.choice(uppers))
            else:
                origin = int(self.rng.choice(uppers))
                r2 = float(self.rng.random())
                if r2 < prof.lobby_destination:
                    dest = self.lobby
                else:
                    choices = [f for f in range(self.floors) if f != origin]
                    dest = int(self.rng.choice(choices))

            is_priority = bool(float(self.rng.random()) < self.priority_prob)
            w = self.priority_weight if is_priority else 1.0
            out.append((origin, dest, w, is_priority))

        return out
