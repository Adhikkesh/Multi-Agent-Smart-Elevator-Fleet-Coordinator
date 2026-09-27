"""Poisson arrival generation with a time-varying rate.

The environment is *stochastic* and its demand is *unknown* to the agents: only this
generator knows the true arrival rate, and the TrafficMonitorAgent has to estimate it
online from what it observes. All randomness comes from the model's seeded RNG, so a
whole run replays identically from its seed.
"""

from __future__ import annotations

import random

from elevator_mas.config import ScenarioConfig, TrafficPattern
from elevator_mas.traffic.profiles import profile_for


class ArrivalGenerator:
    """Draws the passengers that arrive on a given tick."""

    def __init__(self, config: ScenarioConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.floors = config.building.floors
        self.lobby = config.building.lobby

    def pattern_at(self, tick: int) -> TrafficPattern:
        """The true traffic pattern at `tick`."""
        return self.config.traffic.pattern_at(tick)

    def arrivals_for_tick(self, tick: int) -> list[tuple[int, int, float, bool]]:
        """The arrivals on this tick as `(origin, destination, weight, priority)`.

        The count is Poisson with the current rate; a rate of at most a few passengers
        per second makes the small-count inverse-transform draw below exact and cheap.
        """
        rate = self.config.traffic.rate_at(tick)
        if rate <= 0.0:
            return []
        count = self._poisson(rate)
        return [self._one_arrival(tick) for _ in range(count)]

    def _poisson(self, lam: float) -> int:
        """A Poisson draw by Knuth's product method, using the seeded RNG."""
        target = self.rng.random()
        cumulative = 0.0
        term = pow(2.718281828459045, -lam)
        k = 0
        while k < 64:
            cumulative += term
            if cumulative >= target:
                return k
            k += 1
            term *= lam / k
        return k

    def _one_arrival(self, tick: int) -> tuple[int, int, float, bool]:
        """One passenger's origin, destination, weight and priority flag."""
        profile = profile_for(self.pattern_at(tick))
        upper = [f for f in range(self.floors) if f != self.lobby]
        if self.rng.random() < profile.lobby_origin:
            origin = self.lobby
            destination = self.rng.choice(upper)
        else:
            origin = self.rng.choice(upper)
            if self.rng.random() < profile.lobby_destination:
                destination = self.lobby
            else:
                choices = [f for f in range(self.floors) if f != origin]
                destination = self.rng.choice(choices)
        priority = self.rng.random() < self.config.traffic.priority_probability
        weight = self.config.traffic.priority_weight if priority else 1.0
        return origin, destination, weight, priority

    def rush(self, floor: int, count: int, tick: int) -> list[tuple[int, int, float, bool]]:
        """A scripted burst of `count` passengers at `floor` (the `rush` event)."""
        out = []
        for _ in range(count):
            choices = [f for f in range(self.floors) if f != floor]
            destination = self.rng.choice(choices)
            priority = self.rng.random() < self.config.traffic.priority_probability
            weight = self.config.traffic.priority_weight if priority else 1.0
            out.append((floor, destination, weight, priority))
        del tick  # arrivals are stamped by the caller
        return out
