"""Baseline dispatch policies operating purely on encoded observations.

Provides Protocol definition and four built-in rule/heuristic policies:
- nearest: picks eligible car with minimal dist_to_call
- collective: prefers call_on_route and heading_to_call, tie-breaking by dist_to_call
- cost_greedy: teacher-mimicking scorer approximating A* marginal cost
- random_eligible: random choice among eligible cars
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Protocol for observation-driven elevator dispatch policies."""

    name: str

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Choose action per batch item given observation dict."""
        ...


class NearestPolicy:
    """Selects eligible car with minimum distance to call floor."""

    name = "nearest"

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        cars = obs["cars"]  # [B, 16, 26] or [16, 26]
        eligible = obs["eligible"]  # [B, 16] or [16]

        is_batched = cars.ndim == 3
        if not is_batched:
            cars = np.expand_dims(cars, 0)
            eligible = np.expand_dims(eligible, 0)

        b, n_cars, _ = cars.shape
        actions = np.zeros(b, dtype=np.int64)

        for i in range(b):
            elig = eligible[i]
            if not np.any(elig):
                actions[i] = 0
                continue
            # Feature #20 is dist_to_call
            dists = cars[i, :, 20]
            masked_dists = np.where(elig, dists, 1e6)
            actions[i] = int(np.argmin(masked_dists))

        return actions if is_batched else actions[0]


class CollectivePolicy:
    """Prefers cars where call is on route and heading towards call, then nearest."""

    name = "collective"

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        cars = obs["cars"]
        eligible = obs["eligible"]

        is_batched = cars.ndim == 3
        if not is_batched:
            cars = np.expand_dims(cars, 0)
            eligible = np.expand_dims(eligible, 0)

        b, n_cars, _ = cars.shape
        actions = np.zeros(b, dtype=np.int64)

        for i in range(b):
            elig = eligible[i]
            if not np.any(elig):
                actions[i] = 0
                continue

            on_route = cars[i, :, 23]  # Feature 23: call_on_route
            heading = cars[i, :, 22]  # Feature 22: heading_to_call
            dists = cars[i, :, 20]  # Feature 20: dist_to_call

            # Priority 1: on_route == 1.0 (cost = 0.0 + dist)
            # Priority 2: heading > 0 (cost = 10.0 + dist)
            # Priority 3: other (cost = 20.0 + dist)
            score = np.where(
                on_route > 0.5,
                dists,
                np.where(heading > 0.5, 10.0 + dists, 20.0 + dists),
            )
            masked_score = np.where(elig, score, 1e6)
            actions[i] = int(np.argmin(masked_score))

        return actions if is_batched else actions[0]


class CostGreedyPolicy:
    """Teacher-mimicking scorer approximating A* marginal cost from public features."""

    name = "cost_greedy"

    def __init__(
        self,
        w_wait: float = 1.0,
        w_ride: float = 0.5,
        w_crowd: float = 0.3,
        w_energy: float = 0.2,
    ) -> None:
        self.w_wait = w_wait
        self.w_ride = w_ride
        self.w_crowd = w_crowd
        self.w_energy = w_energy

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        cars = obs["cars"]
        eligible = obs["eligible"]

        is_batched = cars.ndim == 3
        if not is_batched:
            cars = np.expand_dims(cars, 0)
            eligible = np.expand_dims(eligible, 0)

        b, n_cars, _ = cars.shape
        actions = np.zeros(b, dtype=np.int64)

        for i in range(b):
            elig = eligible[i]
            if not np.any(elig):
                actions[i] = 0
                continue

            plan_end_eta = cars[i, :, 19] * 240.0
            dist_to_call = cars[i, :, 20] * 40.0
            load_ratio = cars[i, :, 4]
            on_route = cars[i, :, 23]

            # Approximate wait term
            est_wait = np.where(
                on_route > 0.5,
                dist_to_call * 2.0,
                plan_end_eta + dist_to_call * 2.0,
            )
            ride_term = load_ratio * 10.0
            crowding_term = load_ratio * 10.0
            energy_term = dist_to_call * 0.5 + 1.0

            total_cost = (
                self.w_wait * est_wait
                + self.w_ride * ride_term
                + self.w_crowd * crowding_term
                + self.w_energy * energy_term
            )

            masked_cost = np.where(elig, total_cost, 1e8)
            actions[i] = int(np.argmin(masked_cost))

        return actions if is_batched else actions[0]


class RandomEligiblePolicy:
    """Selects a uniformly random eligible car."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        eligible = obs["eligible"]
        is_batched = eligible.ndim == 2
        if not is_batched:
            eligible = np.expand_dims(eligible, 0)

        b, n_cars = eligible.shape
        actions = np.zeros(b, dtype=np.int64)

        for i in range(b):
            elig_indices = np.where(eligible[i])[0]
            if len(elig_indices) > 0:
                actions[i] = int(self.rng.choice(elig_indices))
            else:
                actions[i] = 0

        return actions if is_batched else actions[0]
