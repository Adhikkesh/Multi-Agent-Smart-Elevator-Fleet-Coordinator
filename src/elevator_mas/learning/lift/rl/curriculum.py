"""Training-building sampler with a scheduled curriculum and a no-test-seed guard.

Domain randomisation follows ``configs/learning/regimes.yaml: training``: 6–32 floors (33–40
are held out for ``test_large``), 2–8 cars, capacity 6–16, 1–3 s/floor, 1–4 traffic phases
over the five patterns, arrival rate ``u · cars · 0.04``. The curriculum is *scheduled* (by
training progress), not adaptive:

* stage 1 (first 20 %): ``u ∈ [0.6, 1.2]``, no faults or fire;
* stage 2: full range, faults/fire with probability 0.25;
* stage 3 (last 20 %): as stage 2, but the hard patterns (``down_peak``, ``two_way``) are
  sampled twice as often.

Every simulator seed used for training is drawn from the *train* range (< 8000); asking for
anything else raises, so validation (8000–8499) and test (8500+) seeds can never leak in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elevator_mas.config import (
    ArrivalPhase,
    BuildingConfig,
    EventConfig,
    LiftConfig,
    ScenarioConfig,
    TimingConfig,
    TrafficConfig,
)
from elevator_mas.learning.schema import PATTERNS

TRAIN_SEED_LIMIT = 8000
HARD_PATTERNS = ("down_peak", "two_way")


class SeedLeakError(ValueError):
    """Raised when training code asks for a validation or test seed."""


def assert_no_test_seed(seed: int) -> int:
    """Return ``seed`` if it is a training seed, else raise."""
    if not 0 <= seed < TRAIN_SEED_LIMIT:
        raise SeedLeakError(
            f"seed {seed} is outside the training range [0, {TRAIN_SEED_LIMIT}); validation "
            "and test seeds must never be used for training"
        )
    return seed


@dataclass(frozen=True)
class Stage:
    name: str
    u_low: float
    u_high: float
    disturbance_p: float
    hard_weight: float


def stage_at(progress: float) -> Stage:
    """The curriculum stage at a training progress in [0, 1]."""
    if progress < 0.2:
        return Stage("easy", 0.6, 1.2, 0.0, 1.0)
    if progress < 0.8:
        return Stage("full", 0.6, 1.8, 0.25, 1.0)
    return Stage("hard", 0.6, 1.8, 0.25, 2.0)


#: Training settings, sampled 50/50 like the Phase 4 teachers: the `full`-like system (SA
#: reassignment, parking, adaptive weights) and bare Contract Net. The public cost weights in
#: the global token tell the policy which one it is in.
TRAIN_STRATEGIES = ("liftzero_ppo", "liftzero_ppo_cnp")


def sample_building(
    rng: np.random.Generator, stage: Stage, strategy: str | None = None
) -> ScenarioConfig:
    """One randomised training scenario (RL rollout mode)."""
    if strategy is None:
        strategy = TRAIN_STRATEGIES[int(rng.integers(0, len(TRAIN_STRATEGIES)))]
    floors = int(rng.integers(6, 33))
    min_cars = max(2, (floors + 7) // 8)
    cars = int(rng.integers(min_cars, min(min_cars + 3, 9)))
    capacity = int(rng.integers(6, 17))
    spf = int(rng.integers(1, 4))
    rate = float(rng.uniform(stage.u_low, stage.u_high)) * cars * 0.04
    duration = int(rng.integers(600, 1201))
    n_phases = int(rng.integers(1, 5))
    weights = np.array([stage.hard_weight if p in HARD_PATTERNS else 1.0 for p in PATTERNS])
    weights = weights / weights.sum()
    phases: list[ArrivalPhase] = []
    until = 0
    for k in range(n_phases):
        until = duration if k == n_phases - 1 else until + duration // n_phases
        pattern = str(rng.choice(PATTERNS, p=weights))
        phases.append(
            ArrivalPhase(until=until, pattern=pattern, rate=rate * float(rng.uniform(0.8, 1.2)))  # type: ignore[arg-type]
        )
    events: list[EventConfig] = []
    if rng.random() < stage.disturbance_p:
        t = int(rng.integers(duration // 4, duration * 3 // 4))
        if rng.random() < 0.6:
            car = int(rng.integers(0, cars))
            events += [
                EventConfig(tick=t, kind="car_fault", car=car),
                EventConfig(tick=t + int(rng.integers(60, 150)), kind="car_repair", car=car),
            ]
        else:
            events += [
                EventConfig(tick=t, kind="fire_alarm"),
                EventConfig(tick=t + int(rng.integers(60, 120)), kind="fire_clear"),
            ]
    seed = assert_no_test_seed(int(rng.integers(0, TRAIN_SEED_LIMIT)))
    return ScenarioConfig(
        name=f"ppo_{stage.name}_{seed}",
        strategy=strategy,
        seed=seed,
        duration=duration,
        building=BuildingConfig(floors=floors, cars=cars, capacity=capacity),
        timing=TimingConfig(seconds_per_floor=spf),
        traffic=TrafficConfig(rate=rate, pattern=phases[0].pattern, phases=phases),
        events=events,
        lift=LiftConfig(rollout=True),
    )
