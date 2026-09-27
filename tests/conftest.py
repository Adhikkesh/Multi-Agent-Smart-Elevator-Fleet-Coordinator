"""Shared fixtures."""

from __future__ import annotations

import random

import pytest

from elevator_mas.config import (
    ArrivalPhase,
    BuildingConfig,
    ScenarioConfig,
    TrafficConfig,
)
from elevator_mas.model import ElevatorModel


@pytest.fixture
def rng() -> random.Random:
    """A seeded RNG, so every property test is reproducible."""
    return random.Random(20240)


@pytest.fixture
def small_config() -> ScenarioConfig:
    """A small, quick scenario for tests that just need a running model."""
    return ScenarioConfig(
        name="test",
        duration=200,
        seed=11,
        strategy="full",
        building=BuildingConfig(floors=10, cars=3, capacity=8),
        traffic=TrafficConfig(phases=[ArrivalPhase(until=200, rate=0.18, pattern="two_way")]),
    )


@pytest.fixture
def model(small_config: ScenarioConfig) -> ElevatorModel:
    """A fresh model built from `small_config`."""
    return ElevatorModel(small_config)
