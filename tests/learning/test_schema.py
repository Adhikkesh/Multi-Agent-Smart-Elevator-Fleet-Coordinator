"""Tests for feature schema definitions, dimensions, and names."""

from __future__ import annotations

from elevator_mas.learning.schema import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    KC,
    KCAR,
    KG,
    MAX_CARS,
    PATTERNS,
    SCHEMA_HASH,
    get_schema_hash,
)


def test_schema_constants() -> None:
    assert FEATURE_VERSION == 2
    assert MAX_CARS == 16
    assert KC == 14
    assert KCAR == 26
    assert KG == 10


def test_feature_names_dimensions() -> None:
    assert len(FEATURE_NAMES["call"]) == KC
    assert len(FEATURE_NAMES["cars"]) == KCAR
    assert len(FEATURE_NAMES["glob"]) == KG


def test_patterns_one_hot() -> None:
    assert len(PATTERNS) == 5
    for p in ("up_peak", "down_peak", "two_way", "interfloor", "light"):
        assert p in PATTERNS


def test_schema_hash_stability() -> None:
    assert len(SCHEMA_HASH) == 16
    assert get_schema_hash() == SCHEMA_HASH


def test_glob_token_ends_with_the_four_cost_weights() -> None:
    assert FEATURE_NAMES["glob"][-4:] == [
        "weight_wait",
        "weight_ride",
        "weight_crowding",
        "weight_energy",
    ]
