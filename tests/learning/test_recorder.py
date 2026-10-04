"""Tests for decision recorder, expert dataset loading, shard structure, and splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from elevator_mas.learning.dataset import ExpertDataset
from elevator_mas.learning.schema import FEATURE_VERSION, KC, KCAR, KG, MAX_CARS, SCHEMA_HASH

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "expert_sample.npz"


def test_expert_sample_fixture_exists_and_valid() -> None:
    assert FIXTURE_PATH.exists(), f"Missing fixture at {FIXTURE_PATH}"
    data = np.load(FIXTURE_PATH, allow_pickle=True)

    # Check key existence and array shapes
    assert "call" in data
    assert "cars" in data
    assert "glob" in data
    assert "mask" in data
    assert "eligible" in data
    assert "teacher_cost" in data
    assert "winner" in data
    assert "meta_json" in data

    m = len(data["winner"])
    assert m == 1000
    assert data["call"].shape == (m, KC)
    assert data["cars"].shape == (m, MAX_CARS, KCAR)
    assert data["glob"].shape == (m, KG)
    assert data["mask"].shape == (m, MAX_CARS)
    assert data["eligible"].shape == (m, MAX_CARS)
    assert data["teacher_cost"].shape == (m, MAX_CARS)

    # Check metadata
    meta = json.loads(str(data["meta_json"]))
    assert meta["feature_version"] == FEATURE_VERSION
    assert meta["schema_hash"] == SCHEMA_HASH
    assert meta["max_cars"] == MAX_CARS


def test_expert_dataset_loader_and_stats() -> None:
    ds = ExpertDataset(FIXTURE_PATH)
    assert len(ds.winner) == 1000

    stats = ds.stats()
    assert stats.total_decisions == 1000
    assert 0.0 <= stats.all_refused_pct <= 100.0
    assert 0.0 <= stats.single_eligible_pct <= 100.0
    assert 0.0 <= stats.hard_decision_pct <= 100.0
    # Baselines should fall in reasonable range ~30-90%
    assert 30.0 <= stats.nearest_baseline_acc <= 90.0
    assert 30.0 <= stats.lowest_eta_acc <= 90.0


def test_expert_dataset_batch_iteration() -> None:
    ds = ExpertDataset(FIXTURE_PATH)
    batches = list(ds.iter_batches(batch_size=128, shuffle=True))
    assert len(batches) == 8  # ceil(1000 / 128) = 8
    total = sum(len(b["winner"]) for b in batches)
    assert total == 1000
