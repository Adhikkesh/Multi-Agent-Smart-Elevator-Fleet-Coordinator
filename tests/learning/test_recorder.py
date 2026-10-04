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


def test_fixture_carries_teacher_source_and_executed_arrays() -> None:
    data = np.load(FIXTURE_PATH)
    m = len(data["winner"])
    for key in ("teacher", "source", "executed"):
        assert data[key].shape == (m,)
    # Expert data: the executed award *is* the teacher's award.
    np.testing.assert_array_equal(data["executed"], data["winner"])
    assert set(np.unique(data["teacher"])) <= {0, 1}
    assert np.all(data["source"] == 0)


def test_labels_are_finite_sentinels_and_consistent_with_eligibility() -> None:
    ds = ExpertDataset(FIXTURE_PATH)
    assert np.all(np.isfinite(ds.teacher_cost))
    priced = ds.teacher_cost < 1e5
    np.testing.assert_array_equal(priced & ds.mask, ds.eligible & ds.mask)
    valid = ds.winner >= 0
    rows = np.flatnonzero(valid)
    assert np.all(ds.eligible[rows, ds.winner[rows]])
    # The label winner is the argmin of the teacher costs (ties to the lowest car id).
    np.testing.assert_array_equal(np.argmin(ds.teacher_cost[rows], axis=1), ds.winner[rows])


def test_split_codes_follow_run_seed() -> None:
    from elevator_mas.learning.recorder import determine_split

    assert determine_split(0, 15, 4) == 0
    assert determine_split(7999, 15, 4) == 0
    assert determine_split(8000, 15, 4) == 1
    assert determine_split(8499, 15, 4) == 1
    assert determine_split(8500, 15, 4) == 2
    assert determine_split(8500, 36, 8) == 3


def test_dataset_split_filter_and_unknown_split() -> None:
    import pytest

    assert len(ExpertDataset(FIXTURE_PATH, split="train")) == 1000
    assert len(ExpertDataset(FIXTURE_PATH, split="val")) == 0
    with pytest.raises(ValueError):
        ExpertDataset(FIXTURE_PATH, split="bogus")


def test_hard_mask_counts_exact_ties_at_zero() -> None:
    ds = ExpertDataset(FIXTURE_PATH)
    sub = ds.subset(np.arange(2))
    sub.eligible[:] = False
    sub.mask[:] = False
    sub.eligible[:, :3] = True
    sub.mask[:, :3] = True
    sub.teacher_cost[:] = 1e6
    sub.teacher_cost[0, :3] = [0.0, 0.0, 5.0]  # exact tie at zero -> hard
    sub.teacher_cost[1, :3] = [10.0, 20.0, 30.0]  # clear winner -> easy
    sub.winner[:] = 0
    np.testing.assert_array_equal(sub.hard_mask(), [True, False])


def test_recorder_labels_from_shadow_bids_for_learned_auctions(tmp_path: Path) -> None:
    from elevator_mas.comms.board import DecisionEvent, FleetPolicy
    from elevator_mas.config import BuildingConfig
    from elevator_mas.domain import Bid, Direction, HallCall
    from elevator_mas.learning.recorder import DecisionRecorder

    statuses = (make_status(0, 2), make_status(1, 9))
    common = {
        "tick": 5,
        "call": HallCall(floor=4, direction=Direction.UP),
        "urgency": 0,
        "waiting": 1,
        "statuses": statuses,
        "policy": FleetPolicy(),
        "building": BuildingConfig(floors=12, cars=2),
        "seed": 3,
    }
    learned = (Bid(car_id=0, total=9.0), Bid(car_id=1, total=1.0))
    shadow = (Bid(car_id=0, total=2.0, wait=2.0), Bid(car_id=1, total=7.0, wait=7.0))
    rec = DecisionRecorder(tmp_path, source="dagger_r1")
    rec(DecisionEvent(**common, bids=learned, winner=1, strategy="full", bidder="learned"))
    rec(
        DecisionEvent(
            **common,
            bids=learned,
            winner=1,
            strategy="full",
            bidder="learned",
            shadow_bids=shadow,
        )
    )
    assert rec.skipped_unlabelled == 1
    shard = rec.flush()
    assert shard is not None
    data = np.load(shard)
    assert data["winner"].tolist() == [0]  # teacher's choice
    assert data["executed"].tolist() == [1]  # learner's executed award
    assert data["teacher"].tolist() == [1]  # labelled by the "full" teacher
    assert data["source"].tolist() == [1]
    np.testing.assert_allclose(data["teacher_wait"][0, :2], [2.0, 7.0])


def make_status(car_id: int, floor: int) -> object:
    from elevator_mas.comms.board import CarStatus
    from elevator_mas.domain import Direction

    return CarStatus(
        car_id=car_id,
        address=f"car-{car_id}",
        tick=5,
        floor=floor,
        direction=Direction.IDLE,
        load=0,
        capacity=10,
        space=10,
        available=True,
        out_of_service=False,
        fire_mode=False,
        door="closed",
        door_blocked_ticks=0,
    )
