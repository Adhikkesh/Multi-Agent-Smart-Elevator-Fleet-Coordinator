"""Augmentation keeps labels valid: permutation, winner-safe dropout, noise on reals only."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from elevator_mas.learning.lift.augment import (  # noqa: E402
    add_noise,
    continuous_indices,
    drop_cars,
    permute_cars,
)
from elevator_mas.learning.lift.train_bc import to_tensors  # noqa: E402


@pytest.fixture()
def batch(fixture_decisions: Any) -> dict[str, Any]:
    d = fixture_decisions
    keep = torch.from_numpy(d.winner >= 0).nonzero().squeeze(1)[:400].numpy()
    return to_tensors(d, keep, torch.device("cpu"))


def _winner_cost(b: dict[str, Any]) -> Any:
    return b["teacher_cost"].gather(1, b["winner"][:, None]).squeeze(1)


def test_permutation_moves_labels_with_cars(batch: dict[str, Any]) -> None:
    g = torch.Generator().manual_seed(0)
    p = permute_cars(batch, g)
    torch.testing.assert_close(_winner_cost(p), _winner_cost(batch))
    rows = torch.arange(len(batch["winner"]))
    torch.testing.assert_close(
        p["cars"][rows, p["winner"]], batch["cars"][rows, batch["winner"]], atol=0, rtol=0
    )
    assert torch.equal(p["mask"].sum(1), batch["mask"].sum(1))
    assert torch.equal(
        torch.sort(p["teacher_cost"], 1).values, torch.sort(batch["teacher_cost"], 1).values
    )


def test_permutation_actually_shuffles(batch: dict[str, Any]) -> None:
    p = permute_cars(batch, torch.Generator().manual_seed(1))
    assert not torch.equal(p["cars"], batch["cars"])


@pytest.mark.parametrize("seed", range(5))
def test_dropout_never_drops_a_winner_and_keeps_two_cars(batch: dict[str, Any], seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    out = drop_cars(batch, g, p=1.0, max_drop=3)
    rows = torch.arange(len(batch["winner"]))
    assert torch.all(out["mask"][rows, batch["winner"]])
    assert torch.all(out["eligible"][rows, batch["winner"]])
    before = (batch["mask"] & batch["eligible"]).sum(1)
    after = (out["mask"] & out["eligible"]).sum(1)
    assert torch.all(after >= torch.minimum(before, torch.tensor(2)))
    assert torch.all(before - after <= 3)
    assert torch.all(after <= before)
    assert (after < before).any()
    # Dropped cars are fully invisible: no features, sentinel cost.
    dropped = batch["mask"] & ~out["mask"]
    assert torch.all(out["cars"][dropped] == 0)
    assert torch.all(out["teacher_cost"][dropped] >= 1e5)
    torch.testing.assert_close(_winner_cost(out), _winner_cost(batch))


def test_dropout_keeps_tied_winners(batch: dict[str, Any]) -> None:
    b = {k: v.clone() for k, v in batch.items()}
    b["mask"][:, :3] = True
    b["eligible"][:, :3] = True
    b["teacher_cost"][:, :3] = torch.tensor([1.0, 1.0, 9.0])
    b["teacher_cost"][:, 3:] = 1e6
    b["mask"][:, 3:] = False
    b["winner"][:] = 0
    out = drop_cars(b, torch.Generator().manual_seed(0), p=1.0, max_drop=3)
    assert torch.all(out["mask"][:, :2])  # both tied winners survive


def test_dropout_probability_zero_is_identity(batch: dict[str, Any]) -> None:
    out = drop_cars(batch, torch.Generator().manual_seed(0), p=0.0, max_drop=3)
    assert out is batch


def test_noise_touches_only_continuous_features_of_real_cars(batch: dict[str, Any]) -> None:
    out = add_noise(batch, torch.Generator().manual_seed(0), sigma=0.05)
    for key in ("call", "glob", "cars"):
        cont = set(continuous_indices(key))
        for col in range(batch[key].shape[-1]):
            same = torch.equal(out[key][..., col], batch[key][..., col])
            if col not in cont:
                assert same, (key, col)
    assert not torch.equal(out["cars"], batch["cars"])
    pad = ~batch["mask"]
    assert torch.equal(out["cars"][pad], batch["cars"][pad])
    assert torch.equal(out["mask"], batch["mask"])
    assert out["cars"].min() >= -1.0 and out["cars"].max() <= 1.0


def test_continuous_indices_exclude_one_hots() -> None:
    from elevator_mas.learning.schema import FEATURE_NAMES

    names = [FEATURE_NAMES["cars"][i] for i in continuous_indices("cars")]
    for flag in ("dir_up", "door_open", "available", "call_on_route", "has_same_call"):
        assert flag not in names
    call_names = [FEATURE_NAMES["call"][i] for i in continuous_indices("call")]
    assert "call_dir" not in call_names and "pattern_up_peak" not in call_names
