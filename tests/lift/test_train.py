"""Trainer: can learn (overfit), is deterministic, schedules lr, writes run artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from elevator_mas.learning.lift.config import (  # noqa: E402
    AugmentConfig,
    ModelConfig,
    TrainConfig,
    load_train_config,
    train_config_from_dict,
)
from elevator_mas.learning.lift.train_bc import (  # noqa: E402
    filter_trivial,
    load_checkpoint,
    lr_lambda,
    predict,
    save_checkpoint,
    train,
)

NO_AUG = AugmentConfig(permute=False, car_dropout_p=0.0, feature_noise=0.0)


def _subset(d: Any, n: int) -> Any:
    d = filter_trivial(d, 0.0, 0)
    return d.take(np.arange(min(n, len(d))))


@pytest.mark.slow
def test_pipeline_can_overfit_a_small_set(fixture_decisions: Any, tmp_path: Path) -> None:
    """Proves the losses, data path and optimiser can learn: >= 99 % on 256 decisions."""
    d = _subset(fixture_decisions, 256)
    cfg = TrainConfig(
        name="overfit",
        epochs=80,
        batch_size=64,
        lr=2e-3,
        warmup_steps=20,
        weight_decay=0.0,
        patience=1000,
        device="cpu",
        model=ModelConfig(dropout=0.0),
        augment=NO_AUG,
    )
    res = train(cfg, d, d, tmp_path, log=lambda _: None)
    assert res.best_val["agree_nontrivial_tie"] >= 99.0
    assert res.wall_s < 120


def test_same_seed_gives_identical_first_losses(fixture_decisions: Any, tmp_path: Path) -> None:
    d = _subset(fixture_decisions, 400)
    cfg = TrainConfig(
        name="det",
        epochs=1,
        batch_size=20,
        device="cpu",
        model=ModelConfig(d_model=16, n_layers=1, n_heads=2, ffn_mult=2),
    )
    a = train(cfg, d, d.take(np.arange(10)), tmp_path / "a", log=lambda _: None, max_steps=20)
    b = train(cfg, d, d.take(np.arange(10)), tmp_path / "b", log=lambda _: None, max_steps=20)
    assert len(a.first_losses) == 20
    assert a.first_losses == b.first_losses
    c = train(
        cfg.with_overrides(seed=7),
        d,
        d.take(np.arange(10)),
        tmp_path / "c",
        log=lambda _: None,
        max_steps=20,
    )
    assert c.first_losses != a.first_losses


def test_run_artefacts(fixture_decisions: Any, tmp_path: Path) -> None:
    d = _subset(fixture_decisions, 128)
    cfg = TrainConfig(
        name="art",
        epochs=2,
        batch_size=64,
        device="cpu",
        model=ModelConfig(d_model=16, n_layers=1, n_heads=2, ffn_mult=2),
    )
    res = train(cfg, d, d, tmp_path, log=lambda _: None)
    for f in ("run.json", "metrics.csv", "curves.png", "best.pt"):
        assert (res.run_dir / f).exists(), f
    info = json.loads((res.run_dir / "run.json").read_text())
    for k in ("git_sha", "config", "seed", "train_data_hash", "torch_version", "wall_time_s"):
        assert k in info
    net, ckpt = load_checkpoint(res.run_dir / "best.pt")
    assert ckpt["epoch"] == res.best_epoch
    assert not net.training


def test_checkpoint_with_another_schema_is_refused(tiny_net: Any, tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "x.pt", tiny_net, {})
    ckpt = torch.load(tmp_path / "x.pt", weights_only=False)
    ckpt["schema_hash"] = "deadbeef"
    torch.save(ckpt, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(tmp_path / "x.pt")


def test_lr_schedule_warms_up_then_decays_to_floor() -> None:
    base, floor = 1e-3, 1e-5
    lrs = [lr_lambda(s, 10, 100, base, floor) * base for s in range(101)]
    assert lrs[0] == pytest.approx(base / 10)
    assert lrs[9] == pytest.approx(base)
    assert all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:], strict=False))
    assert lrs[100] == pytest.approx(floor)


def test_filter_trivial_caps_single_car_decisions(fixture_decisions: Any) -> None:
    d = filter_trivial(fixture_decisions, 0.05, 0)
    n_valid = d.valid.sum(axis=1)
    assert np.all(d.winner >= 0)
    assert np.mean(n_valid == 1) <= 0.05 + 1e-9
    assert np.sum(n_valid >= 2) == np.sum(
        (fixture_decisions.winner >= 0) & (fixture_decisions.valid.sum(axis=1) >= 2)
    )


def test_predict_choice_respects_eligibility(tiny_net: Any, fixture_decisions: Any) -> None:
    d = fixture_decisions.take(np.arange(200))
    pred = predict(tiny_net, d)
    rows = np.flatnonzero(pred["choice"] >= 0)
    assert np.all(d.valid[rows, pred["choice"][rows]])
    assert np.all(pred["choice"][~d.valid.any(axis=1)] == -1)


def test_presets_load_and_unknown_keys_are_rejected() -> None:
    for preset in ("smoke", "default", "large", "dagger"):
        cfg = load_train_config(preset=preset)
        assert cfg.epochs >= 1
    assert load_train_config(preset="large").model.d_model == 128
    with pytest.raises(ValueError):
        train_config_from_dict({"epochz": 3})
    with pytest.raises(ValueError):
        train_config_from_dict({"model": {"width": 3}})
