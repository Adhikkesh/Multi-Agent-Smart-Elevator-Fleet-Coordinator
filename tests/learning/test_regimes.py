"""Tests for regimes configuration loader and regime specs."""

from __future__ import annotations

from elevator_mas.learning.regimes import load_regimes_config


def test_load_regimes_config() -> None:
    cfg = load_regimes_config()
    assert "up_peak" in cfg.eval_regimes
    assert "down_peak" in cfg.eval_regimes
    assert "two_way" in cfg.eval_regimes
    assert "interfloor" in cfg.eval_regimes

    up = cfg.eval_regimes["up_peak"]
    assert up.floors == 15
    assert up.cars == 4
    assert up.capacity == 10
    assert up.duration == 900
    assert up.rate == 0.20

    assert len(cfg.val_seeds) == 50
    assert cfg.val_seeds[0] == 8000
    assert len(cfg.test_seeds) == 100
    assert cfg.test_seeds[0] == 8500
