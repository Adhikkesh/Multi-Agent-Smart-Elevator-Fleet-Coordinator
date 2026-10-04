"""Tests for policy evaluation harness and summary metrics."""

from __future__ import annotations

from elevator_mas.learning.evaluate import evaluate, summarise
from elevator_mas.learning.regimes import load_regimes_config


def test_evaluate_twin_smoke() -> None:
    cfg = load_regimes_config()
    reg = cfg.eval_regimes["interfloor"]

    df = evaluate("nearest", regimes=[reg], backend="twin", seeds=[8000, 8001])
    assert len(df) == 2
    assert "avg_wait" in df.columns
    assert "p95_wait" in df.columns
    assert "throughput" in df.columns
    assert df["decisions"].sum() > 0

    summary = summarise(df, n_boot=200)
    assert len(summary) == 1
    assert "avg_wait_mean" in summary.columns
    assert "avg_wait_ci_low" in summary.columns
