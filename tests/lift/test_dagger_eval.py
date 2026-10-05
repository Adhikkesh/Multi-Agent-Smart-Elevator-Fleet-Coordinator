"""DAgger data hygiene, closed-loop evaluation, offline report and the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from elevator_mas.cli import app
from elevator_mas.learning.dataset import ExpertDataset
from elevator_mas.learning.lift import dagger
from elevator_mas.learning.lift.eval_sim import (
    build_config,
    paired_comparisons,
    regime_names,
    run_one,
    seeds_for,
    summary_table,
    write_markdown,
)
from elevator_mas.learning.recorder import determine_split
from elevator_mas.learning.schema import SOURCES, SPLITS

# ------------------------------------------------------------------------- DAgger


@pytest.mark.parametrize("round_", [1, 2, 3])
def test_dagger_seeds_never_leak_into_val_or_test(round_: int) -> None:
    train, val = dagger.round_seeds(round_, 1500, 80)
    assert all(determine_split(s, 15, 4) == SPLITS["train"] for s in train)
    assert all(determine_split(s, 15, 4) == SPLITS["val"] for s in val)
    closed_loop = set(seeds_for("val")) | set(seeds_for("test"))
    assert not closed_loop & set(train)
    assert not closed_loop & set(val)


def test_dagger_rounds_use_disjoint_fresh_seeds() -> None:
    r1, v1 = dagger.round_seeds(1, 1500, 80)
    r2, v2 = dagger.round_seeds(2, 1500, 80)
    assert not set(r1) & set(r2) and not set(v1) & set(v2)
    assert min(r1) > 2258  # clear of the expert harvest's seeds (0-2258)
    with pytest.raises(ValueError):
        dagger.round_seeds(3, 5000, 80)


def test_dagger_collection_tags_and_labels(shipped_model: Path, tmp_path: Path) -> None:
    stats = dagger.collect([4000, 4001], shipped_model, 0.5, tmp_path, "dagger_r1", workers=1)
    assert stats.runs == 2 and stats.decisions > 0
    assert stats.teacher_awards + stats.learner_awards > 0
    ds = ExpertDataset(tmp_path)
    assert len(ds) == stats.decisions
    assert np.all(ds.source == SOURCES.index("dagger_r1"))
    assert np.all(ds.split_arr == SPLITS["train"])
    assert set(np.unique(ds.teacher)) <= {0, 1}
    # Labels are the teacher's argmin, not necessarily what was executed.
    rows = np.flatnonzero(ds.winner >= 0)
    np.testing.assert_array_equal(np.argmin(ds.teacher_cost[rows], axis=1), ds.winner[rows])


# -------------------------------------------------------------------- closed loop


def test_regime_and_seed_resolution() -> None:
    assert regime_names("regimes") == ["up_peak", "down_peak", "two_way", "interfloor"]
    assert len(regime_names("all")) == 13
    assert regime_names("up_peak,fire_emergency") == ["up_peak", "fire_emergency"]
    assert seeds_for("val", 3) == [8000, 8001, 8002]
    assert seeds_for("test")[0] == 8500 and len(seeds_for("test")) == 100
    cfg = build_config("two_way", "full", 8001)
    assert (cfg.building.floors, cfg.building.cars, cfg.duration) == (15, 4, 900)


def test_run_one_reports_safety_and_agreement(shipped_model: Path) -> None:
    row = run_one("liftzero_bc", "interfloor", 8000, shadow=True)
    assert row["violations"] == 0
    assert 0.0 <= row["disagree_pct"] <= 100.0
    assert row["avg_wait"] > 0 and row["calls"] > 0
    classical = run_one("full", "interfloor", 8000, shadow=True)
    assert not classical["shadow"] and np.isnan(classical["disagree_pct"])


def _synthetic() -> pd.DataFrame:
    rows = []
    for seed in range(10):
        for strat, wait in (("full", 10.0), ("liftzero_bc", 10.5)):
            rows.append(
                {
                    "strategy": strat,
                    "regime": "up_peak",
                    "seed": seed,
                    "avg_wait": wait + 0.01 * seed,
                    "p95_wait": 3 * wait,
                    "max_wait": 60.0,
                    "long_wait_pct": 1.0,
                    "throughput": 400.0,
                    "energy": 300.0,
                    "compute_ms_per_tick": 0.5,
                    "violations": 0,
                    "disagree_pct": np.nan if strat == "full" else 12.0,
                }
            )
    return pd.DataFrame(rows)


def test_paired_comparison_and_report(tmp_path: Path) -> None:
    df = _synthetic()
    comps = paired_comparisons(df, {"liftzero_bc": "full"})
    wait = next(c for c in comps if c.metric == "avg_wait")
    assert wait.rel_pct == pytest.approx(5.0, abs=0.1)
    assert wait.ci_low <= wait.rel_pct <= wait.ci_high
    assert wait.n == 10
    summ = summary_table(df)
    assert set(summ["strategy"]) == {"full", "liftzero_bc"}
    out = tmp_path / "cl.md"
    write_markdown(df, out)
    text = out.read_text()
    assert "Learned vs teacher" in text and "up_peak" in text


# ---------------------------------------------------------------------- offline


def test_offline_report_on_fixture(tiny_net: Any, fixture_decisions: Any, tmp_path: Path) -> None:
    from elevator_mas.learning.lift.eval_offline import (
        evaluate_split,
        label_conflicts,
        write_report,
    )

    rep = evaluate_split(tiny_net, fixture_decisions, "val")
    assert set(rep.baselines) == {"nearest", "lowest_eta", "cost_greedy"}
    assert rep.overall["n_nontrivial"] > 0
    assert rep.failure_modes and rep.mistakes
    assert rep.calibration
    c = label_conflicts(fixture_decisions)
    assert 0.0 <= c["conflict_pct"] <= c.get("duplicate_pct", 0.0) + 1e-9
    write_report([rep], "tiny", tmp_path / "off.md", tmp_path)
    assert (tmp_path / "offline_regret_cdf.png").exists()
    assert "Self-consistency" in (tmp_path / "off.md").read_text()


def test_label_conflicts_detects_contradictory_duplicates(fixture_decisions: Any) -> None:
    from elevator_mas.learning.lift.eval_offline import label_conflicts

    d = fixture_decisions
    nt = np.flatnonzero((d.winner >= 0) & (d.valid.sum(axis=1) >= 2))[:2]
    dup = d.take(np.array([nt[0], nt[0], nt[1]]))
    dup.winner[1] = next(int(c) for c in np.flatnonzero(dup.valid[1]) if c != dup.winner[0])
    c = label_conflicts(dup)
    assert c["decisions_in_groups"] == 2
    assert c["conflict_pct"] == pytest.approx(100 / 3)


# --------------------------------------------------------------------------- CLI


def test_lift_cli_lists_every_command() -> None:
    res = CliRunner().invoke(app, ["lift", "--help"])
    assert res.exit_code == 0
    for cmd in ("train-bc", "eval-offline", "export", "bench-infer", "eval-sim", "dagger", "card"):
        assert cmd in res.output


def test_card_command(shipped_model: Path, tmp_path: Path) -> None:
    md = tmp_path / "card.md"
    res = CliRunner().invoke(
        app, ["lift", "card", "--model", str(shipped_model), "--markdown", str(md)]
    )
    assert res.exit_code == 0, res.output
    assert "compatible with this code" in res.output
    card = json.loads(res.output[: res.output.index("\nstatus:")])
    assert card["param_count"] > 0
    assert md.read_text().startswith("# Model card")
