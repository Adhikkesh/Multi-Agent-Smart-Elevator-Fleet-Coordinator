"""Closed-loop evaluation of learned strategies in the *real* Mesa simulator.

Offline agreement measures single decisions against teacher labels; closed loop measures
what passengers experience when the learned bidder drives every auction and its mistakes
compound. Every (strategy, regime, seed) run is the full multi-agent simulation with the
safety invariants checked every tick. Seeds are *paired*: the same seed gives every strategy
the same passengers, so differences are compared run by run.

Regimes: the four Phase 3 evaluation regimes (``up_peak``, ``down_peak``, ``two_way``,
``interfloor``; 15 floors x 4 cars, 900 ticks) and the nine scenario YAMLs.
Seeds: ``val`` = 8000-8049, ``test`` = 8500-8599 (never used for training).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elevator_mas.comms.board import DecisionEvent
from elevator_mas.config import (
    ArrivalPhase,
    BuildingConfig,
    LiftConfig,
    ScenarioConfig,
    TrafficConfig,
    available_scenarios,
)
from elevator_mas.learning.lift.config import ROOT
from elevator_mas.learning.recorder import label_winner
from elevator_mas.learning.regimes import load_regimes_config

REGIMES = ("up_peak", "down_peak", "two_way", "interfloor")
DEFAULT_STRATEGIES = (
    "nearest_car",
    "collective",
    "cnp_astar",
    "full",
    "liftzero_bc",
    "liftzero_bc_cnp",
)
METRICS = (
    "avg_wait",
    "p95_wait",
    "max_wait",
    "long_wait_pct",
    "throughput",
    "energy",
    "compute_ms_per_tick",
)


def seeds_for(which: str, n: int | None = None) -> list[int]:
    """Closed-loop seed list: ``val`` / ``test`` (from regimes.yaml), first ``n``."""
    cfg = load_regimes_config()
    seeds = {"val": cfg.val_seeds, "test": cfg.test_seeds}[which]
    return seeds[:n] if n else seeds


def regime_names(which: str | Iterable[str]) -> list[str]:
    """``all`` = four regimes + nine scenarios; ``regimes`` / ``scenarios``; or a list."""
    if isinstance(which, str):
        if which == "all":
            return [*REGIMES, *available_scenarios()]
        if which == "regimes":
            return list(REGIMES)
        if which == "scenarios":
            return available_scenarios()
        return [w.strip() for w in which.split(",") if w.strip()]
    return list(which)


def build_config(
    regime: str, strategy: str, seed: int, lift: LiftConfig | None = None
) -> ScenarioConfig:
    """The scenario of one closed-loop run (a Phase 3 regime or a scenario YAML)."""
    lift = lift or LiftConfig()
    if regime in REGIMES:
        r = load_regimes_config().eval_regimes[regime]
        return ScenarioConfig(
            name=f"eval_{regime}",
            strategy=strategy,
            seed=seed,
            duration=r.duration,
            building=BuildingConfig(floors=r.floors, cars=r.cars, capacity=r.capacity),
            traffic=TrafficConfig(
                phases=[ArrivalPhase(until=r.duration, rate=r.rate, pattern=r.pattern)]  # type: ignore[arg-type]
            ),
            lift=lift,
        )
    cfg = ScenarioConfig.load(regime)
    return cfg.model_copy(update={"strategy": strategy, "seed": seed, "lift": lift})


class AgreementHook:
    """Counts learned awards that differ from the shadow teacher's choice."""

    def __init__(self) -> None:
        self.decisions = 0
        self.disagree = 0

    def __call__(self, event: DecisionEvent) -> None:
        if event.shadow_bids is None or event.winner is None:
            return
        self.decisions += 1
        if label_winner(event.shadow_bids) != event.winner:
            self.disagree += 1


def run_one(
    strategy: str,
    regime: str,
    seed: int,
    shadow: bool = False,
    model_path: str | None = None,
) -> dict[str, Any]:
    """One closed-loop run; safety invariants are checked every tick."""
    from elevator_mas.model import ElevatorModel
    from elevator_mas.strategies import get_strategy

    learned = get_strategy(strategy).uses_learned_bidder
    lift = LiftConfig(shadow_teacher=shadow and learned, model_path=model_path)
    cfg = build_config(regime, strategy, seed, lift)
    model = ElevatorModel(cfg)
    hook = AgreementHook()
    if lift.shadow_teacher:
        model.decision_hooks.append(hook)
    violations: list[str] = []
    t0 = time.perf_counter()
    for _ in range(cfg.duration):
        model.step()
        violations.extend(model.violations())
    wall = time.perf_counter() - t0
    m = model.latest_metrics
    return {
        "strategy": strategy,
        "regime": regime,
        "seed": seed,
        **{k: float(getattr(m, k)) for k in METRICS},
        "delivered": m.delivered,
        "arrived": m.arrived,
        "calls": m.calls,
        "violations": len(set(violations)),
        "shadow": lift.shadow_teacher,
        "disagree_pct": (100.0 * hook.disagree / hook.decisions) if hook.decisions else np.nan,
        "wall_s": wall,
    }


def _run_star(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_one(*args)


def run_matrix(
    strategies: Iterable[str],
    regimes: Iterable[str],
    seeds: Iterable[int],
    workers: int = 1,
    shadow: bool = False,
    model_path: str | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """All (strategy, regime, seed) runs, in parallel across processes."""
    jobs = [(s, r, sd, shadow, model_path) for s in strategies for r in regimes for sd in seeds]
    log(f"[eval-sim] {len(jobs)} runs on {workers} worker(s)")
    t0 = time.perf_counter()
    if workers <= 1:
        rows = [_run_star(j) for j in jobs]
    else:
        rows = []
        step = max(1, len(jobs) // 10)
        with mp.get_context("spawn").Pool(workers) as pool:
            # One job at a time: a slow run never holds a queue of others hostage.
            for i, row in enumerate(pool.imap_unordered(_run_star, jobs, chunksize=1), 1):
                rows.append(row)
                if i % step == 0:
                    log(f"[eval-sim] {i}/{len(jobs)} runs")
        order = {(j[0], j[1], j[2]): k for k, j in enumerate(jobs)}
        rows.sort(key=lambda r: order[(r["strategy"], r["regime"], r["seed"])])
    log(f"[eval-sim] done in {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame(rows)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile-bootstrap CI of the mean."""
    if len(x) < 2:
        v = float(x[0]) if len(x) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@dataclass(frozen=True)
class Comparison:
    """Paired relative difference of a learned strategy vs its teacher on one metric."""

    regime: str
    strategy: str
    teacher: str
    metric: str
    learned: float
    reference: float
    rel_pct: float
    ci_low: float
    ci_high: float
    n: int


def paired_comparisons(
    df: pd.DataFrame, pairs: dict[str, str], metrics: tuple[str, ...] = ("avg_wait", "p95_wait")
) -> list[Comparison]:
    """Relative difference (learned / teacher - 1) with a paired bootstrap CI per regime."""
    out: list[Comparison] = []
    for learned, teacher in pairs.items():
        for regime in df["regime"].unique():
            a = df[(df.strategy == learned) & (df.regime == regime)].set_index("seed")
            b = df[(df.strategy == teacher) & (df.regime == regime)].set_index("seed")
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            for metric in metrics:
                la, lb = a.loc[common, metric].to_numpy(), b.loc[common, metric].to_numpy()
                ref = float(lb.mean())
                rel = (float(la.mean()) / ref - 1.0) * 100.0 if ref else float("nan")
                rng = np.random.default_rng(0)
                idx = rng.integers(0, len(common), size=(2000, len(common)))
                boots = (la[idx].mean(axis=1) / np.maximum(lb[idx].mean(axis=1), 1e-9) - 1) * 100
                out.append(
                    Comparison(
                        regime=regime,
                        strategy=learned,
                        teacher=teacher,
                        metric=metric,
                        learned=float(la.mean()),
                        reference=ref,
                        rel_pct=rel,
                        ci_low=float(np.percentile(boots, 2.5)),
                        ci_high=float(np.percentile(boots, 97.5)),
                        n=len(common),
                    )
                )
    return out


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95 % CI of every metric per (regime, strategy)."""
    rows = []
    for (regime, strategy), g in df.groupby(["regime", "strategy"], sort=False):
        row: dict[str, Any] = {"regime": regime, "strategy": strategy, "n": len(g)}
        for k in METRICS:
            x = g[k].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x)
            row[k] = float(x.mean())
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        row["violations"] = int(g["violations"].sum())
        dis = g["disagree_pct"].dropna()
        row["disagree_pct"] = float(dis.mean()) if len(dis) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(
    df: pd.DataFrame,
    path: Path,
    title: str = "LiftZero — closed-loop evaluation",
    note: str = "",
    pairs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Markdown report: per-regime table + paired comparisons vs the teachers."""
    pairs = pairs or {"liftzero_bc": "full", "liftzero_bc_cnp": "cnp_astar"}
    summ = summary_table(df)
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += [
        "Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds "
        "with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow "
        "teacher's choice (only for runs with the shadow teacher on).",
        "",
    ]
    for regime in summ["regime"].unique():
        g = summ[summ.regime == regime]
        lines += [
            f"## {regime}",
            "",
            "| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h "
            "| energy | ms/tick | violations | disagree % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in g.iterrows():
            lines.append(
                f"| {r.strategy} | {r.avg_wait:.1f} [{r.avg_wait_lo:.1f}, {r.avg_wait_hi:.1f}] "
                f"| {r.p95_wait:.1f} [{r.p95_wait_lo:.1f}, {r.p95_wait_hi:.1f}] "
                f"| {r.max_wait:.0f} | {r.long_wait_pct:.1f} | {r.throughput:.0f} "
                f"| {r.energy:.0f} | {r.compute_ms_per_tick:.2f} | {r.violations} "
                f"| {'—' if np.isnan(r.disagree_pct) else f'{r.disagree_pct:.1f}'} |"
            )
        lines.append("")
    comps = paired_comparisons(df, pairs)
    if comps:
        lines += [
            "## Learned vs teacher (paired)",
            "",
            "Target: average wait within ±5 % of the teacher, p95 within ±10 %.",
            "",
            "| regime | learned | teacher | metric | learned | teacher | Δ % [95 % CI] | n | "
            "target |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in comps:
            tol = 5.0 if c.metric == "avg_wait" else 10.0
            ok = "met" if abs(c.rel_pct) <= tol or c.rel_pct < 0 else "missed"
            lines.append(
                f"| {c.regime} | {c.strategy} | {c.teacher} | {c.metric} | {c.learned:.1f} "
                f"| {c.reference:.1f} | {c.rel_pct:+.1f} [{c.ci_low:+.1f}, {c.ci_high:+.1f}] "
                f"| {c.n} | {ok} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return summ


def default_out() -> tuple[Path, Path]:
    return ROOT / "docs" / "lift" / "closed_loop.md", ROOT / "reports" / "lift"
