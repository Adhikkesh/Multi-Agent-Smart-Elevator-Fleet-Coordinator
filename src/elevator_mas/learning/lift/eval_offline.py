"""Offline evaluation of a LiftZero checkpoint on recorded teacher decisions.

For each split (val, test, test_large), teacher (cnp_astar, full) and traffic pattern it
reports top-1 agreement (strict and tie-aware; overall, non-trivial and hard decisions),
regret, Kendall's tau, calibration, and the same agreement for the Phase 3 baselines on the
same decisions. It then clusters the mistakes into failure modes, renders concrete mistaken
decisions as text, and estimates how much of the residual error the features themselves
explain (decisions whose features are identical but whose teacher labels differ).

Outputs ``docs/lift/offline_eval.md`` and ``reports/lift/offline_*.png``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from elevator_mas.learning.lift.config import ROOT
from elevator_mas.learning.lift.metrics import (
    TIE_EPS,
    argmin_choice,
    decision_metrics,
    hard,
    nontrivial,
)
from elevator_mas.learning.lift.train_bc import Decisions, load_checkpoint, load_decisions, predict
from elevator_mas.learning.schema import FEATURE_NAMES, PATTERNS, TEACHERS

_CAR = {n: i for i, n in enumerate(FEATURE_NAMES["cars"])}
_CALL = {n: i for i, n in enumerate(FEATURE_NAMES["call"])}
_GLOB = {n: i for i, n in enumerate(FEATURE_NAMES["glob"])}
_DOORS = ("closed", "opening", "open", "closing")


def _pattern(d: Decisions) -> np.ndarray:
    oh = d.call[:, _CALL["pattern_up_peak"] : _CALL["pattern_up_peak"] + len(PATTERNS)]
    return np.where(oh.max(axis=1) > 0, oh.argmax(axis=1), -1)


def baselines(d: Decisions) -> dict[str, np.ndarray]:
    """Decisions of the Phase 3 baselines on the same states."""
    from elevator_mas.learning.twin.policies import CostGreedyPolicy

    v = d.valid
    greedy = CostGreedyPolicy().act({"cars": d.cars, "eligible": v})
    greedy = np.where(v.any(axis=1), greedy, -1)
    return {
        "nearest": argmin_choice(d.cars[:, :, _CAR["dist_to_call"]], v),
        "lowest_eta": argmin_choice(d.cars[:, :, _CAR["plan_end_eta"]], v),
        "cost_greedy": greedy,
    }


def calibration(score: np.ndarray, d: Decisions, n_bins: int = 10) -> list[dict[str, float]]:
    """MAE of expm1(score) vs teacher cost, bucketed by teacher-cost decile (valid cars)."""
    v = d.valid
    cost = d.cost[v]
    pred = np.expm1(np.clip(score[v], 0.0, 12.0))
    if len(cost) == 0:
        return []
    edges = np.unique(np.quantile(cost, np.linspace(0, 1, n_bins + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (cost >= lo) & (cost <= hi)
        if sel.any():
            out.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "mean_cost": float(cost[sel].mean()),
                    "mean_pred": float(pred[sel].mean()),
                    "mae": float(np.abs(pred[sel] - cost[sel]).mean()),
                    "n": int(sel.sum()),
                }
            )
    return out


def label_conflicts(d: Decisions, decimals: int = 3) -> dict[str, float]:
    """Decisions whose rounded features coincide but whose teacher winners differ.

    Such pairs are indistinguishable to *any* model of these features, so their conflict
    rate is an empirical floor on the achievable error (a self-consistency bound).
    """
    nt = np.flatnonzero(nontrivial(d.valid, d.winner))
    if len(nt) == 0:
        return {"groups": 0, "decisions_in_groups": 0, "conflict_pct": 0.0}
    feats = np.concatenate([d.call[nt], d.glob[nt], d.cars[nt].reshape(len(nt), -1)], axis=1).round(
        decimals
    )
    _, inv, counts = np.unique(feats, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    multi = counts[inv] > 1
    conflicts = 0
    in_groups = int(multi.sum())
    winners = d.winner[nt]
    for g in np.unique(inv[multi]):
        w = winners[inv == g]
        conflicts += len(w) - Counter(w.tolist()).most_common(1)[0][1]
    return {
        "groups": int(len(np.unique(inv[multi]))),
        "decisions_in_groups": in_groups,
        "duplicate_pct": float(in_groups / len(nt) * 100.0),
        "conflict_pct": float(conflicts / len(nt) * 100.0),
    }


def failure_modes(d: Decisions, choice: np.ndarray, top: int = 10) -> list[tuple[str, int]]:
    """Cluster wrong (positive-regret) decisions by situation; most frequent first."""
    rows = np.flatnonzero(nontrivial(d.valid, d.winner))
    pat = _pattern(d)
    keys: list[str] = []
    for i in rows:
        w, c = int(d.winner[i]), int(choice[i])
        if c < 0 or d.cost[i, c] <= d.cost[i, w] + TIE_EPS:
            continue
        load = d.cars[i, w, _CAR["load"]]
        load_s = "empty" if load < 0.05 else ("light" if load < 0.5 else "loaded")
        door = _DOORS[int(np.argmax(d.cars[i, w, _CAR["door_closed"] : _CAR["door_closed"] + 4]))]
        w_route = bool(d.cars[i, w, _CAR["call_on_route"]] > 0.5)
        c_route = bool(d.cars[i, c, _CAR["call_on_route"]] > 0.5)
        name = PATTERNS[pat[i]] if pat[i] >= 0 else "?"
        keys.append(
            f"{name} | teacher car {load_s}, doors {door}, "
            f"{'on' if w_route else 'off'}-route | chosen {'on' if c_route else 'off'}-route"
        )
    return Counter(keys).most_common(top)


def describe_car(d: Decisions, i: int, car: int) -> str:
    x = d.cars[i, car]
    floors = max(round(float(d.glob[i, _GLOB["floors_norm"]]) * 40.0), 2)
    fl = round(float(x[_CAR["floor"]]) * (floors - 1))
    end = round(float(x[_CAR["plan_end_floor"]]) * (floors - 1))
    arrow = "↑" if x[_CAR["dir_up"]] > 0.5 else ("↓" if x[_CAR["dir_down"]] > 0.5 else "idle")
    return (
        f"car {car} at floor {fl} {arrow} load {x[_CAR['load']] * 100:.0f}%, "
        f"{x[_CAR['plan_stops']] * 12:.0f} stops planned ending at {end}"
    )


def render_mistakes(d: Decisions, choice: np.ndarray, score: np.ndarray, k: int = 5) -> list[str]:
    """Up to ``k`` mistaken decisions (largest regret first) as readable text."""
    rows = np.flatnonzero(nontrivial(d.valid, d.winner) & (choice >= 0))
    reg = d.cost[rows, choice[rows]] - d.cost[rows, d.winner[rows]]
    order = rows[np.argsort(-reg)][:k]
    out = []
    for i in order:
        w, c = int(d.winner[i]), int(choice[i])
        if d.cost[i, c] <= d.cost[i, w] + TIE_EPS:
            continue
        floors = max(round(float(d.glob[i, _GLOB["floors_norm"]]) * 40.0), 2)
        cf = round(float(d.call[i, _CALL["call_floor"]]) * (floors - 1))
        cd = "↑" if d.call[i, _CALL["call_dir"]] > 0 else "↓"
        out.append(
            f"call {cf}{cd} ({floors} floors): teacher picks {describe_car(d, i, w)} "
            f"(cost {d.cost[i, w]:.1f}); LiftZero picks {describe_car(d, i, c)} "
            f"(teacher cost {d.cost[i, c]:.1f}, net {np.expm1(score[i, c]):.1f})"
        )
    return out


@dataclass
class SplitReport:
    split: str
    overall: dict[str, Any]
    by_teacher: dict[str, dict[str, Any]]
    by_pattern: dict[str, dict[str, Any]]
    by_fleet: dict[int, dict[str, Any]]
    baselines: dict[str, dict[str, Any]]
    calibration: list[dict[str, float]]
    failure_modes: list[tuple[str, int]]
    mistakes: list[str]
    conflicts: dict[str, float]
    regret: np.ndarray


def evaluate_split(model: Any, d: Decisions, split: str) -> SplitReport:
    pred = predict(model, d)
    score, choice = pred["score"], pred["choice"]

    def m(sel: np.ndarray) -> dict[str, Any]:
        return decision_metrics(
            d.cost[sel], d.valid[sel], d.winner[sel], choice[sel], score[sel]
        ).as_dict()

    everything = np.ones(len(d), dtype=bool)
    teacher = d.extra.get("teacher", np.full(len(d), -1))
    pat = _pattern(d)
    fleet = d.mask.sum(axis=1)
    base = {
        name: decision_metrics(d.cost, d.valid, d.winner, ch).as_dict()
        for name, ch in baselines(d).items()
    }
    nt = nontrivial(d.valid, d.winner)
    rows = np.arange(len(d))
    reg = np.where(
        nt & (choice >= 0),
        d.cost[rows, np.maximum(choice, 0)] - d.cost[rows, np.maximum(d.winner, 0)],
        np.nan,
    )
    return SplitReport(
        split=split,
        overall=m(everything),
        by_teacher={t: m(teacher == k) for k, t in enumerate(TEACHERS) if (teacher == k).any()},
        by_pattern={p: m(pat == k) for k, p in enumerate(PATTERNS) if (pat == k).any()},
        by_fleet={int(n): m(fleet == n) for n in np.unique(fleet)},
        baselines=base,
        calibration=calibration(score, d),
        failure_modes=failure_modes(d, choice),
        mistakes=render_mistakes(d, choice, score),
        conflicts={
            **label_conflicts(d),
            **{f"coarse_{k}": v for k, v in label_conflicts(d, decimals=1).items()},
        },
        regret=reg[~np.isnan(reg)],
    )


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "—" if np.isnan(x) else f"{x:.2f}"
    return str(x)


_COLS = (
    ("n_nontrivial", "non-trivial"),
    ("agree_nontrivial", "agree %"),
    ("agree_nontrivial_tie", "agree % (tie-aware)"),
    ("agree_hard", "hard %"),
    ("agree_hard_tie", "hard % (tie-aware)"),
    ("regret_mean", "regret mean"),
    ("regret_p95", "regret p95"),
    ("rel_regret_mean", "rel. regret"),
    ("kendall_tau", "Kendall τ"),
)


def _table(rows: dict[str, dict[str, Any]], label: str) -> list[str]:
    head = f"| {label} | " + " | ".join(c[1] for c in _COLS) + " |"
    sep = "| --- " * (len(_COLS) + 1) + "|"
    body = [
        f"| {k} | " + " | ".join(_fmt(v.get(c[0], float("nan"))) for c in _COLS) + " |"
        for k, v in rows.items()
    ]
    return [head, sep, *body, ""]


def write_report(
    reports: list[SplitReport], ckpt: str, out_md: Path, out_dir: Path, extra: str = ""
) -> None:
    """Markdown tables + PNG figures for a list of split reports."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LiftZero — offline evaluation",
        "",
        f"Checkpoint: `{ckpt}`. Generated by `elevator lift eval-offline`.",
        "",
        "*Agreement* = the network's argmin car equals the teacher's award; *tie-aware* also "
        "accepts a car whose teacher bid equals the winner's (zero regret — the dispatcher "
        "breaks such ties by car id, which an equivariant network cannot see). *Hard* = "
        "runner-up within 10 % of the best bid. Regret is in teacher cost units.",
        "",
    ]
    if extra:
        lines += [extra, ""]
    for r in reports:
        lines += [f"## Split `{r.split}`", ""]
        lines += _table({"LiftZero": r.overall, **r.baselines}, "policy")
        lines += ["### By teacher", "", *_table(r.by_teacher, "teacher")]
        lines += ["### By traffic pattern", "", *_table(r.by_pattern, "pattern")]
        lines += [
            "### By fleet size",
            "",
            *_table({str(k): v for k, v in r.by_fleet.items()}, "cars"),
        ]
        c = r.conflicts
        lines += [
            "### Self-consistency bound",
            "",
            f"{c.get('decisions_in_groups', 0):,} non-trivial decisions "
            f"({c.get('duplicate_pct', 0.0):.2f} %) share their exact (3-decimal) feature "
            f"vector with another decision; labels conflict within those groups for "
            f"{c.get('conflict_pct', 0.0):.3f} % of all non-trivial decisions — error no "
            "model of these features can remove.",
            "",
            f"Coarser (1-decimal) near-duplicates: {c.get('coarse_decisions_in_groups', 0):,} "
            f"decisions ({c.get('coarse_duplicate_pct', 0.0):.1f} %) fall in shared cells; "
            f"labels conflict for {c.get('coarse_conflict_pct', 0.0):.2f} % of all non-trivial "
            "decisions. This is only indicative (a cell mixes genuinely different states), but "
            "it shows how much of the residual error is ambiguity in the public features.",
            "",
            "### Top failure modes (positive-regret decisions)",
            "",
            "| situation | count |",
            "| --- | --- |",
            *[f"| {k} | {n} |" for k, n in r.failure_modes],
            "",
            "### Largest mistakes",
            "",
            *[f"{i + 1}. {t}" for i, t in enumerate(r.mistakes)],
            "",
            "### Calibration (teacher-cost deciles)",
            "",
            "| bucket | mean teacher cost | mean predicted | MAE | n |",
            "| --- | --- | --- | --- | --- |",
            *[
                f"| {b['lo']:.1f}–{b['hi']:.1f} | {b['mean_cost']:.2f} | {b['mean_pred']:.2f} "
                f"| {b['mae']:.2f} | {b['n']:,} |"
                for b in r.calibration
            ],
            "",
        ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    _plots(reports, out_dir)


def _plots(reports: list[SplitReport], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for r in reports:
        ns = sorted(r.by_fleet)
        ax.plot(ns, [r.by_fleet[n]["agree_nontrivial_tie"] for n in ns], marker="o", label=r.split)
    ax.axhline(85, color="grey", ls="--", lw=1)
    ax.set_xlabel("cars in the building")
    ax.set_ylabel("tie-aware agreement (%)")
    ax.set_title("Agreement with the A* teacher vs fleet size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "offline_agreement_vs_fleet.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for r in reports:
        x = np.sort(r.regret)
        if len(x):
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=r.split)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("regret (teacher cost units)")
    ax.set_ylabel("fraction of non-trivial decisions")
    ax.set_title("Regret CDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "offline_regret_cdf.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    hi = 1.0
    for r in reports:
        xs = [b["mean_cost"] for b in r.calibration]
        ys = [b["mean_pred"] for b in r.calibration]
        hi = max([hi, *xs, *ys])
        ax.plot(xs, ys, marker="o", label=r.split)
    ax.plot([0, hi], [0, hi], color="grey", ls="--", lw=1, label="ideal")
    ax.set_xlabel("mean teacher cost (decile)")
    ax.set_ylabel("mean predicted cost expm1(score)")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "offline_calibration.png", dpi=150)
    plt.close(fig)


def run_offline_eval(
    ckpt: Path | str,
    splits: tuple[str, ...] = ("val", "test", "test_large"),
    data: dict[str, list[str]] | None = None,
    limit: int | None = 100_000,
    out_md: Path | None = None,
    out_dir: Path | None = None,
    log: Any = print,
) -> list[SplitReport]:
    """Evaluate ``ckpt`` on the given splits and write the report."""
    model, _ = load_checkpoint(ckpt)
    data = data or {
        "val": ["data/expert/val"],
        "test": ["data/expert/test"],
        "test_large": ["data/expert/test"],
    }
    reports = []
    for split in splits:
        try:
            d = load_decisions(data[split], split, limit, seed=0)
        except (FileNotFoundError, ValueError) as exc:
            log(f"[eval-offline] skip {split}: {exc}")
            continue
        r = evaluate_split(model, d, split)
        o = r.overall
        log(
            f"[eval-offline] {split}: n={o['n_nontrivial']:,} agree={o['agree_nontrivial']:.2f}% "
            f"tie-aware={o['agree_nontrivial_tie']:.2f}% hard={o['agree_hard']:.2f}% "
            f"regret={o['regret_mean']:.3f} tau={o['kendall_tau']:.3f}"
        )
        reports.append(r)
    write_report(
        reports,
        str(ckpt),
        out_md or ROOT / "docs" / "lift" / "offline_eval.md",
        out_dir or ROOT / "reports" / "lift",
    )
    return reports


__all__ = [
    "SplitReport",
    "baselines",
    "calibration",
    "evaluate_split",
    "failure_modes",
    "hard",
    "label_conflicts",
    "render_mistakes",
    "run_offline_eval",
    "write_report",
]
