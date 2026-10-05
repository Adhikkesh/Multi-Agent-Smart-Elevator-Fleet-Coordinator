"""Generate the viva / slide figures (300 DPI PNG, white background) from real results.

Every figure is computed from files produced by the evaluation pipeline — nothing is
hard-coded or simulated here:

* reports/lift/closed_loop_test_runs.csv   — 7,800 real-simulator runs (100 test seeds)
* reports/lift/selection/val_r{0,1,2}.csv   — DAgger rounds, closed loop on val seeds
* docs/lift/runs/bc_default_s*/metrics.csv  — behaviour-cloning training curves
* docs/lift/offline_eval.md, models/liftzero_bc_v1.json, reports/lift/latency.json
* runs/ppo_*/metrics.csv, runs/ppo_*/evals.csv — PPO curves (only if Phase 5 was trained)

Figures whose data does not exist yet (PPO before Kaggle training, MCTS) are skipped with a
message, never invented.

    uv run python scripts/generate_viva_charts.py        # -> reports/viva/*.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "viva"
DPI = 300

# Validated categorical palette, fixed order (colour follows the strategy, never its rank).
STRATEGIES = ["nearest_car", "collective", "cnp_astar", "full", "liftzero_bc", "liftzero_bc_cnp"]
LABELS = {
    "nearest_car": "Nearest car",
    "collective": "Collective",
    "cnp_astar": "CNP + A*",
    "full": "Full (CNP+A*+SA)",
    "liftzero_bc": "LiftZero-BC",
    "liftzero_bc_cnp": "LiftZero-BC (bare CNP)",
    "liftzero_ppo": "LiftZero-PPO",
}
COLORS = {
    "nearest_car": "#2a78d6",
    "collective": "#eb6834",
    "cnp_astar": "#1baf7a",
    "full": "#eda100",
    "liftzero_bc": "#e87ba4",
    "liftzero_bc_cnp": "#008300",
    "liftzero_ppo": "#4a3aa7",
}
REGIMES = ["up_peak", "down_peak", "two_way", "interfloor"]
REGIME_LABEL = {
    "up_peak": "Morning up-peak",
    "down_peak": "Evening down-peak",
    "two_way": "Lunch two-way",
    "interfloor": "Inter-floor",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#b9b8b3",
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote reports/viva/{name}")


def ci95(x: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def grouped_bars(df: pd.DataFrame, metric: str, ylabel: str, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.13
    x = np.arange(len(REGIMES))
    for k, s in enumerate(STRATEGIES):
        means, lo, hi = [], [], []
        for r in REGIMES:
            v = df[(df.strategy == s) & (df.regime == r)][metric].to_numpy(float)
            m = v.mean()
            a, b = ci95(v)
            means.append(m)
            lo.append(m - a)
            hi.append(b - m)
        pos = x + (k - (len(STRATEGIES) - 1) / 2) * (width + 0.012)
        ax.bar(pos, means, width, color=COLORS[s], label=LABELS[s], zorder=2)
        ax.errorbar(pos, means, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(x, [REGIME_LABEL[r] for r in REGIMES])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper right")
    ax.text(
        0,
        -0.14,
        "Real Mesa simulator, 15 floors × 4 cars, 100 paired test seeds per regime; "
        "error bars = 95 % bootstrap CI.",
        transform=ax.transAxes,
        color=INK2,
        fontsize=8,
    )
    save(fig, name)


def fig_p95(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=False)
    for ax, r in zip(axes, REGIMES, strict=True):
        data = [
            df[(df.strategy == s) & (df.regime == r)]["p95_wait"].to_numpy() for s in STRATEGIES
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, s in zip(bp["boxes"], STRATEGIES, strict=True):
            patch.set_facecolor(COLORS[s])
            patch.set_edgecolor("white")
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, 7), ["NC", "COL", "CNP", "FULL", "BC", "BC-c"], fontsize=8)
        ax.set_title(REGIME_LABEL[r], fontsize=10, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("95th-percentile wait per run (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in STRATEGIES]
    fig.legend(
        handles,
        [LABELS[s] for s in STRATEGIES],
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.suptitle(
        "Service guarantee: distribution of p95 wait across 100 test seeds",
        x=0.01,
        ha="left",
        y=1.13,
        fontweight="bold",
    )
    save(fig, "fig2_p95_service_guarantee.png")


def fig_learned_vs_teacher(df: pd.DataFrame) -> None:
    pairs = [("liftzero_bc", "full"), ("liftzero_bc_cnp", "cnp_astar")]
    regimes = sorted(df.regime.unique(), key=lambda r: (r not in REGIMES, r))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(regimes))
    for k, (lrn, tch) in enumerate(pairs):
        mid, lo, hi = [], [], []
        for r in regimes:
            a = df[(df.strategy == lrn) & (df.regime == r)].set_index("seed")["avg_wait"]
            b = df[(df.strategy == tch) & (df.regime == r)].set_index("seed")["avg_wait"]
            a, b = a.loc[a.index.intersection(b.index)].to_numpy(), b.loc[a.index].to_numpy()
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(a), size=(2000, len(a)))
            boots = (a[idx].mean(1) / b[idx].mean(1) - 1) * 100
            mid.append((a.mean() / b.mean() - 1) * 100)
            lo.append(np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5))
        off = (k - 0.5) * 0.3
        ax.errorbar(
            mid,
            y + off,
            xerr=[np.subtract(mid, lo), np.subtract(hi, mid)],
            fmt="o",
            color=COLORS[lrn],
            ms=6,
            capsize=2,
            label=f"{LABELS[lrn]} vs {LABELS[tch]}",
        )
    ax.axvspan(-5, 5, color="#f1f0ec", zorder=0, label="±5 % target band")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, regimes)
    ax.invert_yaxis()
    ax.set_xlabel("Δ average wait vs the A* teacher (%)  — negative = learned is better")
    ax.set_title(
        "Learned bidder vs its teacher, every regime and scenario (100 test seeds)", loc="left"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_learned_vs_teacher.png")


def fig_bc_training() -> None:
    runs = sorted((ROOT / "docs" / "lift" / "runs").glob("bc_default_s*/metrics.csv"))
    if not runs:
        print("  skip fig5 (no BC runs)")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    seq = ["#2a78d6", "#eb6834", "#1baf7a"]
    for c, p in zip(seq, runs, strict=False):
        m = pd.read_csv(p)
        seed = p.parent.name.split("_s")[-1]
        a1.plot(m.epoch, m.train_loss, color=c, lw=2, label=f"seed {seed}")
        a2.plot(m.epoch, m.val_agree_nontrivial_tie, color=c, lw=2, label=f"seed {seed}")
    a2.axhline(85, color=INK2, ls="--", lw=1)
    a2.text(0.2, 85.3, "85 % acceptance gate", color=INK2, fontsize=8)
    a1.set(xlabel="epoch", ylabel="imitation loss", title="Behaviour cloning: training loss")
    a2.set(
        xlabel="epoch",
        ylabel="validation agreement with teacher (%, tie-aware)",
        title="Validation agreement (3 seeds)",
    )
    a1.legend()
    a2.legend(loc="lower right")
    for a in (a1, a2):
        a.title.set_ha("left")
        a.set_title(a.get_title(), loc="left")
    save(fig, "fig5_imitation_learning_curves.png")


def fig_dagger() -> None:
    files = [ROOT / "reports" / "lift" / "selection" / f"val_r{k}.csv" for k in range(3)]
    if not all(f.exists() for f in files):
        print("  skip fig6 (no DAgger validation runs)")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    markers = {"liftzero_bc": "o", "liftzero_bc_cnp": "s"}
    teacher = {"liftzero_bc": "full", "liftzero_bc_cnp": "cnp_astar"}
    for lrn in ("liftzero_bc", "liftzero_bc_cnp"):
        for j, r in enumerate(REGIMES):
            vals = []
            for f in files:
                d = pd.read_csv(f)
                a = d[(d.strategy == lrn) & (d.regime == r)].avg_wait.mean()
                b = d[(d.strategy == teacher[lrn]) & (d.regime == r)].avg_wait.mean()
                vals.append((a / b - 1) * 100)
            x = np.arange(3) + (j - 1.5) * 0.06 + (0.02 if lrn.endswith("cnp") else 0)
            ax.plot(x, vals, marker=markers[lrn], color=COLORS[lrn], alpha=0.35 + 0.15 * j, lw=1.5)
    ax.axhspan(-5, 5, color="#f1f0ec", zorder=0)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(3), ["BC (round 0)", "DAgger round 1", "DAgger round 2 (shipped)"])
    ax.set_ylabel("Δ avg wait vs teacher (%)")
    ax.set_title(
        "DAgger: closed-loop gap to the teacher shrinks (50 val seeds × 4 regimes)", loc="left"
    )
    h = [plt.Line2D([], [], color=COLORS[k], marker=markers[k], label=LABELS[k]) for k in markers]
    ax.legend(handles=h, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_dagger_rounds.png")


def fig_offline() -> None:
    md = ROOT / "docs" / "lift" / "offline_eval.md"
    if not md.exists():
        print("  skip fig7 (no offline eval)")
        return
    text = md.read_text()
    rows = {}
    for split in ("val", "test", "test_large"):
        sec = text.split(f"## Split `{split}`")[1].split("###")[0]
        for line in sec.splitlines():
            m = re.match(
                r"\| (LiftZero|nearest|lowest_eta|cost_greedy) \| \d+ \| ([\d.]+) \| ([\d.]+)", line
            )
            if m:
                rows.setdefault(m.group(1), {})[split] = float(m.group(3))
    names = {
        "nearest": "Nearest car",
        "lowest_eta": "Lowest ETA",
        "cost_greedy": "Cost-greedy",
        "LiftZero": "LiftZero-BC",
    }
    cols = {
        "nearest": "#2a78d6",
        "lowest_eta": "#eb6834",
        "cost_greedy": "#1baf7a",
        "LiftZero": "#e87ba4",
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    splits = ["val", "test", "test_large"]
    x = np.arange(3)
    for k, key in enumerate(["nearest", "lowest_eta", "cost_greedy", "LiftZero"]):
        v = [rows[key][s] for s in splits]
        b = ax.bar(x + (k - 1.5) * 0.2, v, 0.18, color=cols[key], label=names[key], zorder=2)
        if key == "LiftZero":
            ax.bar_label(b, fmt="%.1f", fontsize=8, color=INK)
    ax.axhline(85, color=INK2, ls="--", lw=1)
    ax.set_xticks(x, ["Validation", "Test", "Test-large (33–40 floors, unseen)"])
    ax.set_ylabel("agreement with A* teacher (%, tie-aware)")
    ax.set_ylim(0, 100)
    ax.set_title("Offline decision agreement: LiftZero vs classical scoring rules", loc="left")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.grid(axis="x", visible=False)
    save(fig, "fig7_offline_agreement.png")


def fig_latency() -> None:
    p = ROOT / "reports" / "lift" / "latency.json"
    if not p.exists():
        print("  skip fig8 (no latency benchmark)")
        return
    d = json.loads(p.read_text())
    rows = d["rows"]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(n, [r["median_ms"] for r in rows], marker="o", color="#2a78d6", lw=2, label="median")
    ax.plot(n, [r["p99_ms"] for r in rows], marker="o", color="#eb6834", lw=2, label="p99")
    ax.axhline(2.0, color=INK2, ls="--", lw=1)
    ax.text(2, 2.05, "2 ms budget (N = 8)", color=INK2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [str(k) for k in n])
    ax.set_ylim(0, 2.4)
    ax.set_xlabel("cars in the fleet (N)")
    ax.set_ylabel("ms per decision")
    ax.set_title(f"ONNX inference latency on CPU ({d['cpu']}, 1 thread)", loc="left")
    ax.legend()
    save(fig, "fig8_inference_latency.png")


def fig_ppo() -> None:
    runs = sorted(ROOT.glob("runs/ppo_default_s*/metrics.csv"))
    if not runs:
        print("  skip fig_ppo (train PPO on Kaggle first — docs/lift/KAGGLE_TRAINING.md)")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    seq = ["#2a78d6", "#eb6834", "#1baf7a"]
    for c, p in zip(seq, runs, strict=False):
        m = pd.read_csv(p)
        a1.plot(m.decisions / 1e6, m.ep_return, color=c, lw=1.5, label=p.parent.name)
        a2.plot(m.decisions / 1e6, m.kl_anchor, color=c, lw=1.5)
    a1.set(xlabel="decisions (millions)", ylabel="mean episode return", title="PPO return")
    a2.set(xlabel="decisions (millions)", ylabel="KL(π_BC ‖ π)", title="KL to the imitation policy")
    a1.legend(fontsize=7)
    save(fig, "fig9_ppo_learning_curves.png")


def fig_mcts() -> None:
    p = ROOT / "reports" / "lift" / "mcts_closed_loop.csv"
    if not p.exists():
        print("  skip fig10 (run the look-ahead evaluation first)")
        return
    df = pd.read_csv(p)
    strategies = [s for s in ("full", "liftzero_bc", "liftzero_bc_mcts") if s in set(df.strategy)]
    colors = {
        "full": COLORS["full"],
        "liftzero_bc": COLORS["liftzero_bc"],
        "liftzero_bc_mcts": "#4a3aa7",
    }
    labels = {**LABELS, "liftzero_bc_mcts": "LiftZero-BC + look-ahead"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    x = np.arange(len(REGIMES))
    w = 0.26
    for k, st in enumerate(strategies):
        m, lo, hi = [], [], []
        for r in REGIMES:
            v = df[(df.strategy == st) & (df.regime == r)].avg_wait.to_numpy(float)
            a, b = ci95(v)
            m.append(v.mean())
            lo.append(v.mean() - a)
            hi.append(b - v.mean())
        pos = x + (k - (len(strategies) - 1) / 2) * (w + 0.02)
        a1.bar(pos, m, w, color=colors[st], label=labels[st], zorder=2)
        a1.errorbar(pos, m, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    a1.set_xticks(x, [REGIME_LABEL[r] for r in REGIMES], fontsize=9)
    a1.set_ylabel("average wait (s)")
    a1.set_title("Look-ahead vs reflex learned bidding (20 test seeds)", loc="left")
    a1.legend()
    a1.grid(axis="x", visible=False)
    ms = [df[df.strategy == st].compute_ms_per_tick.mean() for st in strategies]
    short = {"full": "Full", "liftzero_bc": "BC", "liftzero_bc_mcts": "BC + look-ahead"}
    a2.barh([short[s] for s in strategies], ms, color=[colors[s] for s in strategies], zorder=2)
    fig.subplots_adjust(wspace=0.35)
    a2.set_xlabel("compute per simulated second (ms)")
    a2.set_title("Cost of deliberation", loc="left")
    a2.grid(axis="y", visible=False)
    save(fig, "fig10_mcts_lookahead.png")


def fig_classic_failure(df: pd.DataFrame) -> None:
    """Morning up-peak: the four classical strategies (slide 3)."""
    order = ["nearest_car", "collective", "cnp_astar", "full"]
    names = ["Nearest car", "Collective (LOOK)", "Contract Net + A*", "Full agent system"]
    vals = [df[(df.strategy == s) & (df.regime == "up_peak")].avg_wait.mean() for s in order]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.barh(names[::-1], vals[::-1], color=[COLORS[s] for s in order][::-1], zorder=2)
    ax.bar_label(bars, fmt="%.1f s", padding=4, color=INK, fontsize=11)
    ax.set_xlabel("average wait (s)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("Morning up-peak: average wait, 100 test seeds", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=11)
    save(fig, "fig11_classic_failure.png")


def fig_search_comparison() -> None:
    """Nodes expanded and optimality of BFS / UCS / Greedy / A* (slide 9)."""
    p = ROOT / "reports" / "search_benchmark_summary.csv"
    if not p.exists():
        print("  skip fig12 (run scripts/search_benchmark.py first)")
        return
    d = pd.read_csv(p).set_index("algorithm").loc[["bfs", "ucs", "greedy", "astar"]]
    names = ["BFS", "UCS", "Greedy", "A*"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bars = ax.bar(names, d.nodes_mean, color=cols, zorder=2, width=0.6)
    for b, opt in zip(bars, d.optimal_pct, strict=True):
        ax.annotate(
            f"{b.get_height():.1f} nodes\n{opt:.0f} % optimal",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
            xytext=(0, 4),
            textcoords="offset points",
        )
    ax.set_ylabel("mean nodes expanded")
    ax.set_ylim(0, d.nodes_mean.max() * 1.3)
    ax.set_title("200 random 7-stop routing problems", loc="left")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=12)
    save(fig, "fig12_search_comparison.png")


def main() -> None:
    style()
    print("Generating viva charts ->", OUT)
    runs = ROOT / "reports" / "lift" / "closed_loop_test_runs.csv"
    if runs.exists():
        df = pd.read_csv(runs)
        grouped_bars(
            df,
            "avg_wait",
            "average wait (s)",
            "Average passenger wait by strategy and traffic pattern",
            "fig1_wait_time_comparison.png",
        )
        fig_p95(df)
        grouped_bars(
            df,
            "energy",
            "energy units per 15-minute run (floors + stops + reversals)",
            "Energy use of the fleet",
            "fig3_energy_efficiency.png",
        )
        fig_learned_vs_teacher(df)
        fig_classic_failure(df)
    else:
        print("  skip fig1-4 (run `elevator lift eval-sim` first)")
    fig_bc_training()
    fig_dagger()
    fig_offline()
    fig_latency()
    fig_ppo()
    fig_mcts()
    fig_search_comparison()


if __name__ == "__main__":
    main()
