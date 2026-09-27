"""The benchmark harness: strategies x scenarios x seeds, headless.

This is where the case study's central claim is tested. Each strategy runs on the same
scenarios with the same seeds, so any difference is the coordination mechanism and not
the traffic. Results are reported as mean +/- standard deviation over seeds, because a
single seed proves nothing about a stochastic system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from elevator_mas.sim.runner import RunResult, run_scenario

#: The metrics the benchmark tables and charts report.
REPORTED_METRICS: tuple[str, ...] = (
    "avg_wait",
    "p95_wait",
    "max_wait",
    "long_wait_pct",
    "avg_ride",
    "avg_system",
    "throughput",
    "energy",
    "delivered",
    "messages_per_call",
    "nodes_expanded",
    "compute_ms_per_tick",
)

DEFAULT_SCENARIOS: tuple[str, ...] = (
    "morning_up_peak",
    "evening_down_peak",
    "lunch_two_way",
    "interfloor_light",
)

DEFAULT_STRATEGIES: tuple[str, ...] = ("nearest_car", "collective", "cnp_astar", "full")


@dataclass
class BenchmarkResult:
    """Every run of a benchmark, plus the aggregates derived from them."""

    runs: list[RunResult] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        """One row per run."""
        return pd.DataFrame([r.as_row() for r in self.runs])

    def aggregate(self) -> pd.DataFrame:
        """Mean and standard deviation per (scenario, strategy) over the seeds."""
        frame = self.frame()
        if frame.empty:
            return frame
        metrics = [m for m in REPORTED_METRICS if m in frame.columns]
        grouped = frame.groupby(["scenario", "strategy"], sort=False)[metrics]
        aggregated = grouped.agg(["mean", "std"]).reset_index()
        # Flatten the ("metric", "mean") MultiIndex to "metric_mean", while leaving the
        # group keys ("scenario", "") as plain "scenario" rather than "scenario_".
        aggregated.columns = [
            column
            if isinstance(column, str)
            else (column[0] if not column[1] else f"{column[0]}_{column[1]}")
            for column in aggregated.columns
        ]
        return aggregated.fillna(0.0)

    def comparison(self, baseline: str = "nearest_car", champion: str = "full") -> pd.DataFrame:
        """Champion versus baseline per scenario: the headline result.

        Reported as a percentage improvement so the table can be read at a glance, and
        with the sign convention that positive always means better.
        """
        frame = self.frame()
        if frame.empty:
            return frame
        rows: list[dict[str, Any]] = []
        for scenario, group in frame.groupby("scenario", sort=False):
            base = group[group["strategy"] == baseline]
            best = group[group["strategy"] == champion]
            if base.empty or best.empty:
                continue
            row: dict[str, Any] = {"scenario": scenario}
            for metric in ("avg_wait", "p95_wait", "long_wait_pct", "energy"):
                base_mean = float(base[metric].mean())
                best_mean = float(best[metric].mean())
                row[f"{metric}_{baseline}"] = round(base_mean, 2)
                row[f"{metric}_{champion}"] = round(best_mean, 2)
                improvement = 100.0 * (base_mean - best_mean) / base_mean if base_mean else 0.0
                row[f"{metric}_improvement_pct"] = round(improvement, 1)
            rows.append(row)
        return pd.DataFrame(rows)

    def wins(self, baseline: str = "nearest_car", champion: str = "full") -> dict[str, bool]:
        """Whether the champion beat the baseline per scenario, on wait and long waits."""
        comparison = self.comparison(baseline, champion)
        if comparison.empty:
            return {}
        return {
            str(row["scenario"]): bool(
                row["avg_wait_improvement_pct"] > 0 and row["long_wait_pct_improvement_pct"] > 0
            )
            for _, row in comparison.iterrows()
        }


def run_benchmark(
    scenarios: list[str] | tuple[str, ...] = DEFAULT_SCENARIOS,
    strategies: list[str] | tuple[str, ...] = DEFAULT_STRATEGIES,
    seeds: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5),
    ticks: int | None = 600,
    progress: bool = False,
) -> BenchmarkResult:
    """Run every (scenario, strategy, seed) combination headlessly."""
    result = BenchmarkResult()
    total = len(scenarios) * len(strategies) * len(seeds)
    done = 0
    for scenario in scenarios:
        for strategy in strategies:
            for seed in seeds:
                run = run_scenario(scenario, strategy=strategy, seed=seed, ticks=ticks)
                result.runs.append(run)
                done += 1
                if progress:
                    print(
                        f"  [{done:3d}/{total}] {scenario:<20} {strategy:<12} seed={seed} "
                        f"avg_wait={run.metrics.avg_wait:6.1f}s",
                        flush=True,
                    )
    return result


def write_reports(result: BenchmarkResult, out_dir: str | Path = "reports") -> list[Path]:
    """Write the benchmark CSVs and the PNG charts used in the slides."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    runs_csv = directory / "benchmark_runs.csv"
    result.frame().to_csv(runs_csv, index=False)
    written.append(runs_csv)

    summary_csv = directory / "benchmark_summary.csv"
    result.aggregate().to_csv(summary_csv, index=False)
    written.append(summary_csv)

    comparison_csv = directory / "benchmark_comparison.csv"
    result.comparison().to_csv(comparison_csv, index=False)
    written.append(comparison_csv)

    written.extend(_write_charts(result, directory))
    return written


def _write_charts(result: BenchmarkResult, directory: Path) -> list[Path]:
    """Render the benchmark charts.

    matplotlib is imported here rather than at module scope, and with the non-interactive
    Agg backend, so importing this module never tries to open a display — it has to work
    on a headless machine and inside the test suite.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = result.frame()
    if frame.empty:
        return []

    written: list[Path] = []
    charts = [
        ("avg_wait", "Average wait (s)", "benchmark_avg_wait.png"),
        ("p95_wait", "95th-percentile wait (s)", "benchmark_p95_wait.png"),
        ("long_wait_pct", "Waits over threshold (%)", "benchmark_long_waits.png"),
        ("energy", "Energy (weighted)", "benchmark_energy.png"),
    ]
    scenarios = list(dict.fromkeys(frame["scenario"]))
    strategies = list(dict.fromkeys(frame["strategy"]))
    # A colour-blind-safe qualitative palette, dark enough to read when printed.
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for metric, label, filename in charts:
        figure, axes = plt.subplots(figsize=(9.5, 4.6), dpi=150)
        width = 0.8 / max(1, len(strategies))
        for index, strategy in enumerate(strategies):
            means, errors = [], []
            for scenario in scenarios:
                subset = frame[(frame["scenario"] == scenario) & (frame["strategy"] == strategy)][
                    metric
                ]
                means.append(float(subset.mean()) if not subset.empty else 0.0)
                errors.append(float(subset.std()) if len(subset) > 1 else 0.0)
            positions = [i + index * width - 0.4 + width / 2 for i in range(len(scenarios))]
            axes.bar(
                positions,
                means,
                width=width,
                yerr=errors,
                capsize=3,
                label=strategy,
                color=palette[index % len(palette)],
                edgecolor="white",
                linewidth=0.6,
            )
        axes.set_xticks(range(len(scenarios)))
        axes.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
        axes.set_ylabel(label)
        axes.set_title(f"{label} by strategy (mean +/- sd over seeds)", fontsize=11)
        axes.legend(frameon=False, fontsize=9)
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="y", alpha=0.25, linewidth=0.6)
        figure.tight_layout()
        path = directory / filename
        figure.savefig(path)
        plt.close(figure)
        written.append(path)

    return written
