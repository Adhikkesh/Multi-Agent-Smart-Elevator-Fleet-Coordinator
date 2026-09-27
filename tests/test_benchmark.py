"""The benchmark harness, including the report and chart generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from elevator_mas.sim import run_benchmark, run_scenario, write_reports
from elevator_mas.sim.benchmark import BenchmarkResult


@pytest.fixture(scope="module")
def small_benchmark() -> BenchmarkResult:
    """A deliberately tiny benchmark: enough structure, little runtime."""
    return run_benchmark(
        scenarios=["interfloor_light", "morning_up_peak"],
        strategies=["nearest_car", "full"],
        seeds=(1, 2),
        ticks=200,
    )


class TestBenchmark:
    """Running the matrix."""

    def test_runs_every_combination(self, small_benchmark: BenchmarkResult) -> None:
        """2 scenarios x 2 strategies x 2 seeds."""
        assert len(small_benchmark.runs) == 8

    def test_frame_has_a_row_per_run(self, small_benchmark: BenchmarkResult) -> None:
        """One row each, with the metrics flattened in."""
        frame = small_benchmark.frame()
        assert len(frame) == 8
        for column in ("scenario", "strategy", "seed", "avg_wait", "p95_wait", "energy"):
            assert column in frame.columns

    def test_aggregate_reports_mean_and_sd(self, small_benchmark: BenchmarkResult) -> None:
        """Grouped by scenario and strategy, with readable column names."""
        aggregate = small_benchmark.aggregate()
        assert len(aggregate) == 4
        assert "scenario" in aggregate.columns
        assert "strategy" in aggregate.columns
        assert "avg_wait_mean" in aggregate.columns
        assert "avg_wait_std" in aggregate.columns

    def test_comparison_is_signed_so_positive_is_better(
        self, small_benchmark: BenchmarkResult
    ) -> None:
        """The headline table."""
        comparison = small_benchmark.comparison()
        assert len(comparison) == 2
        for _, row in comparison.iterrows():
            baseline = row["avg_wait_nearest_car"]
            champion = row["avg_wait_full"]
            expected = 100.0 * (baseline - champion) / baseline
            assert row["avg_wait_improvement_pct"] == pytest.approx(expected, abs=0.1)

    def test_wins_reports_a_verdict_per_scenario(self, small_benchmark: BenchmarkResult) -> None:
        """A boolean per scenario, used by the CLI and the dashboard."""
        wins = small_benchmark.wins()
        assert set(wins) == {"interfloor_light", "morning_up_peak"}
        assert all(isinstance(value, bool) for value in wins.values())

    def test_empty_benchmark_does_not_crash(self) -> None:
        """Aggregating nothing must return an empty frame, not raise."""
        empty = BenchmarkResult()
        assert empty.frame().empty
        assert empty.aggregate().empty
        assert empty.comparison().empty
        assert empty.wins() == {}


class TestReports:
    """The files the slides are built from."""

    def test_writes_csvs_and_charts(self, small_benchmark: BenchmarkResult, tmp_path: Path) -> None:
        """Three CSVs and four PNGs, all non-empty."""
        written = write_reports(small_benchmark, tmp_path)
        names = {path.name for path in written}
        assert {
            "benchmark_runs.csv",
            "benchmark_summary.csv",
            "benchmark_comparison.csv",
            "benchmark_avg_wait.png",
            "benchmark_p95_wait.png",
            "benchmark_long_waits.png",
            "benchmark_energy.png",
        } <= names
        for path in written:
            assert path.exists()
            assert path.stat().st_size > 0

    def test_charts_are_real_pngs(self, small_benchmark: BenchmarkResult, tmp_path: Path) -> None:
        """Guard against writing a zero-byte or corrupt image."""
        written = write_reports(small_benchmark, tmp_path)
        for path in (p for p in written if p.suffix == ".png"):
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


class TestRunner:
    """The single-run wrapper."""

    def test_summary_mentions_the_key_metrics(self) -> None:
        """`elevator run` prints this."""
        summary = run_scenario("interfloor_light", ticks=150).summary()
        for fragment in ("average wait", "throughput", "energy", "invariants"):
            assert fragment in summary

    def test_csv_export(self, tmp_path: Path) -> None:
        """Per-tick metrics, straight from the Mesa DataCollector."""
        path = tmp_path / "run.csv"
        run_scenario("interfloor_light", ticks=120, csv_path=path)
        assert path.exists()
        lines = path.read_text().splitlines()
        assert len(lines) == 121  # a header plus one row per tick
        assert "avg_wait" in lines[0]

    def test_overrides_are_applied(self) -> None:
        """Strategy and seed can be overridden without editing the YAML."""
        result = run_scenario("interfloor_light", strategy="nearest_car", seed=77, ticks=100)
        assert result.strategy == "nearest_car"
        assert result.seed == 77
