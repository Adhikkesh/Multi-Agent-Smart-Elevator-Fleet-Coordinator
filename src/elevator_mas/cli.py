"""The `elevator` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from elevator_mas.config import ScenarioConfig, available_scenarios
from elevator_mas.learning.cli import learn_app
from elevator_mas.strategies import STRATEGIES, strategy_names

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Multi-Agent Smart Elevator Fleet Coordinator — simulation, dashboard, benchmark.",
)
app.add_typer(learn_app, name="learn")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    scenario: Annotated[str, typer.Option(help="Scenario to load at startup.")] = "demo_story",
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
) -> None:
    """Start the dashboard server (REST + WebSocket) and serve the single-page UI."""
    import uvicorn

    from elevator_mas.api.server import create_app

    if scenario not in available_scenarios():
        typer.echo(f"unknown scenario {scenario!r}; try: {', '.join(available_scenarios())}")
        raise typer.Exit(code=1)

    typer.echo(f"Elevator dashboard on http://{host}:{port}  (scenario: {scenario})")
    if reload:
        uvicorn.run(
            "elevator_mas.api.server:app", host=host, port=port, reload=True, log_level="info"
        )
    else:
        uvicorn.run(create_app(scenario), host=host, port=port, log_level="info")


@app.command()
def run(
    scenario: Annotated[str, typer.Option(help="Scenario name.")] = "morning_up_peak",
    strategy: Annotated[str | None, typer.Option(help="Override the strategy.")] = None,
    seed: Annotated[int | None, typer.Option(help="Override the seed.")] = None,
    ticks: Annotated[int | None, typer.Option(help="Override the duration in ticks.")] = None,
    csv: Annotated[Path | None, typer.Option(help="Write per-tick metrics to CSV.")] = None,
    drain: Annotated[bool, typer.Option(help="Run on until everyone is delivered.")] = False,
) -> None:
    """Run one scenario headlessly and print its metrics."""
    from elevator_mas.sim import run_scenario

    if scenario not in available_scenarios():
        typer.echo(f"unknown scenario {scenario!r}; try: {', '.join(available_scenarios())}")
        raise typer.Exit(code=1)
    if strategy is not None and strategy not in STRATEGIES:
        typer.echo(f"unknown strategy {strategy!r}; try: {', '.join(strategy_names())}")
        raise typer.Exit(code=1)

    result = run_scenario(
        scenario, strategy=strategy, seed=seed, ticks=ticks, csv_path=csv, drain=drain
    )
    typer.echo(result.summary())
    if csv is not None:
        typer.echo(f"\nPer-tick metrics written to {csv}")
    if result.violations:
        raise typer.Exit(code=1)


@app.command()
def bench(
    scenarios: Annotated[
        str | None, typer.Option(help="Comma-separated scenarios (default: the four regimes).")
    ] = None,
    strategies: Annotated[
        str | None, typer.Option(help="Comma-separated strategies (default: all four).")
    ] = None,
    seeds: Annotated[int, typer.Option(help="How many seeds per combination.")] = 5,
    ticks: Annotated[int, typer.Option(help="Ticks per run.")] = 900,
    out: Annotated[Path, typer.Option(help="Output directory for CSVs and charts.")] = Path(
        "reports"
    ),
) -> None:
    """Benchmark strategies x scenarios x seeds and write CSVs plus PNG charts."""
    from elevator_mas.sim import run_benchmark, write_reports
    from elevator_mas.sim.benchmark import DEFAULT_SCENARIOS, DEFAULT_STRATEGIES

    scenario_list = (
        [s.strip() for s in scenarios.split(",")] if scenarios else list(DEFAULT_SCENARIOS)
    )
    strategy_list = (
        [s.strip() for s in strategies.split(",")] if strategies else list(DEFAULT_STRATEGIES)
    )
    for name in scenario_list:
        if name not in available_scenarios():
            typer.echo(f"unknown scenario {name!r}")
            raise typer.Exit(code=1)
    for name in strategy_list:
        if name not in STRATEGIES:
            typer.echo(f"unknown strategy {name!r}")
            raise typer.Exit(code=1)

    seed_list = list(range(1, seeds + 1))
    total = len(scenario_list) * len(strategy_list) * len(seed_list)
    typer.echo(
        f"Benchmarking {len(strategy_list)} strategies x {len(scenario_list)} scenarios "
        f"x {len(seed_list)} seeds = {total} runs of {ticks} ticks\n"
    )

    result = run_benchmark(scenario_list, strategy_list, seed_list, ticks=ticks, progress=True)

    typer.echo("\n=== Mean +/- sd by scenario and strategy ===")
    aggregate = result.aggregate()
    columns = ["scenario", "strategy", "avg_wait_mean", "avg_wait_std", "p95_wait_mean"]
    columns += ["long_wait_pct_mean", "delivered_mean"]
    available = [c for c in columns if c in aggregate.columns]
    typer.echo(aggregate[available].to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    typer.echo("\n=== Full strategy versus the NearestCar baseline ===")
    comparison = result.comparison()
    if not comparison.empty:
        typer.echo(comparison.to_string(index=False))
        for scenario, won in result.wins().items():
            mark = "beats baseline" if won else "does NOT beat baseline"
            typer.echo(f"  {scenario:<22} {mark}")

    written = write_reports(result, out)
    typer.echo(f"\nWrote {len(written)} file(s) to {out}/:")
    for path in written:
        typer.echo(f"  {path}")


@app.command()
def scenarios() -> None:
    """List the bundled scenarios and the available strategies."""
    typer.echo("Scenarios:")
    for name in available_scenarios():
        config = ScenarioConfig.load(name)
        description = " ".join(config.description.split())
        typer.echo(
            f"  {name:<20} {config.duration:>5}s  "
            f"{config.building.floors} floors, {config.building.cars} cars, "
            f"{len(config.events)} event(s)"
        )
        if description:
            typer.echo(f"  {'':<20} {description[:96]}")
    typer.echo("\nStrategies:")
    for name, strategy in STRATEGIES.items():
        typer.echo(f"  {name:<14} {strategy.label}")


@app.command()
def verify() -> None:
    """Run every scenario once and report invariant violations. A quick self-check."""
    from elevator_mas.sim import run_scenario

    failures = 0
    for name in available_scenarios():
        result = run_scenario(name)
        status = "OK" if not result.violations else f"{len(result.violations)} VIOLATION(S)"
        typer.echo(
            f"  {name:<20} {result.runtime_seconds:6.2f}s  "
            f"avg_wait={result.metrics.avg_wait:7.1f}s  "
            f"delivered={result.metrics.delivered:4d}/{result.metrics.arrived:<4d}  {status}"
        )
        for violation in result.violations[:5]:
            typer.echo(f"      ! {violation}")
        failures += len(result.violations)
    if failures:
        typer.echo(f"\n{failures} violation(s) found")
        raise typer.Exit(code=1)
    typer.echo("\nAll scenarios ran with every invariant holding.")


if __name__ == "__main__":
    app()
