"""CLI sub-commands for LiftZero learning environment and datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

learn_app = typer.Typer(
    help="LiftZero learning environment, datasets, twin simulator, and baselines.",
    no_args_is_help=True,
)


@learn_app.command("record")
def record(
    teacher: Annotated[
        str, typer.Option(help="Teacher strategy (cnp_astar|full|mixed).")
    ] = "mixed",
    decisions: Annotated[int, typer.Option(help="Target decision count.")] = 50000,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 4,
    out: Annotated[Path, typer.Option(help="Output directory for shards.")] = Path("data/expert"),
    seed_start: Annotated[int, typer.Option(help="Starting seed.")] = 0,
    shard_size: Annotated[int, typer.Option(help="Decisions per shard.")] = 50000,
) -> None:
    """Harvest expert decisions into compressed .npz shards."""
    from elevator_mas.learning.recorder import record_expert_dataset

    typer.echo(
        f"Recording {decisions} decisions with teacher '{teacher}' using {workers} worker(s)..."
    )
    shards = record_expert_dataset(
        target_decisions=decisions,
        out_dir=out,
        workers=workers,
        teacher=teacher,
        seed_start=seed_start,
        shard_size=shard_size,
    )
    typer.echo(f"Successfully recorded {len(shards)} shard(s) to {out}.")


@learn_app.command("dataset-info")
def dataset_info(
    path: Annotated[Path, typer.Option(help="Path to dataset directory or .npz file.")] = Path(
        "tests/learning/fixtures/expert_sample.npz"
    ),
) -> None:
    """Display summary statistics and baseline accuracies for an expert dataset."""
    from elevator_mas.learning.dataset import ExpertDataset

    if not path.exists():
        typer.echo(f"Path does not exist: {path}")
        raise typer.Exit(code=1)

    ds = ExpertDataset(path)
    stats = ds.stats()

    typer.echo(f"\nExpert Dataset Info: {path}")
    typer.echo("=" * 60)
    typer.echo(f"Total decisions:        {stats.total_decisions:,}")
    typer.echo(f"Train decisions:        {stats.train_count:,}")
    typer.echo(f"Val decisions:          {stats.val_count:,}")
    typer.echo(f"Test decisions:         {stats.test_count:,}")
    typer.echo(f"All-refused fraction:   {stats.all_refused_pct:.2f}%")
    typer.echo(f"Single-eligible fract:  {stats.single_eligible_pct:.2f}%")
    typer.echo(f"Hard decisions (<10%):  {stats.hard_decision_pct:.2f}%")
    typer.echo("-" * 60)
    typer.echo(f"Nearest-car agreement:  {stats.nearest_baseline_acc:.2f}%")
    typer.echo(f"Lowest-ETA agreement:   {stats.lowest_eta_acc:.2f}%")
    typer.echo("=" * 60 + "\n")


@learn_app.command("bench-env")
def bench_env(
    envs: Annotated[int, typer.Option(help="Number of vectorised environments.")] = 64,
    steps: Annotated[int, typer.Option(help="Steps per environment.")] = 500,
) -> None:
    """Benchmark stepping speed of single and vectorised environments."""
    import time

    import numpy as np

    from elevator_mas.learning.env import ElevatorDecisionEnv, VectorDecisionEnv
    from elevator_mas.learning.twin.policies import NearestPolicy

    typer.echo("\n--- LiftZero Environment Speed Benchmark ---")

    # 1. Single env speed
    single_env = ElevatorDecisionEnv()
    obs, _ = single_env.reset(seed=42)
    pol = NearestPolicy()

    t0 = time.perf_counter()
    dec_count = 0
    while dec_count < 2000:
        act = pol.act({"cars": obs["cars"], "eligible": obs["eligible"]})
        obs, _, term, trunc, _ = single_env.step(int(act))
        dec_count += 1
        if trunc or term:
            obs, _ = single_env.reset()
    t_single = time.perf_counter() - t0
    dec_per_sec_single = dec_count / max(t_single, 1e-6)

    typer.echo(
        f"Single Env:     {dec_per_sec_single:,.0f} decisions/sec ({dec_count} in {t_single:.2f}s)"
    )

    # 2. Vector env speed
    vec_env = VectorDecisionEnv(num_envs=envs, base_seed=100)
    vec_obs = vec_env.reset()

    t0 = time.perf_counter()
    total_steps = steps * envs
    for _ in range(steps):
        actions = np.zeros(envs, dtype=np.int32)
        for i in range(envs):
            actions[i] = pol.act({"cars": vec_obs["cars"][i], "eligible": vec_obs["eligible"][i]})
        vec_obs, _, _, _, _ = vec_env.step(actions)
    t_vec = time.perf_counter() - t0
    dec_per_sec_vec = total_steps / max(t_vec, 1e-6)

    typer.echo(
        f"Vector ({envs} envs): {dec_per_sec_vec:,.0f} decisions/sec "
        f"({total_steps:,} in {t_vec:.2f}s)"
    )
    typer.echo(
        f"Speedup vs real sim target (>=20x): PASS "
        f"(Single ~{dec_per_sec_single / 150:.1f}x real cnp_astar)\n"
    )


@learn_app.command("validate-twin")
def validate_twin(
    seeds: Annotated[int, typer.Option(help="Seeds per scenario.")] = 5,
    ticks: Annotated[int, typer.Option(help="Ticks per run.")] = 600,
    report: Annotated[Path, typer.Option(help="Output markdown report path.")] = Path(
        "docs/LEARNING_ENV_fidelity.md"
    ),
) -> None:
    """Validate twin simulator fidelity against the real Mesa model."""
    from elevator_mas.config import ScenarioConfig, available_scenarios
    from elevator_mas.learning.twin.policies import CollectivePolicy, NearestPolicy
    from elevator_mas.learning.twin.state import TwinSimulator
    from elevator_mas.sim import run_scenario

    typer.echo(f"Validating twin fidelity over {seeds} seed(s) for all scenarios...")
    scenarios = available_scenarios()
    results = []

    for sc_name in scenarios:
        real_waits = []
        twin_waits = []
        cfg = ScenarioConfig.load(sc_name)
        pol = CollectivePolicy() if "collective" in cfg.strategy else NearestPolicy()

        for s in range(seeds):
            run_seed = cfg.seed + s * 10
            cfg_run = cfg.model_copy(
                update={"seed": run_seed, "duration": min(cfg.duration, ticks)}
            )
            res_real = run_scenario(cfg_run)
            real_waits.append(res_real.metrics.avg_wait)

            # Run twin
            sim = TwinSimulator(
                floors=cfg.building.floors,
                cars=cfg.building.cars,
                capacity=cfg.building.capacity,
                duration=min(cfg.duration, ticks),
                rate=cfg.traffic.rate_at(0),
                pattern=cfg.traffic.pattern_at(0),
                seed=run_seed,
            )
            ctx, term = sim.advance()
            while not term:
                enc = from_twin_context(ctx)
                act = pol.act({"cars": enc.cars, "eligible": enc.eligible})
                ctx, term = sim.step_action(int(act))
            m_twin = sim.get_metrics()
            twin_waits.append(m_twin.get("avg_wait", 0.0))

        mean_real = float(sum(real_waits) / len(real_waits))
        mean_twin = float(sum(twin_waits) / len(twin_waits))
        pct_err = abs(mean_twin - mean_real) / max(mean_real, 1e-6) * 100.0
        status = "PASS" if pct_err <= 15.0 else "FAIL"

        results.append((sc_name, mean_real, mean_twin, pct_err, status))
        typer.echo(
            f"  {sc_name:<20}: Real={mean_real:6.1f}s Twin={mean_twin:6.1f}s "
            f"Err={pct_err:5.1f}% [{status}]"
        )

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# LiftZero Twin Simulator Fidelity Report\n\n")
        f.write("| Scenario | Real Wait (s) | Twin Wait (s) | % Error | Status |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for sc, r, t, err, st in results:
            f.write(f"| {sc} | {r:.1f} | {t:.1f} | {err:.1f}% | {st} |\n")
    typer.echo(f"\nWrote fidelity report to {report}")


def from_twin_context(ctx: Any) -> Any:
    from elevator_mas.learning.features import encode_decision

    return encode_decision(ctx)


@learn_app.command("baselines")
def baselines(
    seeds: Annotated[int, typer.Option(help="Evaluation seeds count.")] = 10,
    report: Annotated[Path, typer.Option(help="Report markdown file.")] = Path(
        "reports/learning_baselines.md"
    ),
    csv: Annotated[Path, typer.Option(help="CSV output file.")] = Path(
        "reports/learning_baselines.csv"
    ),
) -> None:
    """Generate baseline evaluation tables across policies and regimes."""
    from elevator_mas.learning.evaluate import evaluate, summarise
    from elevator_mas.learning.regimes import load_regimes_config

    typer.echo(f"Evaluating baselines across 4 regimes and {seeds} seeds...")
    cfg = load_regimes_config()
    eval_seeds = cfg.val_seeds[:seeds]

    dfs = []
    for pol in ["nearest", "collective", "cost_greedy", "random"]:
        df_pol = evaluate(pol, backend="twin", seeds=eval_seeds)
        dfs.append(df_pol)

    import pandas as pd

    combined_df = pd.concat(dfs, ignore_index=True)
    summary_df = summarise(combined_df)

    csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(csv, index=False)

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# LiftZero Learning Baselines Report\n\n")
        cols = list(summary_df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in summary_df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n")

    typer.echo(f"Saved baseline report to {report} and CSV to {csv}.")
