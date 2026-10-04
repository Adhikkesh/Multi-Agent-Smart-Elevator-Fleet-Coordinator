"""``elevator lift ...`` — train, evaluate, export and benchmark the LiftZero bidder.

Training commands need the ``lift-train`` dependency group (PyTorch); ``card``,
``bench-infer`` and ``eval-sim`` need only the base install (onnxruntime).
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer

lift_app = typer.Typer(
    help="LiftZero learned bidder: imitation training, DAgger, ONNX export, evaluation.",
    no_args_is_help=True,
)

DEFAULT_MODEL = Path("models/liftzero_bc_v1.onnx")


def _echo(msg: str) -> None:
    typer.echo(msg)


@lift_app.command("train-bc")
def train_bc(
    config: Annotated[Path | None, typer.Option(help="Training YAML (overrides --preset).")] = None,
    preset: Annotated[str, typer.Option(help="smoke | default | large | dagger")] = "default",
    seed: Annotated[int | None, typer.Option(help="Override the config seed.")] = None,
    out: Annotated[Path, typer.Option(help="Directory for run folders.")] = Path("runs"),
    device: Annotated[str | None, typer.Option(help="auto | cpu | mps | cuda")] = None,
) -> None:
    """Train LiftZeroNet by behaviour cloning of the A* teacher.

    Example: elevator lift train-bc --preset smoke --seed 0
    """
    from elevator_mas.learning.lift.config import load_train_config
    from elevator_mas.learning.lift.train_bc import train_from_config

    cfg = load_train_config(config, preset).with_overrides(seed=seed, device=device)
    res = train_from_config(cfg, out, log=_echo)
    typer.echo(f"best checkpoint: {res.run_dir / 'best.pt'}")


@lift_app.command("eval-offline")
def eval_offline(
    ckpt: Annotated[Path, typer.Option(help="Checkpoint (best.pt).")],
    split: Annotated[
        list[str] | None, typer.Option(help="val | test | test_large (repeatable).")
    ] = None,
    limit: Annotated[int, typer.Option(help="Max decisions per split.")] = 100_000,
) -> None:
    """Agreement, regret, calibration and failure modes against recorded teacher decisions.

    Example: elevator lift eval-offline --ckpt runs/bc_default_s0_*/best.pt --split val
    """
    from elevator_mas.learning.lift.eval_offline import run_offline_eval

    splits = tuple(split) if split else ("val", "test", "test_large")
    run_offline_eval(ckpt, splits, limit=limit, log=_echo)
    typer.echo("wrote docs/lift/offline_eval.md and reports/lift/offline_*.png")


@lift_app.command("export")
def export(
    ckpt: Annotated[Path, typer.Option(help="Checkpoint (best.pt).")],
    out: Annotated[Path, typer.Option(help="Output ONNX path.")] = DEFAULT_MODEL,
    version: Annotated[str, typer.Option(help="Model version string.")] = "1",
) -> None:
    """Export to ONNX (dynamic batch/fleet axes), verify parity, write the model card.

    Example: elevator lift export --ckpt runs/.../best.pt --out models/liftzero_bc_v1.onnx
    """
    from elevator_mas.learning.lift.export import export_checkpoint

    card = export_checkpoint(ckpt, out, version=version)
    typer.echo(
        f"wrote {out} ({card.param_count:,} params, parity "
        f"{card.metrics['onnx_parity_max_abs']:.1e}) and {out.with_suffix('.json')}"
    )


def cpu_model() -> str:
    """Human-readable CPU name for benchmark reports."""
    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


@lift_app.command("bench-infer")
def bench_infer(
    model: Annotated[Path, typer.Option(help="ONNX model.")] = DEFAULT_MODEL,
    reps: Annotated[int, typer.Option(help="Timed decisions per fleet size.")] = 2000,
    out: Annotated[Path | None, typer.Option(help="Optional JSON output.")] = None,
) -> None:
    """Single-decision CPU latency (median / p95 / p99 ms) at N = 2, 4, 8, 16, 32.

    Acceptance: median <= 2 ms at N=8 and <= 5 ms at N=32.

    Example: elevator lift bench-infer --model models/liftzero_bc_v1.onnx
    """
    from elevator_mas.learning.lift.runtime import LiftRuntime, bench_latency

    rt = LiftRuntime(model)
    rows = bench_latency(rt, sizes=(2, 4, 8, 16, 32), reps=reps)
    cpu = cpu_model()
    typer.echo(f"\nLiftZero inference latency — {model} on {cpu} (1 thread)")
    typer.echo(f"{'N':>4} {'median ms':>10} {'p95 ms':>8} {'p99 ms':>8}")
    for r in rows:
        typer.echo(f"{r['n']:>4} {r['median_ms']:>10.3f} {r['p95_ms']:>8.3f} {r['p99_ms']:>8.3f}")
    by_n = {r["n"]: r for r in rows}
    ok8 = by_n[8]["median_ms"] <= 2.0
    ok32 = by_n[32]["median_ms"] <= 5.0
    typer.echo(f"N=8 median <= 2 ms: {'PASS' if ok8 else 'FAIL'}")
    typer.echo(f"N=32 median <= 5 ms: {'PASS' if ok32 else 'FAIL'}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"cpu": cpu, "model": str(model), "rows": rows}, indent=2))


@lift_app.command("eval-sim")
def eval_sim(
    strategy: Annotated[
        list[str] | None, typer.Option(help="Strategies (repeatable; default: the 6-way set).")
    ] = None,
    regimes: Annotated[str, typer.Option(help="all | regimes | scenarios | a,b,c")] = "regimes",
    seeds: Annotated[str, typer.Option(help="val (8000-8049) | test (8500-8599)")] = "val",
    n: Annotated[int, typer.Option(help="Seeds per regime.")] = 20,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 8,
    shadow: Annotated[bool, typer.Option(help="Shadow teacher for learned strategies.")] = True,
    model: Annotated[Path | None, typer.Option(help="ONNX model override.")] = None,
    out: Annotated[Path, typer.Option(help="Markdown report.")] = Path("docs/lift/closed_loop.md"),
    csv: Annotated[Path, typer.Option(help="Per-run CSV.")] = Path(
        "reports/lift/closed_loop_runs.csv"
    ),
) -> None:
    """Closed-loop runs in the real simulator: waits, p95, energy, safety, teacher agreement.

    Example: elevator lift eval-sim --strategy liftzero_bc --strategy full --seeds test --n 100
    """
    from elevator_mas.learning.lift.eval_sim import (
        DEFAULT_STRATEGIES,
        regime_names,
        run_matrix,
        seeds_for,
        write_markdown,
    )

    strategies = strategy or list(DEFAULT_STRATEGIES)
    df = run_matrix(
        strategies,
        regime_names(regimes),
        seeds_for(seeds, n),
        workers=workers,
        shadow=shadow,
        model_path=str(model) if model else None,
        log=_echo,
    )
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    write_markdown(df, out, note=f"Seeds: `{seeds}` (n={n}); generated {time.strftime('%F')}.")
    bad = int(df["violations"].sum())
    typer.echo(f"wrote {out} and {csv}; safety violations: {bad}")
    if bad:
        raise typer.Exit(code=1)


@lift_app.command("dagger")
def dagger(
    round_: Annotated[int, typer.Option("--round", help="DAgger round (1, 2, 3).")],
    ckpt: Annotated[Path, typer.Option(help="Checkpoint to roll out and fine-tune.")],
    runs: Annotated[int, typer.Option(help="Fresh training-split rollouts.")] = 1500,
    val_runs: Annotated[int, typer.Option(help="DAgger-state validation rollouts.")] = 80,
    beta: Annotated[float, typer.Option(help="P(award executed by teacher).")] = 0.5,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 8,
    data: Annotated[Path, typer.Option(help="DAgger data root.")] = Path("data/dagger"),
    train: Annotated[bool, typer.Option(help="Fine-tune after collecting.")] = True,
    max_train: Annotated[
        int | None, typer.Option(help="Cap on aggregated training decisions (memory).")
    ] = 1_600_000,
) -> None:
    """One DAgger round: roll out the learner, label with the teacher, aggregate, fine-tune.

    Example: elevator lift dagger --round 1 --ckpt runs/.../best.pt --runs 1500 --beta 0.5
    """
    from elevator_mas.learning.lift.dagger import run_round

    info = run_round(round_, ckpt, runs, beta, data, workers, val_runs, log=_echo)
    if train:
        from elevator_mas.learning.lift.config import load_train_config
        from elevator_mas.learning.lift.train_bc import fine_tune

        cfg = load_train_config(preset="dagger")
        dagger_train = [str(data / f"dagger_r{r}" / "train") for r in range(1, round_ + 1)]
        dagger_val = [str(data / f"dagger_r{r}" / "val") for r in range(1, round_ + 1)]
        cfg = cfg.with_overrides(
            name=f"bc_dagger_r{round_}",
            data_train=[*cfg.data_train, *dagger_train],
            data_val=[*cfg.data_val, *dagger_val],
            max_train_decisions=max_train,
        )
        res = fine_tune(cfg, ckpt, log=_echo)
        info["checkpoint"] = str(res.run_dir / "best.pt")
        info["best_val"] = res.best_val
    path = data / f"dagger_r{round_}.json"
    path.write_text(json.dumps(info, indent=2, default=str))
    typer.echo(f"wrote {path}")


@lift_app.command("card")
def card(
    model: Annotated[Path, typer.Option(help="ONNX model.")] = DEFAULT_MODEL,
    markdown: Annotated[Path | None, typer.Option(help="Also render the card here.")] = None,
) -> None:
    """Print a model's card (and optionally render it as Markdown).

    Example: elevator lift card --model models/liftzero_bc_v1.onnx
    """
    from elevator_mas.learning.lift.card import load_card, render_markdown, validate_card

    c = load_card(model)
    try:
        validate_card(c, model)
        status = "compatible with this code"
    except Exception as exc:  # report, don't crash: the card is still worth printing
        status = f"INCOMPATIBLE: {exc}"
    typer.echo(json.dumps(c.as_dict(), indent=2))
    typer.echo(f"\nstatus: {status}")
    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(c), encoding="utf-8")
        typer.echo(f"wrote {markdown}")
