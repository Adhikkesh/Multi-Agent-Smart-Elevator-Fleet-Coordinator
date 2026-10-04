"""DAgger (Dataset Aggregation) for the LiftZero bidder.

Behaviour cloning trains on the states the *teacher* visits; once the learner drives, its
small mistakes lead to states the teacher never produced, and errors compound (Ross et al.,
2011). DAgger fixes the distribution: roll out the *learner*, ask the teacher what it would
have done in every state the learner reached, aggregate those labels and retrain.

One round here:

1. export the current checkpoint to ONNX (the simulator runs the ONNX runtime);
2. roll out ``liftzero_bc`` / ``liftzero_bc_cnp`` in the real simulator on fresh
   training-split seeds with ``shadow_teacher=True`` — every car also computes its A* bid;
3. award each auction by the teacher with probability beta, else by the learner (keeps
   exploration safe while still visiting learner-induced states);
4. record every decision with the **teacher's** labels (``source = dagger_r{r}``);
5. fine-tune from the previous checkpoint on expert + all DAgger data (lower lr).

Splits stay by run seed: training rollouts use seeds 2500-6999 (train split, disjoint from
the expert harvest's seeds 0-2258), DAgger validation rollouts use val seeds 8260-8499.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elevator_mas.config import LiftConfig
from elevator_mas.learning.recorder import DecisionRecorder, sample_random_regime

#: The learned strategy that stands in for each teacher in DAgger rollouts.
LEARNER_OF = {"full": "liftzero_bc", "cnp_astar": "liftzero_bc_cnp"}

#: Seed blocks per round: training rollouts and DAgger-state validation rollouts.
#: The expert harvest used training seeds 0-2258 and validation seeds 8000-8247, so every
#: DAgger block below is fresh.
TRAIN_SEED_BASE = {1: 2500, 2: 4000, 3: 5500}
VAL_SEED_BASE = {1: 8260, 2: 8340, 3: 8420}


@dataclass
class RolloutStats:
    runs: int
    decisions: int
    teacher_awards: int
    learner_awards: int

    @property
    def teacher_fraction(self) -> float:
        total = self.teacher_awards + self.learner_awards
        return self.teacher_awards / total if total else float("nan")


def rollout(
    seed: int,
    model_path: str,
    beta: float,
    recorder: DecisionRecorder,
) -> tuple[int, int, int]:
    """One learner-driven run on a sampled training building; returns award statistics."""
    from elevator_mas.model import ElevatorModel

    cfg = sample_random_regime(run_id=seed, seed=seed, teacher_choice="mixed")
    cfg = cfg.model_copy(
        update={
            "strategy": LEARNER_OF[cfg.strategy],
            "lift": LiftConfig(shadow_teacher=True, beta=beta, model_path=model_path),
        }
    )
    model = ElevatorModel(cfg)
    model.decision_hooks.append(recorder)
    before = recorder.recorded_total
    for _ in range(cfg.duration):
        model.step()
    d = model.dispatcher
    return recorder.recorded_total - before, d.teacher_awards, d.learner_awards


def _worker(
    worker_id: int,
    seeds: list[int],
    model_path: str,
    beta: float,
    out_dir: str,
    source: str,
) -> tuple[int, int, int, int]:
    rec = DecisionRecorder(out_dir, shard_prefix=f"{source}_w{worker_id}", source=source)
    totals = [0, 0, 0]
    for s in seeds:
        n, t, lrn = rollout(s, model_path, beta, rec)
        totals[0] += n
        totals[1] += t
        totals[2] += lrn
    rec.flush()
    return len(seeds), totals[0], totals[1], totals[2]


def _worker_star(args: tuple[Any, ...]) -> tuple[int, int, int, int]:
    return _worker(*args)


def collect(
    seeds: list[int],
    model_path: str | Path,
    beta: float,
    out_dir: str | Path,
    source: str,
    workers: int = 1,
) -> RolloutStats:
    """Roll out the learner on ``seeds`` and record teacher-labelled shards."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers, len(seeds)))
    chunks = [seeds[w::workers] for w in range(workers)]
    jobs = [(w, chunks[w], str(model_path), beta, str(out_dir), source) for w in range(workers)]
    if workers == 1:
        results = [_worker_star(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_worker_star, jobs)
    return RolloutStats(
        runs=sum(r[0] for r in results),
        decisions=sum(r[1] for r in results),
        teacher_awards=sum(r[2] for r in results),
        learner_awards=sum(r[3] for r in results),
    )


def round_seeds(round_: int, runs: int, val_runs: int) -> tuple[list[int], list[int]]:
    """Fresh, split-safe seeds for one DAgger round."""
    tb, vb = TRAIN_SEED_BASE[round_], VAL_SEED_BASE[round_]
    train = list(range(tb, tb + runs))
    val = list(range(vb, vb + val_runs))
    if train[-1] >= 8000 or val[-1] >= 8500:
        raise ValueError("DAgger seeds would leave their split; reduce --runs")
    return train, val


def run_round(
    round_: int,
    ckpt: str | Path,
    runs: int,
    beta: float,
    data_root: str | Path,
    workers: int,
    val_runs: int = 80,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Export ``ckpt``, roll it out, and record round ``round_`` (train + val shards)."""
    from elevator_mas.learning.lift.export import export_checkpoint

    data_root = Path(data_root)
    model_path = data_root / f"dagger_r{round_}_policy.onnx"
    export_checkpoint(ckpt, model_path, version=f"dagger-r{round_}-input")
    train_seeds, val_seeds = round_seeds(round_, runs, val_runs)
    source = f"dagger_r{round_}"
    log(f"[dagger] round {round_}: {len(train_seeds)} train + {len(val_seeds)} val rollouts")
    tr = collect(train_seeds, model_path, beta, data_root / source / "train", source, workers)
    va = collect(val_seeds, model_path, 0.0, data_root / source / "val", source, workers)
    log(
        f"[dagger] round {round_}: {tr.decisions:,} train decisions, teacher-executed "
        f"{tr.teacher_fraction:.3f} (beta {beta}); {va.decisions:,} val decisions"
    )
    return {
        "round": round_,
        "beta": beta,
        "train_runs": tr.runs,
        "train_decisions": tr.decisions,
        "teacher_fraction": tr.teacher_fraction,
        "val_runs": va.runs,
        "val_decisions": va.decisions,
        "train_dir": str(data_root / source / "train"),
        "val_dir": str(data_root / source / "val"),
        "policy": str(model_path),
    }
