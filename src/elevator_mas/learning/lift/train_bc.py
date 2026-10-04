"""Behaviour-cloning trainer for LiftZeroNet.

Pipeline: load expert shards (shard by shard, trimmed to the largest fleet present) ->
filter trivial decisions -> per-epoch seeded shuffle -> on-the-fly augmentation ->
imitation loss -> AdamW with linear warm-up + cosine decay -> validation agreement ->
early stopping -> ``best.pt``.

Every run writes to ``runs/<name>_s<seed>_<timestamp>/``:

* ``run.json``     — git sha, config, seed, data hashes, torch version, wall time, results;
* ``metrics.csv``  — per epoch: loss components, validation agreement / regret, lr, time;
* ``curves.png``   — training curves (matplotlib, no TensorBoard dependency);
* ``best.pt``      — the checkpoint with the best validation non-trivial agreement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from elevator_mas.learning.dataset import ExpertDataset
from elevator_mas.learning.lift.augment import augment
from elevator_mas.learning.lift.config import (
    ROOT,
    TrainConfig,
    model_config_from_dict,
)
from elevator_mas.learning.lift.losses import Targets, imitation_loss
from elevator_mas.learning.lift.metrics import DecisionMetrics, argmin_choice, decision_metrics
from elevator_mas.learning.lift.model import LiftZeroNet, count_parameters
from elevator_mas.learning.recorder import get_git_sha
from elevator_mas.learning.schema import FEATURE_VERSION, SCHEMA_HASH

Log = Callable[[str], None]

#: Arrays the trainer keeps per decision.
_KEYS = ("call", "cars", "glob", "mask", "eligible", "cost", "parts", "winner")


# ----------------------------------------------------------------------------- data


@dataclass
class Decisions:
    """Training-ready decision arrays (numpy), trimmed to ``N`` car slots."""

    call: np.ndarray  # [M, Kc] float32
    cars: np.ndarray  # [M, N, Kcar] float32
    glob: np.ndarray  # [M, Kg] float32
    mask: np.ndarray  # [M, N] bool
    eligible: np.ndarray  # [M, N] bool
    cost: np.ndarray  # [M, N] float32 teacher bids (1e6 sentinel)
    parts: np.ndarray  # [M, N, 4] float32 teacher breakdown
    winner: np.ndarray  # [M] int64
    data_hash: str = ""
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.winner)

    @property
    def valid(self) -> np.ndarray:
        return self.mask & self.eligible

    def take(self, idx: np.ndarray) -> Decisions:
        return Decisions(
            **{k: getattr(self, k)[idx] for k in _KEYS},
            data_hash=self.data_hash,
            extra={k: v[idx] for k, v in self.extra.items()},
        )

    @classmethod
    def from_dataset(cls, ds: ExpertDataset, n_cars: int | None = None) -> Decisions:
        """Convert an :class:`ExpertDataset`, trimming car slots no decision uses."""
        used = np.flatnonzero(ds.mask.any(axis=0))
        n = int(used.max()) + 1 if len(used) else 1
        n = max(n, n_cars or 0)
        parts = np.stack(
            [ds.teacher_wait, ds.teacher_ride, ds.teacher_crowding, ds.teacher_energy], axis=-1
        )
        return cls(
            call=ds.call.astype(np.float32),
            cars=_pad_cars(ds.cars, n).astype(np.float32),
            glob=ds.glob.astype(np.float32),
            mask=_pad_cars(ds.mask, n),
            eligible=_pad_cars(ds.eligible, n),
            cost=_pad_cars(ds.teacher_cost, n, fill=1e6).astype(np.float32),
            parts=_pad_cars(parts, n).astype(np.float32),
            winner=ds.winner.astype(np.int64),
            data_hash=ds.content_hash(),
            extra={
                "teacher": ds.teacher.copy(),
                "source": ds.source.copy(),
                "run_id": ds.run_id.copy(),
                "split": ds.split_arr.copy(),
                "executed": ds.executed.astype(np.int64),
            },
        )

    @classmethod
    def concat(cls, parts: list[Decisions]) -> Decisions:
        n = max(p.mask.shape[1] for p in parts)
        out: dict[str, np.ndarray] = {}
        for k in _KEYS:
            fill = 1e6 if k == "cost" else 0
            arrays = [getattr(p, k) for p in parts]
            if k in ("cars", "mask", "eligible", "cost", "parts"):
                arrays = [_pad_cars(a, n, fill=fill) for a in arrays]
            out[k] = np.concatenate(arrays, axis=0)
        extra = {k: np.concatenate([p.extra[k] for p in parts]) for k in parts[0].extra}
        digest = "-".join(p.data_hash for p in parts)
        return cls(**out, data_hash=hashlib.sha256(digest.encode()).hexdigest()[:16], extra=extra)


def _pad_cars(a: np.ndarray, n: int, fill: float = 0) -> np.ndarray:
    """Trim or pad axis 1 (cars) to exactly ``n`` slots."""
    if a.shape[1] >= n:
        return a[:, :n]
    pad = [(0, 0)] * a.ndim
    pad[1] = (0, n - a.shape[1])
    return np.pad(a, pad, constant_values=fill)


def filter_trivial(d: Decisions, keep_fraction: float, seed: int) -> Decisions:
    """Drop all-refused decisions and keep trivial ones (one valid car) at <= keep_fraction."""
    n_valid = d.valid.sum(axis=1)
    has = d.winner >= 0
    nt = has & (n_valid >= 2)
    trivial = np.flatnonzero(has & (n_valid == 1))
    n_keep = min(len(trivial), int(keep_fraction / max(1.0 - keep_fraction, 1e-9) * nt.sum()))
    rng = np.random.default_rng(seed)
    kept = rng.choice(trivial, size=n_keep, replace=False) if n_keep else trivial[:0]
    idx = np.sort(np.concatenate([np.flatnonzero(nt), kept]))
    return d.take(idx)


def shard_files(paths: list[str] | list[Path]) -> list[Path]:
    """Every ``.npz`` shard under the given files/directories (repo-relative allowed)."""
    files: list[Path] = []
    for p in paths:
        q = Path(p) if Path(p).is_absolute() else ROOT / p
        files.extend([q] if q.is_file() else sorted(q.glob("*.npz")))
    return files


def load_decisions(
    paths: list[str] | list[Path],
    split: str | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> Decisions:
    """Load shards one at a time (bounded memory), keeping ``split`` only.

    With ``limit``, every shard is subsampled by the same fraction *as it is read* (a seeded
    draw), so the full dataset never has to sit in memory to be cut down.
    """
    files = shard_files(paths)
    if not files:
        raise FileNotFoundError(f"no shards under {paths}")
    frac = 1.0
    if limit is not None:
        total = 0
        for f in files:
            with np.load(f) as z:
                total += len(z["winner"])
        frac = min(1.0, limit / max(total, 1))
    rng = np.random.default_rng(seed)
    chunks: list[Decisions] = []
    for f in files:
        ds = ExpertDataset(f, split=split)
        if frac < 1.0 and len(ds):
            keep = rng.choice(len(ds), size=round(len(ds) * frac), replace=False)
            ds = ds.subset(np.sort(keep))
        if len(ds):
            chunks.append(Decisions.from_dataset(ds))
        del ds
    if not chunks:
        raise ValueError(f"no decisions of split {split!r} under {paths}")
    d = Decisions.concat(chunks)
    if limit is not None and len(d) > limit:
        d = d.take(np.sort(rng.choice(len(d), size=limit, replace=False)))
    return d


def to_tensors(d: Decisions, idx: np.ndarray | slice, device: torch.device) -> dict[str, Any]:
    """A batch of decisions as tensors on ``device``."""
    return {
        "call": torch.from_numpy(d.call[idx]).to(device),
        "cars": torch.from_numpy(d.cars[idx]).to(device),
        "glob": torch.from_numpy(d.glob[idx]).to(device),
        "mask": torch.from_numpy(d.mask[idx]).to(device),
        "eligible": torch.from_numpy(d.eligible[idx]).to(device),
        "teacher_cost": torch.from_numpy(d.cost[idx]).to(device),
        "parts": torch.from_numpy(d.parts[idx]).to(device),
        "winner": torch.from_numpy(d.winner[idx]).to(device),
    }


# ----------------------------------------------------------------------------- model io


def resolve_device(name: str) -> torch.device:
    """``auto`` -> cuda, then mps, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: LiftZeroNet, payload: dict[str, Any]) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model.cfg),
            "feature_version": FEATURE_VERSION,
            "schema_hash": SCHEMA_HASH,
            "param_count": count_parameters(model),
            **payload,
        },
        path,
    )


def load_checkpoint(path: Path | str) -> tuple[LiftZeroNet, dict[str, Any]]:
    """Load a checkpoint into an eval-mode model; refuses a mismatching feature schema."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ckpt.get("schema_hash") != SCHEMA_HASH:
        raise ValueError(
            f"checkpoint schema {ckpt.get('schema_hash')} != current feature schema "
            f"{SCHEMA_HASH} (feature version {FEATURE_VERSION}); retrain the model"
        )
    model = LiftZeroNet(model_config_from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(
    model: LiftZeroNet, d: Decisions, device: torch.device | None = None, batch: int = 4096
) -> dict[str, np.ndarray]:
    """Scores, aux, attention and decisions of ``model`` on every decision of ``d``."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    scores, auxes, attns = [], [], []
    for i in range(0, len(d), batch):
        b = to_tensors(d, slice(i, i + batch), device)
        out = model(b["call"], b["cars"], b["glob"], b["mask"], b["eligible"])
        scores.append(out.score.float().cpu().numpy())
        auxes.append(out.aux.float().cpu().numpy())
        attns.append(out.attn.float().cpu().numpy())
    model.train(was_training)
    score = np.concatenate(scores) if scores else np.zeros((0, d.mask.shape[1]))
    return {
        "score": score,
        "aux": np.concatenate(auxes) if auxes else np.zeros((0, d.mask.shape[1], 4)),
        "attn": np.concatenate(attns) if attns else np.zeros((0, d.mask.shape[1])),
        "choice": argmin_choice(score, d.valid),
    }


def evaluate_decisions(model: LiftZeroNet, d: Decisions) -> DecisionMetrics:
    pred = predict(model, d)
    return decision_metrics(d.cost, d.valid, d.winner, pred["choice"], pred["score"])


# ----------------------------------------------------------------------------- training


@dataclass
class TrainResult:
    run_dir: Path
    best_epoch: int
    best_val: dict[str, Any]
    history: list[dict[str, Any]]
    steps: int
    wall_s: float
    first_losses: list[float]


def lr_lambda(step: int, warmup: int, total: int, base: float, floor: float) -> float:
    """Multiplier of the base lr: linear warm-up, then cosine decay to ``floor``."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return (floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))) / base


def seed_everything(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.default_rng(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def train(
    cfg: TrainConfig,
    train_data: Decisions,
    val_data: Decisions,
    out_dir: Path | str | None = None,
    log: Log = print,
    init_state: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> TrainResult:
    """Train LiftZeroNet by imitation; returns the run summary (``best.pt`` on disk)."""
    t0 = time.perf_counter()
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    gen = seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir or ROOT / "runs") / f"{cfg.name}_s{cfg.seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = LiftZeroNet(cfg.model)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )
    steps_per_epoch = math.ceil(len(train_data) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg.warmup_steps, total, cfg.lr, cfg.min_lr)
    )
    log(
        f"[train] {cfg.name} seed={cfg.seed} device={device} params={count_parameters(model):,} "
        f"train={len(train_data):,} val={len(val_data):,} steps/epoch={steps_per_epoch}"
    )

    history: list[dict[str, Any]] = []
    best_key, best_epoch, best_val, bad_epochs, step = -1.0, -1, {}, 0, 0
    first_losses: list[float] = []
    np_rng = np.random.default_rng(cfg.seed)
    stop = False

    for epoch in range(cfg.epochs):
        model.train()
        order = np_rng.permutation(len(train_data))
        sums: dict[str, float] = {}
        n_batches = 0
        te = time.perf_counter()
        for i in range(0, len(order), cfg.batch_size):
            idx = np.sort(order[i : i + cfg.batch_size])
            batch = augment(to_tensors(train_data, idx, torch.device("cpu")), cfg.augment, gen)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                batch["call"], batch["cars"], batch["glob"], batch["mask"], batch["eligible"]
            )
            tgt = Targets(
                teacher_cost=batch["teacher_cost"],
                parts=batch["parts"],
                winner=batch["winner"],
                valid=batch["mask"] & batch["eligible"],
            )
            loss, parts = imitation_loss(out, tgt, cfg.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if len(first_losses) < 20:
                first_losses.append(float(loss.detach().cpu()))
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.cpu())
            n_batches += 1
            if max_steps is not None and step >= max_steps:
                stop = True
                break

        val = evaluate_decisions(model, val_data).as_dict() if len(val_data) else {}
        row = {
            "epoch": epoch,
            "step": step,
            **{f"train_{k}": v / max(n_batches, 1) for k, v in sums.items()},
            "val_agree_nontrivial": val.get("agree_nontrivial", float("nan")),
            "val_agree_nontrivial_tie": val.get("agree_nontrivial_tie", float("nan")),
            "val_agree_hard": val.get("agree_hard", float("nan")),
            "val_regret": val.get("regret_mean", float("nan")),
            "val_kendall_tau": val.get("kendall_tau", float("nan")),
            "lr": opt.param_groups[0]["lr"],
            "epoch_s": time.perf_counter() - te,
        }
        history.append(row)
        log(
            f"[train] epoch {epoch:2d} loss={row.get('train_loss', 0):.4f} "
            f"val agree={row['val_agree_nontrivial']:.2f}% "
            f"(tie-aware {row['val_agree_nontrivial_tie']:.2f}%, hard {row['val_agree_hard']:.2f}%)"
            f" regret={row['val_regret']:.3f} lr={row['lr']:.2e} {row['epoch_s']:.0f}s"
        )

        key = row["val_agree_nontrivial"]
        if not math.isnan(key) and key > best_key:
            best_key, best_epoch, best_val, bad_epochs = key, epoch, val, 0
            save_checkpoint(
                run_dir / "best.pt",
                model,
                {
                    "epoch": epoch,
                    "val": val,
                    "train_config": cfg.as_dict(),
                    "data_hash": train_data.data_hash,
                },
            )
        else:
            bad_epochs += 1
        if math.isnan(key) and epoch == cfg.epochs - 1:
            save_checkpoint(run_dir / "best.pt", model, {"epoch": epoch, "val": {}})
            best_epoch = epoch
        if stop or bad_epochs >= cfg.patience:
            break

    wall = time.perf_counter() - t0
    _write_history(run_dir, history)
    run_info = {
        "name": cfg.name,
        "seed": cfg.seed,
        "git_sha": get_git_sha(),
        "torch_version": torch.__version__,
        "device": str(device),
        "feature_version": FEATURE_VERSION,
        "schema_hash": SCHEMA_HASH,
        "train_data_hash": train_data.data_hash,
        "val_data_hash": val_data.data_hash,
        "train_decisions": len(train_data),
        "val_decisions": len(val_data),
        "param_count": count_parameters(model),
        "steps": step,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "wall_time_s": round(wall, 1),
        "config": cfg.as_dict(),
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2, default=str) + "\n")
    _plot_curves(run_dir, history)
    log(f"[train] best epoch {best_epoch}: {best_key:.2f}% -> {run_dir / 'best.pt'}")
    return TrainResult(run_dir, best_epoch, best_val, history, step, wall, first_losses)


def _write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    cols = list(dict.fromkeys(k for row in history for k in row))
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(history)


def _plot_curves(run_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in ("train_loss", "train_list", "train_reg", "train_rank", "train_aux", "train_pair"):
        if k in history[0]:
            a1.plot(ep, [r[k] for r in history], label=k.removeprefix("train_"))
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss")
    a1.set_title("Training loss")
    a1.legend(fontsize=8)
    for k, lab in (
        ("val_agree_nontrivial", "non-trivial"),
        ("val_agree_nontrivial_tie", "non-trivial (tie-aware)"),
        ("val_agree_hard", "hard"),
    ):
        a2.plot(ep, [r[k] for r in history], marker="o", label=lab)
    a2.axhline(85.0, color="grey", ls="--", lw=1, label="85 % gate")
    a2.set_xlabel("epoch")
    a2.set_ylabel("agreement with teacher (%)")
    a2.set_title("Validation agreement")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)


def train_from_config(
    cfg: TrainConfig, out_dir: Path | str | None = None, log: Log = print
) -> TrainResult:
    """Load the configured data and train."""
    log(f"[train] loading data {cfg.data_train} / {cfg.data_val}")
    tr = load_decisions(cfg.data_train, "train", cfg.max_train_decisions, cfg.seed)
    tr = filter_trivial(tr, cfg.trivial_keep_fraction, cfg.seed)
    va = load_decisions(cfg.data_val, "val", cfg.max_val_decisions, cfg.seed)
    va = va.take(np.flatnonzero(va.winner >= 0))
    return train(cfg, tr, va, out_dir, log)


def fine_tune(
    cfg: TrainConfig, ckpt: Path | str, out_dir: Path | str | None = None, log: Log = print
) -> TrainResult:
    """Continue training from ``ckpt`` on ``cfg``'s data (the DAgger retraining step)."""
    _, state = load_checkpoint(ckpt)
    log(f"[train] fine-tuning {ckpt} on {cfg.data_train}")
    tr = load_decisions(cfg.data_train, "train", cfg.max_train_decisions, cfg.seed)
    tr = filter_trivial(tr, cfg.trivial_keep_fraction, cfg.seed)
    va = load_decisions(cfg.data_val, "val", cfg.max_val_decisions, cfg.seed)
    va = va.take(np.flatnonzero(va.winner >= 0))
    return train(cfg, tr, va, out_dir, log, init_state=state["state_dict"])
