"""Export a trained LiftZero checkpoint to ONNX and write its model card.

Uses PyTorch's ``torch.export``-based ONNX exporter (opset 18) with dynamic ``batch`` and
``n_cars`` axes. The legacy TorchScript exporter is deprecated in the installed PyTorch and
was measured to diverge from eager mode on this graph (~0.06 abs on scores), so it is not
used. Every export is verified: torch vs onnxruntime max-abs difference must be below
``PARITY_TOL`` on random decisions at N = 1, 2, 8, 16, 32 before the card is written
(over real cars; see :func:`max_abs_diff`).
"""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from elevator_mas.learning.lift.card import ModelCard, card_path_for, file_sha256, write_card
from elevator_mas.learning.lift.model import LiftZeroNet, count_parameters
from elevator_mas.learning.lift.train_bc import load_checkpoint
from elevator_mas.learning.recorder import get_git_sha
from elevator_mas.learning.schema import FEATURE_VERSION, SCHEMA_HASH

OPSET = 18
PARITY_TOL = 1e-4
INPUTS = ("call", "cars", "glob", "mask", "eligible")
OUTPUTS = ("score", "aux", "value", "attn")

DEFAULT_LIMITATIONS = [
    "Routing (stop sequencing) is still the classical A* planner; only the bid is learned.",
    "Eligibility (out of service, fire mode, full) is decided by the classical rules and "
    "never by the network.",
    "Trained on simulated buildings of 6-32 floors and 2-8 cars from one building family "
    "(single lobby at floor 0); 33-40 floors x 7-8 cars is held out as test_large.",
    "Priority-passenger weights are not on the public board, so the network cannot see "
    "them; the teacher can.",
    "Imitation only: the network can at best match the A* teacher (Phase 5 adds RL).",
]


class _Graph(nn.Module):
    """Tuple-output wrapper (the exporter needs plain tensors, not a dataclass)."""

    def __init__(self, net: LiftZeroNet) -> None:
        super().__init__()
        self.net = net

    def forward(
        self,
        call: torch.Tensor,
        cars: torch.Tensor,
        glob: torch.Tensor,
        mask: torch.Tensor,
        eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(call, cars, glob, mask, eligible)
        return out.score, out.aux, out.value, out.attn


def random_inputs(
    batch: int, n: int, gen: torch.Generator, kc: int, kcar: int, kg: int
) -> tuple[torch.Tensor, ...]:
    """Valid-looking random decisions: car 0 always exists and is eligible."""
    mask = torch.rand(batch, n, generator=gen) > 0.2
    elig = torch.rand(batch, n, generator=gen) > 0.25
    mask[:, 0] = True
    elig[:, 0] = True
    return (
        torch.rand(batch, kc, generator=gen),
        torch.rand(batch, n, kcar, generator=gen),
        torch.rand(batch, kg, generator=gen),
        mask,
        elig & mask,
    )


def export_model(net: LiftZeroNet, out_path: Path | str) -> Path:
    """Write the ONNX graph of ``net`` (eval mode) with dynamic batch and fleet axes."""
    from torch.export import Dim

    net = net.eval()
    cfg = net.cfg
    gen = torch.Generator().manual_seed(0)
    example = random_inputs(2, 4, gen, cfg.kc, cfg.kcar, cfg.kg)
    b = Dim("batch", min=1, max=65536)
    n = Dim("n_cars", min=1, max=256)
    shapes = {
        "call": {0: b},
        "cars": {0: b, 1: n},
        "glob": {0: b},
        "mask": {0: b, 1: n},
        "eligible": {0: b, 1: n},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    quiet = [logging.getLogger(name) for name in ("torch.onnx", "torch.export", "onnxscript")]
    levels = [lg.level for lg in quiet]
    for lg in quiet:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                _Graph(net),
                example,
                str(out),
                input_names=list(INPUTS),
                output_names=list(OUTPUTS),
                dynamic_shapes=shapes,
                opset_version=OPSET,
                dynamo=True,
                external_data=False,
                verbose=False,
            )
    finally:
        for lg, lvl in zip(quiet, levels, strict=True):
            lg.setLevel(lvl)
    return out


def parity(
    net: LiftZeroNet,
    onnx_path: Path | str,
    sizes: tuple[int, ...] = (1, 2, 8, 16, 32),
    batch: int = 64,
    seed: int = 0,
) -> float:
    """Max-abs torch-vs-onnxruntime difference over all outputs and the given fleet sizes."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(seed)
    cfg = net.cfg
    worst = 0.0
    net = net.eval()
    with torch.no_grad():
        for n in sizes:
            args = random_inputs(batch, n, gen, cfg.kc, cfg.kcar, cfg.kg)
            ref = [t.numpy() for t in _Graph(net)(*args)]
            got = sess.run(None, {k: v.numpy() for k, v in zip(INPUTS, args, strict=True)})
            worst = max(worst, max_abs_diff(ref, got, args[3].numpy(), args[4].numpy()))
    return worst


def max_abs_diff(
    ref: list[np.ndarray], got: list[np.ndarray], mask: np.ndarray, eligible: np.ndarray
) -> float:
    """Worst |torch - onnx| over the outputs that carry information.

    Scores of padded/ineligible cars are ``raw + 1e4`` sentinels; at that magnitude a float32
    ulp is ~1e-3, so their low bits are compared with a relative tolerance by the tests and
    excluded here. aux/attn of padded cars are don't-care values.
    """
    valid = mask & eligible
    diffs = [
        np.abs(ref[0] - got[0])[valid],
        np.abs(ref[1] - got[1])[mask],
        np.abs(ref[2] - got[2]).ravel(),
        np.abs(ref[3] - got[3])[mask],
    ]
    return max(float(d.max()) for d in diffs if d.size)


def export_checkpoint(
    ckpt_path: Path | str,
    out_path: Path | str,
    name: str | None = None,
    version: str = "1",
    metrics: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
) -> ModelCard:
    """Export a checkpoint to ONNX, verify parity, and write the model card next to it."""
    net, ckpt = load_checkpoint(ckpt_path)
    out = export_model(net, out_path)
    worst = parity(net, out)
    if worst >= PARITY_TOL:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX parity check failed: max abs diff {worst:.2e} >= {PARITY_TOL}")
    train_cfg = ckpt.get("train_config", {})
    card = ModelCard(
        name=Path(out).stem,
        version=version,
        feature_version=FEATURE_VERSION,
        schema_hash=SCHEMA_HASH,
        param_count=count_parameters(net),
        logit_temp=float(net.logit_temp.detach()),
        model_config=dict(ckpt["model_config"]),
        opset=OPSET,
        onnx_sha256=file_sha256(out),
        git_sha=get_git_sha(),
        created=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        training={
            "checkpoint": str(ckpt_path),
            "epoch": ckpt.get("epoch"),
            "seed": train_cfg.get("seed"),
            "preset": train_cfg.get("name"),
            "data_train": train_cfg.get("data_train"),
            "train_data_hash": ckpt.get("data_hash"),
            "splits": "by run seed: train 0-7999, val 8000-8499, test 8500+, "
            "test_large = 33-40 floors x 7-8 cars",
            **(training or {}),
        },
        metrics={"val": ckpt.get("val", {}), "onnx_parity_max_abs": worst, **(metrics or {})},
        limitations=list(DEFAULT_LIMITATIONS),
    )
    if name:
        card.name = name
    write_card(card, card_path_for(out))
    return card
