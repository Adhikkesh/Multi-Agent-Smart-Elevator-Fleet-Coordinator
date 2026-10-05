"""Shared fixtures for the LiftZero tests.

Torch-dependent fixtures skip cleanly when the ``lift-train`` group is not installed; the
runtime/integration tests use the committed ONNX model and need only onnxruntime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "learning" / "fixtures" / "expert_sample.npz"
SHIPPED_MODEL = ROOT / "models" / "liftzero_bc_v1.onnx"


@pytest.fixture(scope="session")
def fixture_decisions() -> Any:
    pytest.importorskip("torch")
    from elevator_mas.learning.dataset import ExpertDataset
    from elevator_mas.learning.lift.train_bc import Decisions

    return Decisions.from_dataset(ExpertDataset(FIXTURE))


@pytest.fixture(scope="session")
def tiny_cfg() -> Any:
    from elevator_mas.learning.lift.config import ModelConfig

    return ModelConfig(d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0)


@pytest.fixture()
def tiny_net(tiny_cfg: Any) -> Any:
    torch = pytest.importorskip("torch")
    from elevator_mas.learning.lift.model import LiftZeroNet

    torch.manual_seed(0)
    return LiftZeroNet(tiny_cfg).eval()


@pytest.fixture(scope="session")
def tiny_export(tmp_path_factory: pytest.TempPathFactory, tiny_cfg: Any) -> tuple[Any, Path]:
    """A tiny random LiftZeroNet, its checkpoint and its exported ONNX + card."""
    torch = pytest.importorskip("torch")
    from elevator_mas.learning.lift.export import export_checkpoint
    from elevator_mas.learning.lift.model import LiftZeroNet
    from elevator_mas.learning.lift.train_bc import save_checkpoint

    torch.manual_seed(1)
    net = LiftZeroNet(tiny_cfg).eval()
    d = tmp_path_factory.mktemp("tiny")
    save_checkpoint(d / "tiny.pt", net, {"epoch": 0, "val": {}, "train_config": {}})
    export_checkpoint(d / "tiny.pt", d / "tiny.onnx", version="test")
    return net, d / "tiny.onnx"


@pytest.fixture(scope="session")
def shipped_model() -> Path:
    if not SHIPPED_MODEL.exists():
        pytest.fail(f"the shipped model {SHIPPED_MODEL} is missing (run the Phase 4 pipeline)")
    return SHIPPED_MODEL
