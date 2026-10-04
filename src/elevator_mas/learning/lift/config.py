"""Configuration dataclasses for the LiftZero network and its imitation training.

Everything a training run depends on is a field here and can be set from YAML
(``configs/learning/bc*.yaml``); nothing is hard-coded in the trainer. This module imports
neither torch nor onnx, so the runtime can read model cards and configs without them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from elevator_mas.learning.schema import KC, KCAR, KG

ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "configs" / "learning"


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the set-Transformer (defaults give ~0.24 M parameters)."""

    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.05
    kc: int = KC
    kcar: int = KCAR
    kg: int = KG


@dataclass(frozen=True)
class LossWeights:
    """Weights of the five imitation-loss terms (see ``losses.py``)."""

    listwise: float = 1.0
    regression: float = 0.5
    rank: float = 0.5
    aux: float = 0.2
    pairwise: float = 0.3
    pair_margin: float = 0.1
    hard_pair_weight: float = 2.0


@dataclass(frozen=True)
class AugmentConfig:
    """On-the-fly batch augmentation (see ``augment.py``)."""

    permute: bool = True
    car_dropout_p: float = 0.3
    car_dropout_max: int = 3
    feature_noise: float = 0.01


@dataclass(frozen=True)
class TrainConfig:
    """One imitation-training run."""

    name: str = "default"
    data_train: list[str] = field(default_factory=lambda: ["data/expert/train"])
    data_val: list[str] = field(default_factory=lambda: ["data/expert/val"])
    max_train_decisions: int | None = None
    max_val_decisions: int | None = None
    trivial_keep_fraction: float = 0.05
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)
    warmup_steps: int = 500
    grad_clip: float = 1.0
    patience: int = 4
    device: str = "auto"
    seed: int = 0
    num_threads: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, written into ``run.json``."""
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> TrainConfig:
        """A copy with top-level fields replaced (``None`` values are ignored)."""
        data = self.as_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return train_config_from_dict(data)


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Build a :class:`TrainConfig` from a nested mapping, rejecting unknown keys."""
    data = dict(data)
    model = _build(ModelConfig, data.pop("model", None))
    loss = _build(LossWeights, data.pop("loss", None))
    augment = _build(AugmentConfig, data.pop("augment", None))
    if "betas" in data:
        data["betas"] = tuple(data["betas"])
    cfg = _build(TrainConfig, data)
    return TrainConfig(**{**asdict(cfg), "model": model, "loss": loss, "augment": augment})


def model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    """Build a :class:`ModelConfig` from a mapping (e.g. a checkpoint's stored config)."""
    return _build(ModelConfig, data)


def load_train_config(path: Path | str | None = None, preset: str | None = None) -> TrainConfig:
    """Load a training config: ``preset`` names ``configs/learning/bc_<preset>.yaml``.

    ``default`` maps to ``configs/learning/bc.yaml``. An explicit ``path`` wins.
    """
    if path is None:
        name = "bc.yaml" if preset in (None, "default") else f"bc_{preset}.yaml"
        path = CONFIG_DIR / name
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"training config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return train_config_from_dict(data)
