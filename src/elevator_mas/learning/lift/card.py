"""The LiftZero model card: provenance and contract of a shipped ONNX model.

The card is a JSON file next to the ONNX graph (``liftzero_bc_v1.onnx`` ->
``liftzero_bc_v1.json``). The runtime refuses to load a model whose card does not match the
feature schema compiled into this code — a model trained on other features would silently
produce garbage bids otherwise. Imports neither torch nor onnx.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from elevator_mas.learning.schema import FEATURE_VERSION, SCHEMA_HASH


class ModelCardError(RuntimeError):
    """The model or its card is missing, corrupt, or incompatible with this code."""


@dataclass
class ModelCard:
    """Everything needed to trust (or refuse) a LiftZero model file."""

    name: str
    version: str
    feature_version: int
    schema_hash: str
    param_count: int
    logit_temp: float
    model_config: dict[str, Any]
    opset: int
    onnx_sha256: str
    git_sha: str
    created: str
    inputs: list[str] = field(default_factory=lambda: ["call", "cars", "glob", "mask", "eligible"])
    outputs: list[str] = field(default_factory=lambda: ["score", "aux", "value", "attn"])
    training: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    intended_use: str = (
        "Pricing hall calls inside each car's Contract Net bid in the elevator_mas "
        "simulator (strategies liftzero_bc / liftzero_bc_cnp). Research and teaching only."
    )
    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def card_path_for(model_path: Path | str) -> Path:
    """The card that belongs to a model file."""
    return Path(model_path).with_suffix(".json")


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_card(card: ModelCard, path: Path | str) -> Path:
    p = Path(path)
    p.write_text(json.dumps(card.as_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return p


def load_card(model_path: Path | str) -> ModelCard:
    """Read the card next to ``model_path`` (no compatibility check; see validate_card)."""
    p = card_path_for(model_path)
    if not p.exists():
        raise ModelCardError(f"model card not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ModelCard(**data)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ModelCardError(f"corrupt model card {p}: {exc}") from exc


def validate_card(card: ModelCard, model_path: Path | str | None = None) -> None:
    """Raise :class:`ModelCardError` unless the model matches this code's feature schema."""
    if card.feature_version != FEATURE_VERSION or card.schema_hash != SCHEMA_HASH:
        raise ModelCardError(
            f"model {card.name} v{card.version} was trained on feature schema "
            f"v{card.feature_version} ({card.schema_hash}); this code uses "
            f"v{FEATURE_VERSION} ({SCHEMA_HASH}). Retrain or export a matching model."
        )
    if model_path is not None and card.onnx_sha256:
        actual = file_sha256(model_path)
        if actual != card.onnx_sha256:
            raise ModelCardError(
                f"{model_path} does not match its card (sha256 {actual[:12]} != "
                f"{card.onnx_sha256[:12]}); re-export the model"
            )


def render_markdown(card: ModelCard) -> str:
    """The card as Markdown (``docs/lift/MODEL_CARD.md``)."""
    lines = [
        f"# Model card — {card.name} v{card.version}",
        "",
        f"*Generated from `models/{card.name}.json` on {card.created}; git `{card.git_sha}`.*",
        "",
        "## Contract",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Feature schema | v{card.feature_version} (`{card.schema_hash}`) |",
        f"| Parameters | {card.param_count:,} |",
        f"| ONNX opset | {card.opset} |",
        f"| Inputs | {', '.join(f'`{i}`' for i in card.inputs)} |",
        f"| Outputs | {', '.join(f'`{o}`' for o in card.outputs)} |",
        f"| Policy temperature (log) | {card.logit_temp:.4f} |",
        f"| ONNX sha256 | `{card.onnx_sha256[:16]}…` |",
        "",
        "## Architecture",
        "",
        "| Hyper-parameter | Value |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in card.model_config.items()],
        "",
        "## Intended use",
        "",
        card.intended_use,
        "",
        "## Training",
        "",
        "```json",
        json.dumps(card.training, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(card.metrics, indent=2),
        "```",
        "",
        "## Limitations",
        "",
        *[f"- {lim}" for lim in card.limitations],
        "",
    ]
    return "\n".join(lines)
