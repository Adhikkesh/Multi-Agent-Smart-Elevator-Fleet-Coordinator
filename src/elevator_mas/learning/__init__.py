"""LiftZero Learning Environment package.

Exports feature schemas, encoders, decision recorder, expert dataset, fast twin simulator,
and Gymnasium environments.

Exports are resolved lazily (PEP 562): importing ``elevator_mas.learning`` — or a light
submodule such as ``learning.features`` or the ONNX runtime in ``learning.lift`` — does not
pull in Gymnasium, Numba or PyTorch. Those live in the optional ``learn`` / ``lift-train``
dependency groups and load only when the name that needs them is first used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "DatasetStats": "dataset",
    "ExpertDataset": "dataset",
    "ElevatorDecisionEnv": "env",
    "RewardConfig": "env",
    "VectorDecisionEnv": "env",
    "Policy": "evaluate",
    "evaluate": "evaluate",
    "summarise": "evaluate",
    "Encoded": "features",
    "EncodedBatch": "features",
    "encode_batch": "features",
    "encode_decision": "features",
    "DecisionRecorder": "recorder",
    "record_expert_dataset": "recorder",
    "EvalRegime": "regimes",
    "RegimesConfig": "regimes",
    "load_regimes_config": "regimes",
    "FEATURE_NAMES": "schema",
    "FEATURE_VERSION": "schema",
    "KC": "schema",
    "KCAR": "schema",
    "KG": "schema",
    "MAX_CARS": "schema",
    "SCHEMA_HASH": "schema",
    "TwinSimulator": "twin.state",
    "DecisionContext": "view",
    "FleetView": "view",
    "from_board": "view",
    "from_event": "view",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
