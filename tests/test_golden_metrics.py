"""Golden-metrics regression for the four classical strategies.

`tests/golden/strategy_metrics.json` was generated *before* the LiftZero engine changes
(Phase 4). Every later engine change must leave these numbers bit-identical, which is the
proof that the learned bidders are purely additive.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((ROOT / "tests" / "golden" / "strategy_metrics.json").read_text())


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_golden_metrics", ROOT / "scripts" / "generate_golden_metrics.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _generator()


def test_golden_file_covers_the_matrix() -> None:
    assert len(GOLDEN) == len(GEN.STRATEGIES) * len(GEN.SCENARIOS) * len(GEN.SEEDS)


@pytest.mark.parametrize("strategy", GEN.STRATEGIES)
@pytest.mark.parametrize("scenario", GEN.SCENARIOS)
@pytest.mark.parametrize("seed", GEN.SEEDS)
def test_classical_strategy_metrics_unchanged(strategy: str, scenario: str, seed: int) -> None:
    expected = GOLDEN[GEN.golden_key(strategy, scenario, seed)]
    assert GEN.golden_metrics(strategy, scenario, seed) == expected
