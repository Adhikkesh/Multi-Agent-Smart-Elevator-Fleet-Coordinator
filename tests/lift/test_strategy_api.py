"""Registry entries, /api/meta availability and strict-JSON frames for learned strategies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elevator_mas.api.server import create_app, strategy_meta
from elevator_mas.config import ScenarioConfig
from elevator_mas.learning.lift.bidder import StrategyUnavailableError, strategy_availability
from elevator_mas.model import ElevatorModel
from elevator_mas.strategies import STRATEGIES, DispatchStrategy, get_strategy


def test_learned_strategies_are_registered_as_single_variable_changes() -> None:
    bc, full = get_strategy("liftzero_bc"), get_strategy("full")
    bc_cnp, cnp = get_strategy("liftzero_bc_cnp"), get_strategy("cnp_astar")
    for learned, teacher in ((bc, full), (bc_cnp, cnp)):
        assert learned.bidder == "learned" and teacher.bidder == "classical"
        assert learned.teacher == teacher.name
        for f in ("assignment", "routing", "reassignment", "parking_policy", "adapts_weights"):
            assert getattr(learned, f) == getattr(teacher, f), f


def test_classical_strategies_are_untouched() -> None:
    for name in ("nearest_car", "collective", "cnp_astar", "full"):
        s = STRATEGIES[name]
        assert s.bidder == "classical" and not s.uses_learned_bidder
        assert strategy_availability(s) == (True, "")


def test_meta_lists_bidder_and_availability(shipped_model: Path) -> None:
    client = TestClient(create_app("morning_up_peak"))
    meta = client.get("/api/meta").json()
    by_name = {s["name"]: s for s in meta["strategies"]}
    assert {"liftzero_bc", "liftzero_bc_cnp"} <= set(by_name)
    assert by_name["liftzero_bc"]["bidder"] == "learned"
    assert by_name["liftzero_bc"]["available"] is True
    assert by_name["full"]["bidder"] == "classical"
    assert by_name["full"]["learning"] is False


def test_missing_model_marks_the_strategy_unavailable(tmp_path: Path) -> None:
    ghost = DispatchStrategy(
        name="ghost",
        label="ghost",
        description="",
        bidder="learned",
        model_path=str(tmp_path / "missing.onnx"),
    )
    entry = strategy_meta(ghost)
    assert entry["available"] is False
    assert "not found" in entry["reason"]


def test_model_construction_fails_loudly_without_a_model(tmp_path: Path) -> None:
    from elevator_mas.config import LiftConfig

    cfg = ScenarioConfig.load("morning_up_peak").model_copy(
        update={
            "strategy": "liftzero_bc",
            "lift": LiftConfig(model_path=str(tmp_path / "missing.onnx")),
        }
    )
    with pytest.raises(StrategyUnavailableError):
        ElevatorModel(cfg)


def test_reset_to_learned_strategy_and_frames_are_strict_json(shipped_model: Path) -> None:
    client = TestClient(create_app("morning_up_peak"))
    snap = client.post("/api/reset", json={"strategy": "liftzero_bc"}).json()
    assert snap["strategy"] == "liftzero_bc"
    for _ in range(3):
        client.post("/api/step", json={"ticks": 40})
    for path in ("/api/state", "/api/auctions", "/api/board"):
        body = client.get(path).json()
        json.dumps(body, allow_nan=False)  # raises on inf/nan
    auctions = client.get("/api/auctions").json()["auctions"]
    assert any("[LiftZero]" in (a.get("reason") or "") for a in auctions)
