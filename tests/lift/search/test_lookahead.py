"""Phase 6 look-ahead: PUCT selection, determinism, budget, safety, Brain API."""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from elevator_mas.api.server import create_app
from elevator_mas.config import LiftConfig, ScenarioConfig
from elevator_mas.learning.lift.search.mcts import Node
from elevator_mas.model import ElevatorModel


def test_puct_prefers_the_prior_when_unvisited_and_value_after() -> None:
    node = Node([0, 1], np.array([0.2, 0.8]))
    assert node.select(1.25) == 1
    node.n[:] = [10, 10]
    node.w[:] = [-10.0, -50.0]  # car 0 is much better
    assert node.select(1.25) == 0


def _run(budget: float = 30.0, sims: int = 16, ticks: int = 250) -> ElevatorModel:
    cfg = ScenarioConfig.load("lunch_two_way").model_copy(
        update={
            "strategy": "liftzero_bc_mcts",
            "seed": 8003,
            "lift": LiftConfig(search_sims=sims, search_budget_ms=budget),
        }
    )
    m = ElevatorModel(cfg)
    for _ in range(ticks):
        m.step()
        assert m.violations() == []
    return m


def test_mcts_strategy_runs_safely_and_searches_contested_calls() -> None:
    m = _run()
    assert m.arbiter.searched > 0
    assert m.arbiter.fallbacks == 0
    searched = [d for d in m.brain.recent(200) if d["search"]]
    assert searched
    for d in searched:
        s = d["search"]
        assert d["chosen"] in s["candidates"]
        assert s["time_ms"] < 30.0 + 25.0  # anytime: one simulation past the budget at most
        refused = {c["car_id"] for c in d["candidates"] if c["refused"]}
        assert d["chosen"] not in refused


def test_search_is_deterministic_given_the_seed() -> None:
    a = [d["chosen"] for d in _run(budget=1000, sims=8, ticks=150).brain.recent(200)]
    b = [d["chosen"] for d in _run(budget=1000, sims=8, ticks=150).brain.recent(200)]
    assert a == b


def test_brain_api_is_strict_json_and_explains() -> None:
    client = TestClient(create_app("lunch_two_way"))
    assert client.get("/api/brain").json()["available"] is False  # demo strategy is classical
    client.post("/api/reset", json={"strategy": "liftzero_bc_mcts"})
    client.post("/api/brain/config", json={"sims": 8, "time_budget_ms": 20})
    client.post("/api/step", json={"ticks": 150})
    brain = client.get("/api/brain").json()
    json.dumps(brain, allow_nan=False)
    assert brain["available"] and brain["active"]["search"]["sims"] == 8
    decisions = client.get("/api/brain/decisions?limit=5").json()["decisions"]
    json.dumps(decisions, allow_nan=False)
    r = client.post("/api/brain/explain", json={"conversation_id": decisions[0]["conversation_id"]})
    assert r.status_code == 200 and len(r.json()["groups"]) == 9
    assert client.post("/api/brain/explain", json={"conversation_id": "nope"}).status_code == 404
    assert client.post("/api/brain/config", json={"sims": 9999}).status_code == 422
