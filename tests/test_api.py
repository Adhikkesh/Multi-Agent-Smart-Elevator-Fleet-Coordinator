"""REST and WebSocket smoke tests for the dashboard server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from elevator_mas.api.server import create_app


@pytest.fixture
def client():
    """A test client with the server's lifespan running."""
    with TestClient(create_app("demo_story")) as test_client:
        yield test_client


class TestStateEndpoints:
    """Reading the world."""

    def test_health(self, client) -> None:
        """Liveness."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_state_has_everything_the_dashboard_draws(self, client) -> None:
        """A missing key would leave a panel blank."""
        payload = client.get("/api/state").json()
        for key in (
            "tick",
            "cars",
            "floors",
            "metrics",
            "messages",
            "rules",
            "traffic",
            "building",
            "session",
        ):
            assert key in payload, f"missing {key!r}"
        assert len(payload["cars"]) == payload["building"]["cars"]
        assert len(payload["floors"]) == payload["building"]["floors"]

    def test_meta_describes_the_theory_tab(self, client) -> None:
        """PEAS, environment classification, rules and strategies."""
        payload = client.get("/api/meta").json()
        assert len(payload["scenarios"]) == 9
        assert len(payload["strategies"]) == 10
        assert [s["bidder"] for s in payload["strategies"]].count("learned") == 6
        assert len(payload["agents"]) == 6
        assert len(payload["environment"]) == 7
        assert len(payload["rules"]) == 7
        for agent in payload["agents"]:
            assert set(agent["peas"]) == {"performance", "environment", "actuators", "sensors"}

    def test_index_and_static_assets_are_served(self, client) -> None:
        """The dashboard must work offline, including the vendored chart library."""
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/vendor/chart.umd.min.js").status_code == 200


class TestControls:
    """Driving the simulation."""

    def test_step_advances_the_tick(self, client) -> None:
        """Stepping while paused."""
        before = client.get("/api/state").json()["tick"]
        client.post("/api/step", json={"ticks": 5})
        assert client.get("/api/state").json()["tick"] == before + 5

    def test_pause_and_play(self, client) -> None:
        """The session flag must follow the control."""
        assert client.post("/api/play").json()["running"] is True
        assert client.post("/api/pause").json()["running"] is False

    def test_speed_is_applied(self, client) -> None:
        """Playback multiplier."""
        assert client.post("/api/speed", json={"speed": 25}).json()["speed"] == 25.0

    def test_speed_is_validated(self, client) -> None:
        """Out-of-range values must be rejected, not clamped silently."""
        assert client.post("/api/speed", json={"speed": 500}).status_code == 422

    def test_reset_changes_scenario_and_strategy(self, client) -> None:
        """Everything the UI can change must round-trip."""
        payload = client.post(
            "/api/reset",
            json={"scenario": "morning_up_peak", "strategy": "cnp_astar", "seed": 99},
        ).json()
        assert payload["scenario"] == "morning_up_peak"
        assert payload["strategy"] == "cnp_astar"
        assert payload["seed"] == 99
        assert payload["tick"] == 0

    def test_reset_resizes_the_building(self, client) -> None:
        """Floors and cars are configuration, not code."""
        payload = client.post("/api/reset", json={"floors": 22, "cars": 6}).json()
        assert payload["building"]["floors"] == 22
        assert payload["building"]["cars"] == 6
        assert len(payload["cars"]) == 6

    def test_unknown_scenario_is_rejected(self, client) -> None:
        """A bad name must 400, not 500."""
        assert client.post("/api/reset", json={"scenario": "nope"}).status_code == 400

    def test_unknown_strategy_is_rejected(self, client) -> None:
        """Likewise for strategies."""
        assert client.post("/api/reset", json={"strategy": "nope"}).status_code == 400


class TestInteraction:
    """Clicking the building and injecting disturbances."""

    def test_adding_a_passenger(self, client) -> None:
        """Clicking a floor."""
        payload = client.post("/api/passenger", json={"origin": 3}).json()
        assert payload["origin"] == 3
        assert payload["destination"] != 3

    def test_adding_a_priority_passenger(self, client) -> None:
        """Shift-clicking a floor."""
        assert client.post("/api/passenger", json={"origin": 2, "priority": True}).json()[
            "priority"
        ]

    def test_passenger_floor_is_validated(self, client) -> None:
        """A floor outside the building must 400."""
        assert client.post("/api/passenger", json={"origin": 999}).status_code == 400

    @pytest.mark.parametrize(
        "kind", ["car_fault", "car_repair", "fire_alarm", "fire_clear", "rush"]
    )
    def test_every_injection_kind_works(self, client, kind: str) -> None:
        """All five dashboard buttons."""
        response = client.post("/api/inject", json={"kind": kind, "car": 0, "floor": 0})
        assert response.status_code == 200
        assert response.json()["kind"] == kind

    def test_unknown_injection_is_rejected(self, client) -> None:
        """A typo must 400."""
        assert client.post("/api/inject", json={"kind": "earthquake"}).status_code == 400

    def test_fire_alarm_shows_in_the_state(self, client) -> None:
        """The UI banner reads this flag."""
        client.post("/api/inject", json={"kind": "fire_alarm"})
        assert client.get("/api/state").json()["fire_alarm"] is True


class TestInspection:
    """The agent inspector and the logs."""

    @pytest.mark.parametrize("address", ["car-0", "dispatcher", "monitor", "safety", "floor-0"])
    def test_every_agent_can_be_inspected(self, client, address: str) -> None:
        """Each agent must report its type and PEAS."""
        payload = client.get(f"/api/agent/{address}").json()
        assert payload["address"] == address
        assert payload["agent_type"]
        assert set(payload["peas"]) == {"performance", "environment", "actuators", "sensors"}

    def test_unknown_agent_is_404(self, client) -> None:
        """A missing agent must not 500."""
        assert client.get("/api/agent/car-999").status_code == 404

    def test_message_log(self, client) -> None:
        """The protocol log the examiner reads."""
        client.post("/api/step", json={"ticks": 60})
        payload = client.get("/api/messages").json()
        assert payload["total"] > 0
        for message in payload["messages"]:
            assert {"performative", "sender", "receiver", "conversation_id", "tick"} <= set(message)

    def test_auction_history(self, client) -> None:
        """The auction panel."""
        client.post("/api/step", json={"ticks": 120})
        auctions = client.get("/api/auctions").json()["auctions"]
        if auctions:
            assert {"floor", "direction", "bids", "winner"} <= set(auctions[0])


class TestSearchLab:
    """The three Search Lab panels."""

    def test_search_comparison_runs_all_four_algorithms(self, client) -> None:
        """BFS, UCS, Greedy and A* on one instance."""
        client.post("/api/step", json={"ticks": 100})
        payload = client.get("/api/search-lab").json()
        assert {r["algorithm"] for r in payload["results"]} == {"bfs", "ucs", "greedy", "astar"}
        found = [r for r in payload["results"] if r["found"]]
        if found:
            optimal = min(r["cost"] for r in found)
            astar = next(r for r in payload["results"] if r["algorithm"] == "astar")
            ucs = next(r for r in payload["results"] if r["algorithm"] == "ucs")
            assert astar["cost"] == pytest.approx(optimal)
            assert astar["nodes_expanded"] <= ucs["nodes_expanded"]

    def test_annealing_panel_always_has_something_to_show(self, client) -> None:
        """It falls back to a constructed instance when traffic is thin."""
        payload = client.get("/api/search-lab/annealing").json()
        assert payload["available"]
        assert (
            payload["simulated_annealing"]["final_cost"]
            <= payload["simulated_annealing"]["initial_cost"]
        )

    def test_minimax_panel_reports_pruning(self, client) -> None:
        """Alpha-beta must match the value with no more nodes."""
        payload = client.get("/api/search-lab/minimax").json()
        assert payload["nodes_alphabeta"] <= payload["nodes_minimax"]
        assert "best_parking" in payload


class TestBenchmarkEndpoint:
    """The benchmark tab."""

    def test_small_benchmark_returns_tables(self, client) -> None:
        """A tiny run, just to prove the plumbing."""
        payload = client.post(
            "/api/benchmark",
            json={
                "scenarios": ["interfloor_light"],
                "strategies": ["nearest_car", "full"],
                "seeds": 1,
                "ticks": 120,
            },
        ).json()
        assert payload["runs"] == 2
        assert len(payload["summary"]) == 2
        for row in payload["summary"]:
            assert "avg_wait_mean" in row
            assert row["scenario"] == "interfloor_light"

    def test_unknown_scenario_is_rejected(self, client) -> None:
        """Validation before doing expensive work."""
        assert client.post("/api/benchmark", json={"scenarios": ["nope"]}).status_code == 400


class TestWebSocket:
    """Streaming."""

    def test_socket_sends_a_snapshot_on_connect(self, client) -> None:
        """The dashboard paints from the first frame."""
        with client.websocket_connect("/ws") as socket:
            frame = socket.receive_json()
            assert "tick" in frame
            assert "cars" in frame
            assert "metrics" in frame

    def test_socket_streams_further_frames(self, client) -> None:
        """Stepping must push a new frame to connected clients."""
        with client.websocket_connect("/ws") as socket:
            first = socket.receive_json()
            client.post("/api/step", json={"ticks": 3})
            second = socket.receive_json()
            assert second["tick"] > first["tick"]


class TestStrictJson:
    """Browsers' JSON.parse rejects Infinity/NaN and Python objects that are not JSON."""

    def test_websocket_frames_are_strict_json(self) -> None:
        import json

        from fastapi.testclient import TestClient

        from elevator_mas.api.server import create_app

        app = create_app("demo_story")
        app.state.session.model.run(150)  # long enough for refused bids and calls in messages
        client = TestClient(app)
        with client.websocket_connect("/ws") as socket:
            text = socket.receive_text()

        def reject(constant: str) -> None:
            raise ValueError(f"non-strict JSON constant {constant}")

        frame = json.loads(text, parse_constant=reject)
        assert frame["tick"] == 150
        assert any(m["performative"] == "REQUEST" for m in frame["messages"]) or frame["messages"]


class TestPhase2Additions:
    """New endpoints and routing added for Phase 2."""

    def test_classic_dashboard_served(self, client) -> None:
        """GET /classic serves the original single-page HTML."""
        response = client.get("/classic")
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text
        assert "Smart Elevator Fleet Coordinator" in response.text

    def test_spa_fallback_and_deep_links(self, client) -> None:
        """Deep links like /lab and /agents return the dashboard page."""
        for path in ("/lab", "/agents", "/experiments", "/theory", "/story"):
            response = client.get(path)
            assert response.status_code == 200
            assert "<!doctype html>" in response.text.lower()

    def test_spa_fallback_does_not_mask_api_or_static_404s(self, client) -> None:
        """Nonexistent API and static paths return 404."""
        assert client.get("/api/unknown_endpoint").status_code == 404
        assert client.get("/static/unknown_file.js").status_code == 404

    def test_board_endpoint(self, client) -> None:
        """GET /api/board returns the public status blackboard."""
        import json

        def reject(constant: str) -> None:
            raise ValueError(f"non-strict JSON constant {constant}")

        response = client.get("/api/board")
        assert response.status_code == 200
        payload = json.loads(response.text, parse_constant=reject)
        assert "cars" in payload
        assert "policy" in payload
        assert "writes" in payload
        assert isinstance(payload["cars"], list)
        assert len(payload["cars"]) == 4

    def test_version_endpoint(self, client) -> None:
        """GET /api/version returns version info."""
        response = client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "2.0.0-dev"
        assert "git" in data

    def test_run_endpoint_success_and_determinism(self, client) -> None:
        """POST /api/run performs a fast headless run and returns metrics and series."""
        import json

        def reject(constant: str) -> None:
            raise ValueError(f"non-strict JSON constant {constant}")

        body = {
            "scenario": "interfloor_light",
            "strategy": "collective",
            "seed": 42,
            "ticks": 60,
            "sample_every": 10,
        }
        res1 = client.post("/api/run", json=body)
        assert res1.status_code == 200
        data1 = json.loads(res1.text, parse_constant=reject)

        assert data1["scenario"] == "interfloor_light"
        assert data1["strategy"] == "collective"
        assert data1["seed"] == 42
        assert data1["ticks"] == 60
        assert "runtime_s" in data1
        assert "final" in data1
        assert "series" in data1
        assert "rules_fired" in data1
        assert "violations" in data1

        series = data1["series"]
        for key in (
            "tick",
            "avg_wait",
            "p95_wait",
            "waiting",
            "riding",
            "delivered",
            "energy",
            "long_wait_pct",
            "messages",
        ):
            assert key in series
            assert len(series[key]) >= 6

        # Determinism check (metrics other than wall-clock compute time must match exactly)
        res2 = client.post("/api/run", json=body)
        assert res2.status_code == 200
        data2 = json.loads(res2.text, parse_constant=reject)
        f1 = {k: v for k, v in data1["final"].items() if k != "compute_ms_per_tick"}
        f2 = {k: v for k, v in data2["final"].items() if k != "compute_ms_per_tick"}
        assert f1 == f2
        assert data1["series"] == data2["series"]

    def test_run_endpoint_validation(self, client) -> None:
        """Validation errors on unknown scenario/strategy and out-of-range bounds."""
        res_bad_scenario = client.post(
            "/api/run", json={"scenario": "nonexistent", "strategy": "collective", "seed": 1}
        )
        assert res_bad_scenario.status_code == 400

        res_bad_strategy = client.post(
            "/api/run", json={"scenario": "interfloor_light", "strategy": "bad_strat", "seed": 1}
        )
        assert res_bad_strategy.status_code == 400

        res_bad_ticks = client.post(
            "/api/run",
            json={"scenario": "interfloor_light", "strategy": "collective", "seed": 1, "ticks": 10},
        )
        assert res_bad_ticks.status_code == 422
