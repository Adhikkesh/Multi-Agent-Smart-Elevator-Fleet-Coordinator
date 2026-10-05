"""The learned bidder inside the multi-agent system: public data only, modes, shadow, DAgger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from lift_helpers import make_status

from elevator_mas.comms.board import DecisionEvent, FleetPolicy
from elevator_mas.config import BuildingConfig, LiftConfig, ScenarioConfig, TimingConfig
from elevator_mas.domain import Direction, HallCall
from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.lift.bidder import LearnedBidder
from elevator_mas.learning.view import from_event
from elevator_mas.model import ElevatorModel


class Strict:
    """Exposes only the given attributes; reading anything else fails the test."""

    def __init__(self, **allowed: Any) -> None:
        self.__dict__.update(allowed)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"learned bidder read non-public attribute {name!r}")


def run_model(
    strategy: str = "liftzero_bc",
    scenario: str = "morning_up_peak",
    seed: int = 42,
    ticks: int = 300,
    **lift: Any,
) -> tuple[ElevatorModel, list[DecisionEvent]]:
    cfg = ScenarioConfig.load(scenario).model_copy(
        update={"strategy": strategy, "seed": seed, "lift": LiftConfig(**lift)}
    )
    model = ElevatorModel(cfg)
    events: list[DecisionEvent] = []
    model.decision_hooks.append(events.append)
    for _ in range(ticks):
        model.step()
    return model, events


def test_bidder_reads_only_the_public_board(shipped_model: Path) -> None:
    statuses = [
        make_status(0, 2),
        make_status(1, 9, Direction.DOWN, load=4),
        make_status(2, 5, Direction.UP, load=9),
    ]
    by_id = {s.car_id: s for s in statuses}
    board = Strict(cars=lambda: list(statuses), car=by_id.get, policy=FleetPolicy())
    config = Strict(building=BuildingConfig(floors=12, cars=3), timing=TimingConfig())
    stub = Strict(board=board, config=config, tick=17)
    bidder = LearnedBidder(stub, model_path=str(shipped_model))
    call = HallCall(floor=6, direction=Direction.UP)
    bids = [bidder.bid(i, call, urgency=1, waiting=2, conversation_id="c1") for i in range(3)]
    assert [b.car_id for b in bids] == [0, 1, 2]
    for b in bids:
        assert np.isfinite(b.total) and 0.0 <= b.total <= 1e5
        assert not b.refused
        assert min(b.wait, b.ride, b.crowding, b.energy) >= 0.0
    assert bidder.forward_passes == 1  # shared mode: one pass per CFP


def test_shared_and_per_car_inference_give_identical_runs(shipped_model: Path) -> None:
    shared, ev_s = run_model(inference="shared")
    per_car, ev_p = run_model(inference="per_car")
    assert len(ev_s) == len(ev_p) > 0
    for a, b in zip(ev_s, ev_p, strict=True):
        assert a.winner == b.winner
        assert [x.total for x in a.bids] == [x.total for x in b.bids]
    da, db = shared.latest_metrics.as_dict(), per_car.latest_metrics.as_dict()
    da.pop("compute_ms_per_tick")
    db.pop("compute_ms_per_tick")
    assert da == db
    assert per_car.learned_bidder.forward_passes > shared.learned_bidder.forward_passes


def test_live_features_equal_the_recorded_announce_time_features(shipped_model: Path) -> None:
    """Pitfall 2: the bidder's board read at CFP time == the DecisionEvent snapshot."""
    cfg = ScenarioConfig.load("lunch_two_way").model_copy(
        update={"strategy": "liftzero_bc", "seed": 7}
    )
    model = ElevatorModel(cfg)
    live: dict[tuple[int, int, int], Any] = {}
    original = model.learned_bidder._price

    def spy(call: HallCall, urgency: int, waiting: int) -> Any:
        from elevator_mas.learning.view import from_board

        ctx = from_board(
            call.floor,
            1 if call.direction is Direction.UP else -1,
            waiting,
            urgency,
            list(model.board.cars()),
            model.board.policy,
            model.config.building,
            model.tick,
        )
        live[(model.tick, call.floor, call.direction.sign)] = encode_decision(ctx)
        return original(call, urgency, waiting)

    model.learned_bidder._price = spy  # type: ignore[method-assign]
    events: list[DecisionEvent] = []
    model.decision_hooks.append(events.append)
    for _ in range(300):
        model.step()
    checked = 0
    for ev in events:
        key = (ev.tick, ev.call.floor, ev.call.direction.sign)
        if key not in live:
            continue  # every car refused: the network was never consulted
        rec = encode_decision(from_event(ev))
        for field in ("call", "cars", "glob", "mask", "eligible"):
            np.testing.assert_array_equal(getattr(rec, field), getattr(live[key], field))
        checked += 1
    assert checked > 20


def test_shadow_fields_only_when_enabled(shipped_model: Path) -> None:
    _, plain = run_model(ticks=200)
    assert plain and all(e.shadow_bids is None and e.bidder == "learned" for e in plain)
    model, shadowed = run_model(ticks=200, shadow_teacher=True)
    assert shadowed and all(e.shadow_bids is not None for e in shadowed)
    for e in shadowed:
        assert [b.car_id for b in e.shadow_bids] == [b.car_id for b in e.bids]
        # Eligibility is classical in both: the same cars refuse.
        assert [b.refused for b in e.shadow_bids] == [b.refused for b in e.bids]
    assert any("[LiftZero]" in a.reason for a in model.dispatcher.auction_history)
    assert any("teacher" in a.reason for a in model.dispatcher.auction_history)


def test_shadow_teacher_does_not_change_the_learned_run(shipped_model: Path) -> None:
    a, ev_a = run_model(ticks=250)
    b, ev_b = run_model(ticks=250, shadow_teacher=True)
    assert [e.winner for e in ev_a] == [e.winner for e in ev_b]
    assert a.latest_metrics.avg_wait == b.latest_metrics.avg_wait


def test_ineligible_cars_never_win_and_still_refuse(shipped_model: Path) -> None:
    _, events = run_model(scenario="car_breakdown", ticks=600)
    refusals = 0
    for e in events:
        status = {s.car_id: s for s in e.statuses}
        for b in e.bids:
            s = status[b.car_id]
            if not s.available or s.space <= 0:
                assert b.refused, "an ineligible car proposed"
                refusals += 1
        if e.winner is not None:
            s = status[e.winner]
            assert s.available and s.space > 0
    assert refusals > 0


@pytest.mark.slow
def test_dagger_beta_mixing_statistics(shipped_model: Path) -> None:
    teacher = learner = 0
    seed = 0
    while teacher + learner < 2000:
        model, _ = run_model(
            scenario="lunch_two_way", seed=100 + seed, ticks=900, shadow_teacher=True, beta=0.3
        )
        teacher += model.dispatcher.teacher_awards
        learner += model.dispatcher.learner_awards
        seed += 1
    frac = teacher / (teacher + learner)
    assert abs(frac - 0.3) <= 0.05, frac


def test_dagger_draws_never_touch_the_model_streams(shipped_model: Path) -> None:
    """Mixing uses a spawned stream: the arrival/SA stream (`model.random`) is untouched."""
    cfg = ScenarioConfig.load("morning_up_peak").model_copy(
        update={"strategy": "liftzero_bc", "lift": LiftConfig(shadow_teacher=True, beta=0.5)}
    )
    model = ElevatorModel(cfg)
    py_state = model.random.getstate()
    np_state = model.rng.bit_generator.state
    for _ in range(100):
        model.dagger_rng.random()
    assert model.random.getstate() == py_state
    assert model.rng.bit_generator.state == np_state


def test_bare_cnp_pair_sees_identical_passengers(shipped_model: Path) -> None:
    """Without SA/parking (which share the arrival stream) seeds give identical arrivals."""
    a, _ = run_model(strategy="liftzero_bc_cnp", ticks=300)
    b, _ = run_model(strategy="cnp_astar", ticks=300)
    assert a.latest_metrics.arrived == b.latest_metrics.arrived


def test_recorder_maps_learned_strategies_to_their_teacher(
    shipped_model: Path, tmp_path: Path
) -> None:
    from elevator_mas.learning.recorder import DecisionRecorder
    from elevator_mas.learning.schema import TEACHERS

    for strategy, teacher in (("liftzero_bc", "full"), ("liftzero_bc_cnp", "cnp_astar")):
        rec = DecisionRecorder(tmp_path / strategy, source="dagger_r1")
        cfg = ScenarioConfig.load("lunch_two_way").model_copy(
            update={"strategy": strategy, "lift": LiftConfig(shadow_teacher=True)}
        )
        model = ElevatorModel(cfg)
        model.decision_hooks.append(rec)
        for _ in range(150):
            model.step()
        shard = rec.flush()
        assert shard is not None
        data = np.load(shard)
        assert set(np.unique(data["teacher"])) == {TEACHERS.index(teacher)}
