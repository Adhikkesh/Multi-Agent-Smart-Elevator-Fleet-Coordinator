"""Imitation losses: hand-computed values, masking, ties, hard weighting."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from elevator_mas.learning.lift.config import LossWeights  # noqa: E402
from elevator_mas.learning.lift.losses import (  # noqa: E402
    Targets,
    hard_decisions,
    imitation_loss,
    winner_targets,
)
from elevator_mas.learning.lift.model import NetOut  # noqa: E402

ONLY = {
    "list": LossWeights(listwise=1, regression=0, rank=0, aux=0, pairwise=0),
    "reg": LossWeights(listwise=0, regression=1, rank=0, aux=0, pairwise=0),
    "rank": LossWeights(listwise=0, regression=0, rank=1, aux=0, pairwise=0),
    "aux": LossWeights(listwise=0, regression=0, rank=0, aux=1, pairwise=0),
    "pair": LossWeights(listwise=0, regression=0, rank=0, aux=0, pairwise=1),
}


def out_of(score: list[list[float]], temp: float = 0.0) -> NetOut:
    s = torch.tensor(score)
    return NetOut(
        score=s,
        aux=torch.zeros(*s.shape, 4),
        value=torch.zeros(s.shape[0]),
        attn=torch.zeros_like(s),
        logit_temp=torch.tensor(temp),
    )


def targets(cost: list[list[float]], winner: list[int], valid: list[list[bool]]) -> Targets:
    c = torch.tensor(cost)
    return Targets(
        teacher_cost=c,
        parts=torch.zeros(*c.shape, 4),
        winner=torch.tensor(winner),
        valid=torch.tensor(valid),
    )


def test_listwise_matches_hand_computation() -> None:
    out = out_of([[1.0, 2.0, 3.0]])
    tgt = targets([[1.0, 5.0, 9.0]], [0], [[True, True, True]])
    loss, _ = imitation_loss(out, tgt, ONLY["list"])
    expected = -math.log(math.exp(-1) / (math.exp(-1) + math.exp(-2) + math.exp(-3)))
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_temperature_scales_the_logits() -> None:
    tgt = targets([[1.0, 5.0]], [0], [[True, True]])
    cold, _ = imitation_loss(out_of([[1.0, 2.0]], temp=-1.0), tgt, ONLY["list"])
    warm, _ = imitation_loss(out_of([[1.0, 2.0]], temp=1.0), tgt, ONLY["list"])
    assert float(cold) < float(warm)


@pytest.mark.parametrize("term", list(ONLY))
def test_ineligible_and_padded_cars_contribute_exactly_zero(term: str) -> None:
    valid = [[True, True, False, False]]
    tgt = targets([[1.0, 4.0, 1e6, 1e6]], [0], valid)
    base, _ = imitation_loss(out_of([[0.5, 2.0, 0.0, 0.0]]), tgt, ONLY[term])
    tgt2 = targets([[1.0, 4.0, 3.0, 1e6]], [0], valid)  # junk label on an invalid car
    changed, _ = imitation_loss(out_of([[0.5, 2.0, -50.0, 99.0]]), tgt2, ONLY[term])
    assert float(changed) == pytest.approx(float(base), abs=1e-6)


def test_regression_is_huber_on_log_costs() -> None:
    cost = [[math.expm1(1.0), math.expm1(3.0)]]
    tgt = targets(cost, [0], [[True, True]])
    loss, _ = imitation_loss(out_of([[1.5, 3.0]]), tgt, ONLY["reg"])
    assert float(loss) == pytest.approx((0.5 * 0.5**2 + 0.0) / 2, rel=1e-5)


def test_rank_term_ignores_a_constant_offset() -> None:
    cost = [[math.expm1(1.0), math.expm1(2.0), math.expm1(4.0)]]
    tgt = targets(cost, [0], [[True, True, True]])
    loss, _ = imitation_loss(out_of([[11.0, 12.0, 14.0]]), tgt, ONLY["rank"])
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_aux_term_compares_log1p_parts() -> None:
    out = out_of([[0.0, 0.0]])
    out.aux[:] = torch.log1p(torch.tensor([2.0, 0.0, 1.0, 0.5]))
    tgt = targets([[1.0, 2.0]], [0], [[True, True]])
    tgt.parts[:] = torch.tensor([2.0, 0.0, 1.0, 0.5])
    loss, _ = imitation_loss(out, tgt, ONLY["aux"])
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_pairwise_hinge_and_margin() -> None:
    w = ONLY["pair"]
    tgt = targets([[10.0, 30.0]], [0], [[True, True]])  # easy decision
    ok, _ = imitation_loss(out_of([[1.0, 2.0]]), tgt, w)
    assert float(ok) == 0.0
    bad, _ = imitation_loss(out_of([[2.0, 1.95]]), tgt, w)
    assert float(bad) == pytest.approx(w.pair_margin + 0.05, rel=1e-5)


def test_hard_decisions_weigh_twice_in_the_pairwise_term() -> None:
    w = ONLY["pair"]
    easy = targets([[10.0, 30.0]], [0], [[True, True]])
    hard = targets([[10.0, 10.5]], [0], [[True, True]])
    le, _ = imitation_loss(out_of([[2.0, 2.0]]), easy, w)
    lh, _ = imitation_loss(out_of([[2.0, 2.0]]), hard, w)
    assert float(lh) == pytest.approx(w.hard_pair_weight * float(le), rel=1e-6)


def test_trivial_decisions_have_no_ranking_loss() -> None:
    tgt = targets([[3.0, 1e6]], [0], [[True, False]])
    for term in ("list", "rank", "pair"):
        loss, _ = imitation_loss(out_of([[7.0, 0.0]]), tgt, ONLY[term])
        assert float(loss) == 0.0


def test_ties_get_a_uniform_soft_target() -> None:
    t = winner_targets(torch.tensor([[2.0, 2.0, 5.0]]), torch.tensor([[True, True, True]]))
    torch.testing.assert_close(t, torch.tensor([[0.5, 0.5, 0.0]]))
    tgt = targets([[2.0, 2.0, 5.0]], [0], [[True, True, True]])
    # Either tied car scoring lowest is equally right.
    a, _ = imitation_loss(out_of([[0.0, 9.0, 9.0]]), tgt, ONLY["list"])
    b, _ = imitation_loss(out_of([[9.0, 0.0, 9.0]]), tgt, ONLY["list"])
    assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_hard_decision_detection() -> None:
    cost = torch.tensor([[10.0, 10.9, 50.0], [10.0, 20.0, 30.0], [0.0, 0.0, 1.0], [5.0, 1e6, 1e6]])
    valid = torch.tensor([[True] * 3, [True] * 3, [True] * 3, [True, False, False]])
    assert hard_decisions(cost, valid).tolist() == [True, False, True, False]


def test_all_refused_batch_gives_zero_not_nan() -> None:
    tgt = targets([[1e6, 1e6]], [-1], [[False, False]])
    loss, parts = imitation_loss(out_of([[1.0, 2.0]]), tgt, LossWeights())
    assert float(loss) == 0.0
    assert all(math.isfinite(float(v)) for v in parts.values())
