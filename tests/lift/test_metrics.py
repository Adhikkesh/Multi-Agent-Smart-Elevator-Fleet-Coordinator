"""Offline decision metrics on synthetic cases with known answers (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from elevator_mas.learning.lift.metrics import (
    argmin_choice,
    decision_metrics,
    hard,
    kendall_tau,
    nontrivial,
    regret,
)

COST = np.array(
    [
        [1.0, 5.0, 9.0],  # clear winner 0
        [4.0, 4.0, 8.0],  # exact tie 0/1
        [3.0, 3.2, 1e6],  # hard (within 10 %), car 2 refused
        [7.0, 1e6, 1e6],  # trivial: one valid car
    ]
)
VALID = COST < 1e5
WINNER = np.array([0, 0, 0, 0])


def test_argmin_choice_ties_and_empty_rows() -> None:
    s = np.array([[2.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    v = np.array([[True, True, True], [False, False, False]])
    assert argmin_choice(s, v).tolist() == [1, -1]


def test_nontrivial_and_hard_masks() -> None:
    assert nontrivial(VALID, WINNER).tolist() == [True, True, True, False]
    assert hard(COST, VALID, WINNER).tolist() == [False, True, True, False]


def test_regret_values() -> None:
    choice = np.array([1, 1, 1, 0])
    np.testing.assert_allclose(regret(COST, WINNER, choice), [4.0, 0.0, 0.2, 0.0], rtol=1e-6)


def test_agreement_strict_vs_tie_aware() -> None:
    choice = np.array([0, 1, 0, 0])  # picks the other tied car in row 1
    m = decision_metrics(COST, VALID, WINNER, choice)
    assert m.n_nontrivial == 3
    assert m.agree_nontrivial == pytest.approx(200 / 3)
    assert m.agree_nontrivial_tie == pytest.approx(100.0)
    assert m.regret_mean == pytest.approx(0.0)


def test_perfect_scores_give_tau_one_and_reversed_minus_one() -> None:
    cost = np.array([[1.0, 2.0, 3.0, 4.0]])
    valid = np.ones_like(cost, dtype=bool)
    assert kendall_tau(np.log1p(cost), cost, valid)[0] == pytest.approx(1.0)
    assert kendall_tau(-cost, cost, valid)[0] == pytest.approx(-1.0)
    assert np.isnan(kendall_tau(cost, cost, np.array([[True, False, False, False]]))[0])


def test_metrics_on_the_teacher_itself_are_perfect() -> None:
    choice = argmin_choice(COST, VALID)
    m = decision_metrics(COST, VALID, WINNER, choice, score=np.log1p(COST))
    assert m.agree == 100.0 and m.agree_nontrivial == 100.0 and m.agree_hard == 100.0
    assert m.regret_p95 == 0.0
    assert m.kendall_tau == pytest.approx(1.0)
