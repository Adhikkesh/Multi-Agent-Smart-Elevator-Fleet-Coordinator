"""Decision-quality metrics for a learned bidder against the teacher (numpy only).

All functions take per-decision arrays: ``cost`` [M, N] teacher bids (``>= 1e5`` refused or
padded), ``valid`` [M, N] (mask & eligible), ``winner`` [M] teacher award, and the learner's
``choice`` [M] (or ``score`` [M, N]).

Two agreement notions are reported everywhere:

* **strict** — ``choice == winner`` (the teacher breaks exact ties by lowest car id);
* **tie-aware** — the chosen car's teacher cost equals the winner's (zero regret). An
  equivariant network has no car ids, so on an exact tie between *different-looking* cars it
  cannot know which one the id tie-break picked, and both choices are equally good.

The acceptance gate (>= 85 %) is evaluated on both and reported side by side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Bids at or above this are refused / padded sentinels.
SENTINEL = 1e5

#: Two bids within this absolute distance tie.
TIE_EPS = 1e-6


def argmin_choice(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Deterministic decision: argmin over valid cars, ties -> lowest id; -1 if none valid."""
    masked = np.where(valid, score, np.inf)
    pick = np.argmin(masked, axis=1)
    return np.where(valid.any(axis=1), pick, -1)


def nontrivial(valid: np.ndarray, winner: np.ndarray) -> np.ndarray:
    """Decisions with a winner and at least two valid cars."""
    return (winner >= 0) & (valid.sum(axis=1) >= 2)


def hard(cost: np.ndarray, valid: np.ndarray, winner: np.ndarray) -> np.ndarray:
    """Non-trivial decisions whose runner-up bid is within 10 % of the best bid."""
    c = np.sort(np.where(valid, cost, np.inf), axis=1)
    if c.shape[1] < 2:
        return np.zeros(len(cost), dtype=bool)
    best, second = c[:, 0], c[:, 1]
    with np.errstate(invalid="ignore"):
        close = (second - best) <= 0.10 * best
    return nontrivial(valid, winner) & np.isfinite(second) & close


def regret(cost: np.ndarray, winner: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """``cost[choice] - cost[winner]`` per decision (NaN where there is no winner)."""
    rows = np.arange(len(winner))
    ok = (winner >= 0) & (choice >= 0)
    out = np.full(len(winner), np.nan)
    out[ok] = cost[rows[ok], choice[ok]] - cost[rows[ok], winner[ok]]
    return out


def kendall_tau(score: np.ndarray, cost: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Kendall's tau-b between score and teacher cost over valid cars, per decision.

    Tau-b corrects for ties (exact teacher ties are common), so a perfect ranking scores
    1.0 even with tied bids. NaN where it is undefined (fewer than two valid cars, or all
    scores / all costs tied). Vectorised over all car pairs.
    """
    n = score.shape[1]
    i, j = np.triu_indices(n, k=1)
    pair_ok = valid[:, i] & valid[:, j]
    ds = np.sign(score[:, i] - score[:, j])
    dc = np.sign(cost[:, i] - cost[:, j])
    concord = np.where(pair_ok, ds * dc, 0.0).sum(axis=1)
    n0 = pair_ok.sum(axis=1)
    n1 = (pair_ok & (ds == 0)).sum(axis=1)
    n2 = (pair_ok & (dc == 0)).sum(axis=1)
    denom = np.sqrt((n0 - n1).astype(float) * (n0 - n2).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, concord / np.where(denom > 0, denom, 1.0), np.nan)


@dataclass
class DecisionMetrics:
    """Summary of a learner's decisions against the teacher."""

    n: int
    n_nontrivial: int
    n_hard: int
    agree: float
    agree_nontrivial: float
    agree_nontrivial_tie: float
    agree_hard: float
    agree_hard_tie: float
    regret_mean: float
    regret_median: float
    regret_p95: float
    rel_regret_mean: float
    kendall_tau: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _pct(x: np.ndarray) -> float:
    return float(np.mean(x) * 100.0) if len(x) else float("nan")


def decision_metrics(
    cost: np.ndarray,
    valid: np.ndarray,
    winner: np.ndarray,
    choice: np.ndarray,
    score: np.ndarray | None = None,
) -> DecisionMetrics:
    """Agreement, regret and rank correlation of ``choice`` (and ``score``) vs the teacher."""
    has = winner >= 0
    nt = nontrivial(valid, winner)
    hd = hard(cost, valid, winner)
    reg = regret(cost, winner, choice)
    zero = np.nan_to_num(reg, nan=np.inf) <= TIE_EPS
    rows = np.arange(len(winner))
    best = np.where(has, cost[rows, np.maximum(winner, 0)], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = reg / np.maximum(best, 1.0)
    reg_nt = reg[nt]
    tau = (
        float(np.nanmean(kendall_tau(score, cost, valid)[nt]))
        if score is not None and nt.any()
        else float("nan")
    )
    return DecisionMetrics(
        n=int(has.sum()),
        n_nontrivial=int(nt.sum()),
        n_hard=int(hd.sum()),
        agree=_pct((choice == winner)[has]),
        agree_nontrivial=_pct((choice == winner)[nt]),
        agree_nontrivial_tie=_pct(zero[nt]),
        agree_hard=_pct((choice == winner)[hd]),
        agree_hard_tie=_pct(zero[hd]),
        regret_mean=float(np.mean(reg_nt)) if len(reg_nt) else float("nan"),
        regret_median=float(np.median(reg_nt)) if len(reg_nt) else float("nan"),
        regret_p95=float(np.percentile(reg_nt, 95)) if len(reg_nt) else float("nan"),
        rel_regret_mean=float(np.nanmean(rel[nt])) if nt.any() else float("nan"),
        kendall_tau=tau,
    )
