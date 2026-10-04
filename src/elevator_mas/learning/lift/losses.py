"""Imitation losses for LiftZeroNet (behaviour cloning of the A* bidder).

Notation for one decision: ``E`` is the set of eligible cars (``mask & eligible``), the
teacher's bids are ``c_i`` (``1e6`` where refused/padded), ``y_i = log1p(c_i)`` and the
network's score is ``s_i`` (lower is better). All terms are computed over ``E`` only, so
padded and ineligible cars contribute exactly zero.

1. **Listwise cross-entropy** ``-sum_i t_i log softmax_E(-s_i * exp(-tau))``. The target
   ``t`` is one-hot on the teacher's winner, or uniform over the cars that *tie* for the
   best bid: the network is permutation-equivariant and cannot see car ids, so it cannot
   (and should not) learn the dispatcher's lowest-id tie-break.
2. **Regression** ``Huber_1(s_i - y_i)`` — calibrates scores as log-costs for MCTS and UI.
3. **Centred rank regression** ``Huber_0.5((s_i - mean_E s) - (y_i - mean_E y))`` — removes
   the per-decision cost scale, so the ordering is learnt even where the level is not.
4. **Auxiliary breakdown** ``Huber_1(aux_ik - log1p(teacher_k_i))`` for wait, ride, crowding,
   energy.
5. **Pairwise hinge** ``max(0, m - (s_runner_up - s_winner))`` against the best strictly
   worse car, weighted x2 on *hard* decisions (runner-up within 10 % of the best).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from elevator_mas.learning.lift.config import LossWeights
from elevator_mas.learning.lift.model import NetOut

#: Teacher costs at or above this are refused/padded sentinels.
SENTINEL = 1e5

#: Two bids within this absolute distance are an exact tie.
TIE_EPS = 1e-6


@dataclass
class Targets:
    """Teacher labels for a batch, as tensors."""

    teacher_cost: Tensor  # [B, N] float
    parts: Tensor  # [B, N, 4] float (wait, ride, crowding, energy)
    winner: Tensor  # [B] long (-1 = all refused)
    valid: Tensor  # [B, N] bool, mask & eligible


def _masked_mean(x: Tensor, w: Tensor) -> Tensor:
    return (x * w).sum() / w.sum().clamp(min=1.0)


def winner_targets(cost: Tensor, valid: Tensor) -> Tensor:
    """Soft listwise target: uniform over the cars that tie for the best valid bid."""
    big = torch.full_like(cost, float("inf"))
    c = torch.where(valid, cost, big)
    best = c.min(dim=1, keepdim=True).values
    tie = valid & (c <= best + TIE_EPS)
    t = tie.to(cost.dtype)
    return t / t.sum(dim=1, keepdim=True).clamp(min=1.0)


def hard_decisions(cost: Tensor, valid: Tensor) -> Tensor:
    """[B] bool: at least two valid cars and runner-up within 10 % of the best bid."""
    c = torch.where(valid, cost, torch.full_like(cost, float("inf")))
    two = torch.topk(c, k=min(2, c.shape[1]), dim=1, largest=False).values
    if two.shape[1] < 2:
        return torch.zeros(cost.shape[0], dtype=torch.bool, device=cost.device)
    best, second = two[:, 0], two[:, 1]
    return torch.isfinite(second) & ((second - best) <= 0.10 * best)


def imitation_loss(out: NetOut, tgt: Targets, w: LossWeights) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted sum of the five terms; returns ``(total, components)``."""
    valid = tgt.valid
    validf = valid.to(out.score.dtype)
    n_valid = valid.sum(dim=1)
    has_winner = (tgt.winner >= 0) & (n_valid >= 1)
    nontrivial = has_winner & (n_valid >= 2)

    y = torch.log1p(tgt.teacher_cost.clamp(min=0.0, max=SENTINEL))
    s = out.score

    # 1. listwise CE with tie-aware soft targets
    logits = (-s * torch.exp(-out.logit_temp)).masked_fill(~valid, -1e9)
    logp = F.log_softmax(logits, dim=1)
    target = winner_targets(tgt.teacher_cost, valid)
    ce = -(target * logp.masked_fill(~valid, 0.0)).sum(dim=1)
    l_list = _masked_mean(ce, nontrivial.to(ce.dtype))

    # 2. per-car regression on log-costs
    reg = F.huber_loss(s.masked_fill(~valid, 0.0), y.masked_fill(~valid, 0.0), reduction="none")
    l_reg = _masked_mean(reg, validf)

    # 3. per-decision centred rank regression
    denom = validf.sum(dim=1, keepdim=True).clamp(min=1.0)
    s_c = s.masked_fill(~valid, 0.0) - (s.masked_fill(~valid, 0.0).sum(1, keepdim=True) / denom)
    y_c = y.masked_fill(~valid, 0.0) - (y.masked_fill(~valid, 0.0).sum(1, keepdim=True) / denom)
    rank = F.huber_loss(s_c, y_c, reduction="none", delta=0.5)
    l_rank = _masked_mean(rank, validf * nontrivial[:, None].to(validf.dtype))

    # 4. auxiliary breakdown
    parts_y = torch.log1p(tgt.parts.clamp(min=0.0))
    aux = F.huber_loss(out.aux, parts_y, reduction="none").mean(dim=-1)
    l_aux = _masked_mean(aux, validf)

    # 5. pairwise hinge: winner vs best strictly-worse car
    inf = torch.full_like(tgt.teacher_cost, float("inf"))
    c = torch.where(valid, tgt.teacher_cost, inf)
    best = c.min(dim=1, keepdim=True).values
    winners = valid & (c <= best + TIE_EPS)
    worse = torch.where(valid & ~winners, c, inf)
    has_runner = torch.isfinite(worse.min(dim=1).values) & nontrivial
    s_win = s.masked_fill(~winners, 1e9).min(dim=1).values
    runner_idx = worse.argmin(dim=1, keepdim=True)
    s_run = s.gather(1, runner_idx).squeeze(1)
    gap = torch.where(has_runner, s_run - s_win, torch.zeros_like(s_win))
    hinge = F.relu(w.pair_margin - gap)
    pair_w = has_runner.to(s.dtype) * torch.where(
        hard_decisions(tgt.teacher_cost, valid),
        torch.full_like(s_win, w.hard_pair_weight),
        torch.ones_like(s_win),
    )
    l_pair = (hinge * pair_w).sum() / has_runner.sum().clamp(min=1)

    total = (
        w.listwise * l_list
        + w.regression * l_reg
        + w.rank * l_rank
        + w.aux * l_aux
        + w.pairwise * l_pair
    )
    return total, {
        "list": l_list.detach(),
        "reg": l_reg.detach(),
        "rank": l_rank.detach(),
        "aux": l_aux.detach(),
        "pair": l_pair.detach(),
    }
