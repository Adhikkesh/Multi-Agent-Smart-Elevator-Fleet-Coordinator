"""Masked categorical policy over eligible cars, the critic heads, and the KL anchor.

Policy: ``π(i | s) = softmax_{i ∈ E}(−score_i · e^{−τ})`` over eligible cars ``E`` (the same
semantics as Phase 4). Ineligible/padded cars get logit −1e9 (probability exactly 0 in
float32). Decisions with ``|E| = 1`` have entropy 0 and carry no policy gradient; the
trainer skips them in the policy loss but keeps them for the value targets.

Critics (CTDE): ``V_priv`` sees the decision representation *and* the privileged state;
``V_pub`` is the network's own public value head, distilled from ``V_priv`` so Phase 6's
search can use it. Both read a *detached* copy of the actor's pooled representation, so
value errors cannot corrupt the imitation-learned features; the actor never sees the
privileged state (tested: it is not an input of the actor, nor of the ONNX graph).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from elevator_mas.learning.lift.model import NetOut
from elevator_mas.learning.lift.rl.privileged import KPRIV

NEG = -1.0e9


def masked_logits(out: NetOut, valid: Tensor) -> Tensor:
    """Policy logits with invalid cars at −1e9."""
    logits = -out.score * torch.exp(-out.logit_temp)
    return logits.masked_fill(~valid, NEG)


def log_probs(logits: Tensor) -> Tensor:
    return torch.log_softmax(logits, dim=-1)


def entropy(logits: Tensor, valid: Tensor) -> Tensor:
    """[B] entropy over valid cars (0 when a single car is valid)."""
    lp = log_probs(logits)
    p = lp.exp()
    return -(p * lp.masked_fill(~valid, 0.0)).sum(dim=-1)


def kl_anchor(anchor_logits: Tensor, logits: Tensor, valid: Tensor) -> Tensor:
    """[B] KL(π_BC ‖ π) over valid cars — zero iff the policies agree on E."""
    lp_bc = log_probs(anchor_logits)
    lp = log_probs(logits)
    p_bc = lp_bc.exp()
    return (p_bc * (lp_bc - lp)).masked_fill(~valid, 0.0).sum(dim=-1)


class PrivilegedCritic(nn.Module):
    """``V_priv(pooled, privileged)`` — the centralised critic used for GAE."""

    def __init__(self, d_pooled: int, d_hidden: int = 128) -> None:
        super().__init__()
        self.priv = nn.Sequential(nn.Linear(KPRIV, d_hidden), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(d_pooled + d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def forward(self, pooled: Tensor, privileged: Tensor) -> Tensor:
        h = torch.cat([pooled, self.priv(privileged)], dim=-1)
        return self.net(h).squeeze(-1)
