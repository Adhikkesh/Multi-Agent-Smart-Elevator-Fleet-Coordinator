"""LiftZeroNet — the permutation-equivariant set-Transformer that prices hall calls.

The network replaces the A* marginal-cost computation inside each car's Contract Net bid.
It reads one *decision*: a global token (time, building, fleet load, the board's cost
weights), the call token (where, which way, how urgent) and one token per car (position,
direction, load, door, plan summary, geometry relative to the call), and returns for every
car a predicted cost of taking the call — lower is better, as with the teacher's bids.

Design (see ``docs/lift/PHASE4_IMITATION.md``):

* Token embeddings are small MLPs plus a learned *type* embedding (global / call / car).
  There is **no positional encoding and no car-id embedding**, so the network is
  permutation-*equivariant* over cars: permuting the cars permutes the outputs. One set of
  weights therefore serves fleets of any size, which is what the 40-floor x 8-car held-out
  split tests.
* ``L`` pre-LayerNorm Transformer encoder layers over ``[glob, call, car_1 .. car_N]``.
  Padded cars are removed from attention by an additive key bias, so adding padding never
  changes the real cars' scores.
* Attention is written out explicitly (linear projections, scaled dot product, softmax)
  rather than with ``nn.MultiheadAttention``: the weights of every layer are available for
  explainability, and the graph exports to ONNX with dynamic batch and fleet axes without
  relying on the fused fast-path.

AIMA framing: this is the *performance element* of a learning agent; the imitation loss
against the A* teacher is its *critic* and the trainer its *learning element*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from elevator_mas.learning.lift.config import ModelConfig

#: Score offset added to cars that cannot take the call (padded or ineligible). Finite, so
#: the ONNX graph never produces inf/NaN, and far above any real log-cost (log1p(1e6) < 14).
INELIGIBLE_SCORE = 1.0e4

#: Additive attention bias for padded keys.
_MASKED_LOGIT = -1.0e9

#: Token types: index into the type embedding.
_GLOBAL, _CALL, _CAR = 0, 1, 2


@dataclass
class NetOut:
    """Network outputs for a batch of B decisions over N cars."""

    #: [B, N] predicted log-cost of awarding the call to each car (LOWER is better).
    #: Padded / ineligible cars carry ``+INELIGIBLE_SCORE``.
    score: Tensor
    #: [B, N, 4] predicted bid breakdown (log1p of wait, ride, crowding, energy).
    aux: Tensor
    #: [B] state value of the decision — trained in Phase 5, untrained here.
    value: Tensor
    #: [B, N] attention of the call token over the car tokens (last layer, head-averaged,
    #: renormalised over real cars).
    attn: Tensor
    #: scalar learnable log-temperature of the stochastic policy.
    logit_temp: Tensor


def _token_mlp(k_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(k_in, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))


class SetAttention(nn.Module):
    """Multi-head self-attention with a key-padding bias; returns head-averaged weights."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d % heads != 0:
            raise ValueError(f"d_model={d} is not divisible by n_heads={heads}")
        self.heads = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_bias: Tensor) -> tuple[Tensor, Tensor]:
        """``x`` [B, T, d]; ``key_bias`` [B, T] (0 for real tokens, -1e9 for padding)."""
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, dh]
        logits = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dh))
        logits = logits + key_bias[:, None, None, :]
        weights = torch.softmax(logits, dim=-1)  # [B, H, T, T]
        ctx = torch.matmul(self.drop(weights), v).transpose(1, 2).reshape(b, t, d)
        return self.out(ctx), weights.mean(dim=1)


class EncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = SetAttention(d, heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_bias: Tensor) -> tuple[Tensor, Tensor]:
        h, weights = self.attn(self.ln1(x), key_bias)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, weights


class LiftZeroNet(nn.Module):
    """Permutation-equivariant set-Transformer bidder (see module docstring)."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d = cfg.d_model
        self.call_mlp = _token_mlp(cfg.kc, d)
        self.car_mlp = _token_mlp(cfg.kcar, d)
        self.glob_mlp = _token_mlp(cfg.kg, d)
        self.type_emb = nn.Embedding(3, d)
        self.layers = nn.ModuleList(
            EncoderLayer(d, cfg.n_heads, cfg.ffn_mult * d, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.final_ln = nn.LayerNorm(d)
        self.score_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.aux_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.value_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.logit_temp = nn.Parameter(torch.zeros(()))
        self._init_heads()

    def _init_heads(self) -> None:
        # The value head is trained only in Phase 5: start it near zero.
        last = self.value_head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        call: Tensor,
        cars: Tensor,
        glob: Tensor,
        mask: Tensor,
        eligible: Tensor | None = None,
    ) -> NetOut:
        """Score every car for the call.

        Args:
            call: [B, Kc] call token features.
            cars: [B, N, Kcar] car token features (any N >= 1).
            glob: [B, Kg] global token features.
            mask: [B, N] bool, True where the car exists.
            eligible: [B, N] bool, True where the car may take the call (defaults to mask).
        """
        if eligible is None:
            eligible = mask
        b = call.shape[0]
        maskf = mask.to(cars.dtype)
        okf = (mask & eligible).to(cars.dtype)

        types = self.type_emb.weight
        g = self.glob_mlp(glob)[:, None, :] + types[_GLOBAL]
        c = self.call_mlp(call)[:, None, :] + types[_CALL]
        k = self.car_mlp(cars) + types[_CAR]
        x = torch.cat([g, c, k], dim=1)  # [B, 2 + N, d]

        ones = torch.ones(b, 2, dtype=cars.dtype, device=cars.device)
        key_bias = (1.0 - torch.cat([ones, maskf], dim=1)) * _MASKED_LOGIT

        weights = key_bias  # replaced by the last layer's weights
        for layer in self.layers:
            x, weights = layer(x, key_bias)
        x = self.final_ln(x)

        car_tok = x[:, 2:, :]
        score = self.score_head(car_tok).squeeze(-1) + (1.0 - okf) * INELIGIBLE_SCORE
        aux = self.aux_head(car_tok)

        denom = maskf.sum(dim=1, keepdim=True).clamp(min=1.0)
        car_mean = (car_tok * maskf[:, :, None]).sum(dim=1) / denom
        value = self.value_head(torch.cat([x[:, 0, :], x[:, 1, :], car_mean], dim=-1))

        call_attn = weights[:, 1, 2:] * maskf
        call_attn = call_attn / call_attn.sum(dim=1, keepdim=True).clamp(min=1e-9)

        return NetOut(
            score=score,
            aux=aux,
            value=value.squeeze(-1),
            attn=call_attn,
            logit_temp=self.logit_temp,
        )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def policy_logits(out: NetOut, mask: Tensor, eligible: Tensor) -> Tensor:
    """Logits of the stochastic policy: ``-score * exp(-logit_temp)``, -inf where invalid."""
    logits = -out.score * torch.exp(-out.logit_temp)
    return logits.masked_fill(~(mask & eligible), float("-inf"))


def choose(score: Tensor, mask: Tensor, eligible: Tensor) -> Tensor:
    """Deterministic decision: argmin score over valid cars (ties -> lowest car id).

    Returns -1 for decisions with no valid car. ``torch.argmin`` returns the first minimum,
    which is the lowest car id, matching the dispatcher's tie-break.
    """
    valid = mask & eligible
    masked = score.masked_fill(~valid, float("inf"))
    pick = torch.argmin(masked, dim=1)
    return torch.where(valid.any(dim=1), pick, torch.full_like(pick, -1))
