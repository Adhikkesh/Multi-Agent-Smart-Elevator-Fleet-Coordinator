"""LiftZeroNet: symmetry, padding, sizes, budget, determinism, gradients, policy helpers."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from lift_helpers import random_batch  # noqa: E402

from elevator_mas.learning.lift.config import ModelConfig  # noqa: E402
from elevator_mas.learning.lift.model import (  # noqa: E402
    INELIGIBLE_SCORE,
    LiftZeroNet,
    choose,
    count_parameters,
    policy_logits,
)


def run(net: Any, b: dict[str, Any]) -> Any:
    with torch.no_grad():
        return net(b["call"], b["cars"], b["glob"], b["mask"], b["eligible"])


def test_default_parameter_budget() -> None:
    n = count_parameters(LiftZeroNet(ModelConfig()))
    assert 200_000 <= n <= 400_000, n


def test_output_shapes(tiny_net: Any) -> None:
    out = run(tiny_net, random_batch(3, 5))
    assert out.score.shape == (3, 5)
    assert out.aux.shape == (3, 5, 4)
    assert out.value.shape == (3,)
    assert out.attn.shape == (3, 5)
    assert out.logit_temp.shape == ()


@pytest.mark.parametrize("seed", range(5))
def test_permutation_equivariance(tiny_net: Any, seed: int) -> None:
    b = random_batch(4, 7, seed)
    perm = torch.randperm(7, generator=torch.Generator().manual_seed(seed))
    pb = {**b, "cars": b["cars"][:, perm], "mask": b["mask"][:, perm]}
    pb["eligible"] = b["eligible"][:, perm]
    o, po = run(tiny_net, b), run(tiny_net, pb)
    torch.testing.assert_close(po.score, o.score[:, perm], atol=1e-5, rtol=0)
    torch.testing.assert_close(po.aux, o.aux[:, perm], atol=1e-5, rtol=0)
    torch.testing.assert_close(po.attn, o.attn[:, perm], atol=1e-5, rtol=0)
    torch.testing.assert_close(po.value, o.value, atol=1e-5, rtol=0)


@pytest.mark.parametrize("pad", [1, 5, 20])
def test_padding_invariance(tiny_net: Any, pad: int) -> None:
    b = random_batch(3, 4, seed=pad)
    b["mask"][:] = True
    junk = torch.rand(3, pad, 26) * 7.0  # padding content must not matter
    padded = {
        **b,
        "cars": torch.cat([b["cars"], junk], dim=1),
        "mask": torch.cat([b["mask"], torch.zeros(3, pad, dtype=torch.bool)], dim=1),
        "eligible": torch.cat([b["eligible"], torch.ones(3, pad, dtype=torch.bool)], dim=1),
    }
    o, po = run(tiny_net, b), run(tiny_net, padded)
    torch.testing.assert_close(po.score[:, :4], o.score, atol=1e-5, rtol=0)
    torch.testing.assert_close(po.aux[:, :4], o.aux, atol=1e-5, rtol=0)
    torch.testing.assert_close(po.value, o.value, atol=1e-5, rtol=0)
    torch.testing.assert_close(po.attn[:, :4], o.attn, atol=1e-5, rtol=0)
    assert torch.all(po.attn[:, 4:] == 0)


@pytest.mark.parametrize("n", [1, 2, 8, 16, 32])
def test_any_fleet_size_is_finite(tiny_net: Any, n: int) -> None:
    out = run(tiny_net, random_batch(2, n))
    assert torch.isfinite(out.score).all()
    assert torch.isfinite(out.value).all()
    torch.testing.assert_close(out.attn.sum(dim=1), torch.ones(2), atol=1e-5, rtol=0)


def test_padded_and_ineligible_cars_get_the_offset(tiny_net: Any) -> None:
    b = random_batch(2, 4)
    b["mask"][:] = True
    b["eligible"][:] = True
    b["mask"][:, 3] = False
    b["eligible"][:, 2] = False
    out = run(tiny_net, b)
    assert torch.all(out.score[:, 2:] > INELIGIBLE_SCORE / 2)
    assert torch.all(out.score[:, :2] < INELIGIBLE_SCORE / 2)


def test_eval_forward_is_deterministic(tiny_net: Any) -> None:
    b = random_batch(5, 6)
    torch.testing.assert_close(run(tiny_net, b).score, run(tiny_net, b).score, atol=0, rtol=0)


def test_no_nan_under_extreme_valid_inputs(tiny_net: Any) -> None:
    b = random_batch(4, 8)
    for k in ("call", "cars", "glob"):
        b[k] = torch.where(torch.rand_like(b[k]) > 0.5, 1.0, 0.0)
    b["cars"][..., 21] = -1.0
    out = run(tiny_net, b)
    for t in (out.score, out.aux, out.value, out.attn):
        assert torch.isfinite(t).all()


def test_gradient_reaches_every_parameter_except_value(tiny_cfg: Any) -> None:
    torch.manual_seed(0)
    net = LiftZeroNet(tiny_cfg)
    b = random_batch(6, 5)
    out = net(b["call"], b["cars"], b["glob"], b["mask"], b["eligible"])
    valid = b["mask"] & b["eligible"]
    loss = (out.score * valid).sum() + (out.aux * valid[..., None]).sum()
    loss = loss + policy_logits(out, b["mask"], b["eligible"]).logsumexp(dim=1).sum()
    loss.backward()
    dead = [
        name
        for name, p in net.named_parameters()
        if (p.grad is None or float(p.grad.abs().sum()) == 0.0) and not name.startswith("value")
    ]
    assert dead == []


def test_choose_tie_breaks_to_lowest_valid_id() -> None:
    score = torch.tensor([[3.0, 1.0, 1.0, 0.5], [2.0, 2.0, 5.0, 9.0], [1.0, 1.0, 1.0, 1.0]])
    mask = torch.ones(3, 4, dtype=torch.bool)
    elig = torch.tensor(
        [[True, True, True, False], [True, True, True, True], [False, False, False, False]]
    )
    assert choose(score, mask, elig).tolist() == [1, 0, -1]


def test_policy_logits_mask_invalid_cars(tiny_net: Any) -> None:
    b = random_batch(3, 5)
    out = run(tiny_net, b)
    logits = policy_logits(out, b["mask"], b["eligible"])
    assert torch.all(torch.isinf(logits[~(b["mask"] & b["eligible"])]))
    probs = torch.softmax(logits, dim=1)
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0)


def test_bad_head_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        LiftZeroNet(ModelConfig(d_model=30, n_heads=4))
