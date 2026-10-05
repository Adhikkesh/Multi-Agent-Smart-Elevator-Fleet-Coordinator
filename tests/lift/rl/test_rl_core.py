"""Phase 5 core: SMDP-GAE, PPO loss, KL anchor, privileged isolation, guards, statistics."""

from __future__ import annotations

import numpy as np
import pytest

from elevator_mas.learning.lift.rl.curriculum import (
    SeedLeakError,
    assert_no_test_seed,
    sample_building,
    stage_at,
)
from elevator_mas.learning.lift.rl.gae import gae, gae_reference, smdp_discount
from elevator_mas.learning.lift.rl.stats import bootstrap_ci, holm, paired, wilcoxon_p

torch = pytest.importorskip("torch")


def test_smdp_discount_is_time_aware() -> None:
    d = smdp_discount(np.array([0.0, 5.0, 10.0]), 0.99, 5.0)
    np.testing.assert_allclose(d, [1.0, 0.99, 0.99**2])


def test_gae_hand_computed_with_irregular_dt_and_truncation() -> None:
    r = np.array([-1.0, -2.0, -0.5])
    v = np.array([0.5, 0.2, 0.1])
    dt = np.array([5.0, 0.0, 10.0])
    end = np.array([False, False, True])
    term = np.zeros(3, dtype=bool)
    boot = np.array([0.0, 0.0, 0.3])
    g, lam = 0.99, 0.95
    d2 = -0.5 + g**2 * 0.3 - 0.1
    d1 = -2.0 + 1.0 * 0.1 - 0.2
    d0 = -1.0 + g * 0.2 - 0.5
    a2 = d2
    a1 = d1 + 1.0 * lam * a2
    a0 = d0 + g * lam * a1
    adv, ret = gae(r, v, dt, end, term, boot, g, 5.0, lam)
    np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-9)
    np.testing.assert_allclose(ret, adv + v)


def test_gae_matches_reference_on_random_episodes() -> None:
    rng = np.random.default_rng(0)
    n = 200
    end = rng.random(n) < 0.05
    end[-1] = True
    args = (
        rng.normal(size=n),
        rng.normal(size=n),
        rng.integers(0, 20, size=n).astype(float),
        end,
        rng.random(n) < 0.3,
        rng.normal(size=n),
    )
    adv, _ = gae(*args)
    np.testing.assert_allclose(adv, gae_reference(*args), rtol=1e-9, atol=1e-12)


def test_truncation_bootstraps_but_termination_does_not() -> None:
    base = (np.array([0.0]), np.array([0.0]), np.array([5.0]), np.array([True]))
    trunc, _ = gae(*base, np.array([False]), np.array([1.0]))
    term, _ = gae(*base, np.array([True]), np.array([1.0]))
    assert trunc[0] == pytest.approx(0.99) and term[0] == 0.0


def _batch(n: int = 128, cars: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    valid = torch.ones(n, cars, dtype=torch.bool)
    valid[: n // 4, 1:] = False  # some single-eligible decisions
    return {
        "call": torch.rand(n, 14, generator=g),
        "cars": torch.rand(n, cars, 26, generator=g),
        "glob": torch.rand(n, 10, generator=g),
        "mask": torch.ones(n, cars, dtype=torch.bool),
        "valid": valid,
        "priv": torch.rand(n, 240, generator=g),
        "action": torch.zeros(n, dtype=torch.long),
        "adv": torch.randn(n, generator=g),
        "ret": torch.randn(n, generator=g),
    }


def _nets() -> tuple:
    from elevator_mas.learning.lift.config import ModelConfig
    from elevator_mas.learning.lift.model import LiftZeroNet
    from elevator_mas.learning.lift.rl.policy import PrivilegedCritic

    torch.manual_seed(0)
    cfg = ModelConfig(d_model=16, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0)
    return LiftZeroNet(cfg).eval(), PrivilegedCritic(48)


def test_ppo_loss_is_zero_policy_gradient_at_ratio_one_and_kl_zero_vs_itself() -> None:
    from elevator_mas.learning.lift.rl.policy import masked_logits
    from elevator_mas.learning.lift.rl.ppo import LossCoefs, ppo_loss

    net, critic = _nets()
    b = _batch()
    with torch.no_grad():
        out = net(b["call"], b["cars"], b["glob"], b["mask"], b["valid"])
        b["logp"] = torch.log_softmax(masked_logits(out, b["valid"]), -1)[:, 0]
    k = LossCoefs(clip=0.2, c_value=0.5, c_distill=0.5, c_entropy=0.01, beta=0.2)
    _, st = ppo_loss(net, critic, net, b, k)
    assert st["kl_anchor"] == pytest.approx(0.0, abs=1e-6)
    assert st["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert st["clipfrac"] == 0.0


def test_clipping_caps_the_objective() -> None:
    from elevator_mas.learning.lift.rl.ppo import LossCoefs, ppo_loss

    net, critic = _nets()
    b = _batch(n=40)  # < 64: advantages are not renormalised
    b["valid"][:] = True
    b["adv"] = torch.ones(40)
    b["logp"] = torch.full((40,), -10.0)  # ratio >> 1 + eps
    k = LossCoefs(clip=0.2, c_value=0.0, c_distill=0.0, c_entropy=0.0, beta=0.0)
    _, st = ppo_loss(net, critic, None, b, k)
    assert st["policy"] == pytest.approx(-1.2, abs=1e-5)  # min(r·A, 1.2·A) = 1.2
    assert st["clipfrac"] == 1.0


def test_masked_policy_gives_zero_mass_to_ineligible_and_zero_entropy_if_single() -> None:
    from elevator_mas.learning.lift.rl.policy import entropy, kl_anchor, masked_logits

    net, _ = _nets()
    b = _batch(n=8)
    with torch.no_grad():
        out = net(b["call"], b["cars"], b["glob"], b["mask"], b["valid"])
    logits = masked_logits(out, b["valid"])
    p = torch.softmax(logits, -1)
    assert torch.all(p[~b["valid"]] == 0)
    h = entropy(logits, b["valid"])
    assert torch.all(h[:2] == 0) and torch.all(h[2:] > 0)
    other = logits + torch.randn_like(logits) * b["valid"]
    assert torch.all(kl_anchor(logits, other, b["valid"])[2:] > 0)


def test_actor_never_sees_privileged_state() -> None:
    import inspect

    from elevator_mas.learning.lift.model import LiftZeroNet

    params = inspect.signature(LiftZeroNet.forward).parameters
    assert "priv" not in params and "privileged" not in params
    net, critic = _nets()
    b = _batch(n=4)
    priv = b["priv"].clone().requires_grad_(True)
    out = net(b["call"], b["cars"], b["glob"], b["mask"], b["valid"])
    v = critic(out.pooled.detach(), priv)
    v.sum().backward()
    assert priv.grad is not None and priv.grad.abs().sum() > 0
    assert torch.equal(out.score, net(b["call"], b["cars"], b["glob"], b["mask"], b["valid"]).score)


def test_no_test_seed_guard_and_curriculum() -> None:
    assert assert_no_test_seed(0) == 0 and assert_no_test_seed(7999) == 7999
    for bad in (8000, 8500, -1):
        with pytest.raises(SeedLeakError):
            assert_no_test_seed(bad)
    rng = np.random.default_rng(0)
    for p in (0.0, 0.5, 0.95):
        cfg = sample_building(rng, stage_at(p))
        assert cfg.seed < 8000 and cfg.lift.rollout
        assert 6 <= cfg.building.floors <= 32
    assert stage_at(0.1).disturbance_p == 0.0 and stage_at(0.9).hard_weight == 2.0


def test_statistics_helpers() -> None:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 1, 50), rng.normal(10.5, 1, 50)
    assert wilcoxon_p(a - b) == pytest.approx(wilcoxon(a - b).pvalue)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    r = paired(a, b)
    assert r.ci_low <= r.mean_diff <= r.ci_high and 0 <= r.win_rate <= 1
    covered = 0
    for s in range(100):
        x = np.random.default_rng(s).normal(0, 1, 40)
        lo, hi = bootstrap_ci(x, n_boot=1000, seed=s)
        covered += lo <= 0 <= hi
    assert covered >= 90


def test_award_hook_forcing_the_classical_choice_reproduces_the_classical_run() -> None:
    from elevator_mas.config import ScenarioConfig
    from elevator_mas.model import ElevatorModel

    cfg = ScenarioConfig.load("lunch_two_way").model_copy(
        update={"strategy": "cnp_astar", "duration": 200}
    )
    plain = ElevatorModel(cfg)
    hooked = ElevatorModel(cfg)
    hooked.award_hook = lambda call, rec, view, viable: (
        min(viable, key=lambda b: (b.total, b.car_id)).car_id
    )
    for _ in range(200):
        plain.step()
        hooked.step()
    a, b = plain.latest_metrics.as_dict(), hooked.latest_metrics.as_dict()
    a.pop("compute_ms_per_tick")
    b.pop("compute_ms_per_tick")
    assert a == b


def test_reward_tracker_matches_recomputation() -> None:
    from elevator_mas.config import LiftConfig, ScenarioConfig
    from elevator_mas.learning.lift.rl.reward import RewardTracker
    from elevator_mas.model import ElevatorModel

    cfg = ScenarioConfig.load("morning_up_peak").model_copy(
        update={"strategy": "cnp_astar", "lift": LiftConfig()}
    )
    m = ElevatorModel(cfg)
    tr = RewardTracker(m)
    total = 0.0
    for _ in range(150):
        m.step()
        total += tr.tick_reward()
    c = tr.components
    assert total == pytest.approx(-(c["wait"] + c["ride"] + c["energy"] + c["threshold"]))
    assert c["energy"] == pytest.approx(0.02 * m.collector.energy.floors_travelled)
