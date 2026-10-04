"""Tests for Gymnasium single and vectorised environments."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from elevator_mas.learning.env import ElevatorDecisionEnv, VectorDecisionEnv
from elevator_mas.learning.twin.policies import NearestPolicy


def test_gym_check_env() -> None:
    env = ElevatorDecisionEnv()
    check_env(env, skip_render_check=True)


def test_env_determinism() -> None:
    env1 = ElevatorDecisionEnv(regime="up_peak")
    env2 = ElevatorDecisionEnv(regime="up_peak")

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    pol = NearestPolicy()

    for _ in range(50):
        a1 = pol.act({"cars": obs1["cars"], "eligible": obs1["eligible"]})
        a2 = pol.act({"cars": obs2["cars"], "eligible": obs2["eligible"]})
        assert a1 == a2

        obs1, r1, term1, trunc1, _ = env1.step(int(a1))
        obs2, r2, term2, trunc2, _ = env2.step(int(a2))

        assert np.allclose(obs1["call"], obs2["call"])
        assert np.allclose(obs1["cars"], obs2["cars"])
        assert np.isclose(r1, r2)
        assert term1 == term2
        assert trunc1 == trunc2
        if trunc1:
            break


def test_invalid_action_fallback() -> None:
    env = ElevatorDecisionEnv()
    obs, info = env.reset(seed=42)

    # Force invalid action (e.g. index 15 when there are only 4 cars)
    obs, r, term, trunc, info = env.step(15)
    assert info["invalid_action"] is True
    assert not np.isnan(r)


def test_vector_decision_env() -> None:
    vec = VectorDecisionEnv(num_envs=4, regimes=["up_peak", "down_peak"], base_seed=500)
    obs = vec.reset()
    assert obs["call"].shape == (4, 14)
    assert obs["cars"].shape == (4, 16, 26)

    pol = NearestPolicy()
    for _ in range(30):
        # Determine actions for batch
        actions = np.zeros(4, dtype=np.int32)
        for i in range(4):
            actions[i] = pol.act({"cars": obs["cars"][i], "eligible": obs["eligible"][i]})

        obs, rewards, terms, truncs, infos = vec.step(actions)
        assert len(rewards) == 4
        assert not np.isnan(rewards).any()
