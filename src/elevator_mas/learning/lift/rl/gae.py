"""Generalised advantage estimation for a semi-Markov decision process.

Decisions arrive at irregular times. With ``Δt_t`` simulated seconds between decision ``t``
and the next one, the per-step discount is time-aware::

    γ_t = γ ** (Δt_t / τ)          (γ = 0.99 per τ = 5 s; Δt = 0 ⇒ γ_t = 1)
    δ_t = r_t + γ_t · V_{t+1} - V_t
    A_t = δ_t + γ_t · λ · A_{t+1}
    R_t = A_t + V_t                (value target)

At an episode's last decision ``V_{t+1}`` is the *bootstrap* value when the episode was
truncated by the horizon (a time limit is never treated as termination), and 0 only for a
true termination; the advantage recursion restarts at every episode boundary.
"""

from __future__ import annotations

import numpy as np


def smdp_discount(dt: np.ndarray, gamma: float, tau: float) -> np.ndarray:
    """Per-step discount γ^(Δt/τ)."""
    return np.power(gamma, np.asarray(dt, dtype=np.float64) / tau)


def _next_values(
    values: np.ndarray, episode_end: np.ndarray, terminated: np.ndarray, bootstrap: np.ndarray
) -> np.ndarray:
    nxt = np.zeros(len(values), dtype=np.float64)
    nxt[:-1] = values[1:]
    nxt = np.where(episode_end, np.where(terminated, 0.0, bootstrap), nxt)
    return nxt


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dt: np.ndarray,
    episode_end: np.ndarray,
    terminated: np.ndarray,
    bootstrap: np.ndarray,
    gamma: float = 0.99,
    tau: float = 5.0,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for a flat batch of consecutive episodes.

    Args:
        rewards: [T] reward accrued after each decision until the next one.
        values: [T] V(s_t).
        dt: [T] seconds from decision t to the next decision (or to the episode end).
        episode_end: [T] True for the last decision of an episode.
        terminated: [T] True if that end is a real termination (no bootstrap).
        bootstrap: [T] value to bootstrap from at a truncated episode end (ignored elsewhere).
    """
    n = len(rewards)
    disc = smdp_discount(dt, gamma, tau)
    nxt = _next_values(values, episode_end, terminated, bootstrap)
    deltas = rewards + disc * nxt - values
    adv = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        if episode_end[t]:
            acc = 0.0
        acc = deltas[t] + disc[t] * lam * acc
        adv[t] = acc
    return adv, adv + values


def gae_reference(
    rewards: np.ndarray,
    values: np.ndarray,
    dt: np.ndarray,
    episode_end: np.ndarray,
    terminated: np.ndarray,
    bootstrap: np.ndarray,
    gamma: float = 0.99,
    tau: float = 5.0,
    lam: float = 0.95,
) -> np.ndarray:
    """Slow O(T²) definition-level GAE (sum of discounted TD errors), for tests."""
    n = len(rewards)
    disc = smdp_discount(dt, gamma, tau)
    nxt = _next_values(values, episode_end, terminated, bootstrap)
    deltas = rewards + disc * nxt - values
    adv = np.zeros(n)
    for t in range(n):
        acc, w = 0.0, 1.0
        for k in range(t, n):
            acc += w * deltas[k]
            if episode_end[k]:
                break
            w *= disc[k] * lam
        adv[t] = acc
    return adv
