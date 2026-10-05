"""Rollout collection in the real multi-agent simulator (CPU worker processes).

Each worker owns a CPU copy of the actor and the privileged critic. For every Contract Net
auction of a rollout-mode model (``LiftConfig.rollout``), the dispatcher's
``award_hook`` asks the *policy* which of the proposing cars wins:

1. the decision is encoded from the announce-time public snapshot (exactly what a learned
   bidder sees — Phase 4 tests prove the two are identical);
2. the actor's masked softmax is sampled (training) or arg-minned (evaluation);
3. ``log π(a|s)``, ``V_priv(s, privileged)`` and ``V_pub(s)`` are stored with the action.

After every simulated tick the team reward (``reward.RewardTracker``) is added to the most
recent decision, so a transition's reward is everything that happened until the next
decision. Eligibility stays classical: the hook can only pick a car that proposed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from elevator_mas.config import LiftConfig, ScenarioConfig
from elevator_mas.domain import Direction
from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.lift.config import ModelConfig
from elevator_mas.learning.lift.model import LiftZeroNet
from elevator_mas.learning.lift.rl.curriculum import sample_building, stage_at
from elevator_mas.learning.lift.rl.policy import PrivilegedCritic, masked_logits
from elevator_mas.learning.lift.rl.privileged import KPRIV, privileged_features
from elevator_mas.learning.lift.rl.reward import RewardConfig, RewardTracker
from elevator_mas.learning.schema import KC, KCAR, KG
from elevator_mas.learning.view import from_board

#: Car slots stored per transition (training buildings have at most 8 cars).
N_PAD = 8

FIELDS = (
    "call",
    "cars",
    "glob",
    "mask",
    "valid",
    "priv",
    "action",
    "logp",
    "v_priv",
    "v_pub",
    "reward",
    "tick",
    "dt",
    "episode_end",
    "terminated",
    "bootstrap",
)


@dataclass
class Policy:
    """Actor + critic on CPU, rebuilt from state dicts in each worker."""

    net: LiftZeroNet
    critic: PrivilegedCritic

    @classmethod
    def build(cls, model_cfg: ModelConfig) -> Policy:
        net = LiftZeroNet(model_cfg).eval()
        critic = PrivilegedCritic(3 * model_cfg.d_model).eval()
        return cls(net, critic)

    def load(self, net_state: dict[str, Any], critic_state: dict[str, Any]) -> None:
        self.net.load_state_dict(net_state)
        self.critic.load_state_dict(critic_state)
        self.net.eval()
        self.critic.eval()


class Episode:
    """Runs one model under the policy and records its transitions."""

    def __init__(
        self,
        cfg: ScenarioConfig,
        policy: Policy,
        rng: np.random.Generator,
        deterministic: bool = False,
        reward_cfg: RewardConfig | None = None,
    ) -> None:
        from elevator_mas.model import ElevatorModel

        self.cfg = cfg
        self.policy = policy
        self.rng = rng
        self.deterministic = deterministic
        self.model = ElevatorModel(cfg)
        self.model.award_hook = self._hook
        self.tracker = RewardTracker(self.model, reward_cfg)
        self.rows: list[dict[str, Any]] = []
        self.fallbacks = 0

    @torch.no_grad()
    def _hook(self, call: Any, record: dict[str, Any], round_view: Any, viable: list[Any]) -> int:
        model = self.model
        statuses, policy = round_view
        ctx = from_board(
            call.floor,
            1 if call.direction is Direction.UP else -1,
            int(record.get("waiting", 0)),
            int(record.get("urgency", 0)),
            list(statuses),
            policy,
            model.config.building,
            model.tick,
        )
        enc = encode_decision(ctx)
        valid = np.zeros(N_PAD, dtype=bool)
        for b in viable:
            if b.car_id < N_PAD:
                valid[b.car_id] = True
        valid &= enc.mask[:N_PAD]
        if not valid.any():  # cannot happen for buildings of <= N_PAD cars
            self.fallbacks += 1
            return int(viable[0].car_id)
        t = {
            "call": torch.from_numpy(enc.call[None]),
            "cars": torch.from_numpy(enc.cars[None, :N_PAD]),
            "glob": torch.from_numpy(enc.glob[None]),
            "mask": torch.from_numpy(enc.mask[None, :N_PAD]),
        }
        valid_t = torch.from_numpy(valid[None])
        out = self.policy.net(t["call"], t["cars"], t["glob"], t["mask"], valid_t)
        logits = masked_logits(out, valid_t)[0]
        logp_all = torch.log_softmax(logits, dim=-1).numpy()
        if self.deterministic:
            action = int(np.argmin(np.where(valid, out.score[0].numpy(), np.inf)))
        else:
            p = np.exp(logp_all - logp_all.max())
            p = np.where(valid, p, 0.0)
            action = int(self.rng.choice(N_PAD, p=p / p.sum()))
        priv = privileged_features(model)
        pooled = out.pooled
        assert pooled is not None
        v_priv = float(self.policy.critic(pooled, torch.from_numpy(priv[None]))[0])
        self.rows.append(
            {
                "call": enc.call,
                "cars": enc.cars[:N_PAD],
                "glob": enc.glob,
                "mask": enc.mask[:N_PAD],
                "valid": valid,
                "priv": priv,
                "action": action,
                "logp": float(logp_all[action]),
                "v_priv": v_priv,
                "v_pub": float(out.value[0]),
                "reward": 0.0,
                "tick": model.tick,
            }
        )
        return action

    def run(self) -> None:
        for _ in range(self.cfg.duration):
            self.model.step()
            r = self.tracker.tick_reward()
            if self.rows:
                self.rows[-1]["reward"] += r
        end = self.model.tick
        for k, row in enumerate(self.rows):
            nxt = self.rows[k + 1]["tick"] if k + 1 < len(self.rows) else end
            row["dt"] = float(max(nxt - row["tick"], 0))
            row["episode_end"] = k == len(self.rows) - 1
            row["terminated"] = False  # the horizon truncates; it never terminates
            row["bootstrap"] = row["v_priv"]  # approximation: no decision at the horizon

    def metrics(self) -> dict[str, float]:
        m = self.model.latest_metrics
        return {
            "avg_wait": float(m.avg_wait),
            "p95_wait": float(m.p95_wait),
            "long_wait_pct": float(m.long_wait_pct),
            "energy": float(m.energy),
            "decisions": float(len(self.rows)),
            "return": float(sum(r["reward"] for r in self.rows)),
        }


def stack(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Rows -> arrays (empty arrays with the right shapes if there are no rows)."""
    if not rows:
        shapes = {
            "call": (0, KC),
            "cars": (0, N_PAD, KCAR),
            "glob": (0, KG),
            "mask": (0, N_PAD),
            "valid": (0, N_PAD),
            "priv": (0, KPRIV),
        }
        return {k: np.zeros(shapes.get(k, (0,)), dtype=np.float32) for k in FIELDS}
    out: dict[str, np.ndarray] = {}
    for k in FIELDS:
        vals = [r[k] for r in rows]
        if k in ("mask", "valid", "episode_end", "terminated"):
            out[k] = np.asarray(vals, dtype=bool)
        elif k == "action":
            out[k] = np.asarray(vals, dtype=np.int64)
        else:
            out[k] = np.asarray(vals, dtype=np.float32)
    return out


# ------------------------------------------------------------------- worker processes

_POLICY: Policy | None = None


def _init_worker(model_cfg: dict[str, Any]) -> None:
    global _POLICY
    torch.set_num_threads(1)
    _POLICY = Policy.build(ModelConfig(**model_cfg))


def collect_task(
    args: tuple[dict[str, Any], dict[str, Any], int, int, float, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """Worker task: run sampled training episodes until ``n_decisions`` are recorded."""
    net_state, critic_state, n_decisions, seed, progress, reward_cfg = args
    assert _POLICY is not None
    _POLICY.load(net_state, critic_state)
    rng = np.random.default_rng(seed)
    stage = stage_at(progress)
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, float]] = []
    while len(rows) < n_decisions:
        ep = Episode(
            sample_building(rng, stage), _POLICY, rng, reward_cfg=RewardConfig(**reward_cfg)
        )
        t0 = time.perf_counter()
        ep.run()
        rows.extend(ep.rows)
        stats.append({**ep.metrics(), "wall_s": time.perf_counter() - t0})
    return stack(rows), stats


def eval_task(
    args: tuple[dict[str, Any], dict[str, Any], str, str, int],
) -> dict[str, Any]:
    """Worker task: one deterministic real-simulator run of the policy on a fixed regime."""
    from elevator_mas.learning.lift.eval_sim import build_config

    net_state, critic_state, regime, strategy, seed = args
    assert _POLICY is not None
    _POLICY.load(net_state, critic_state)
    cfg = build_config(regime, strategy, seed, LiftConfig(rollout=True))
    ep = Episode(cfg, _POLICY, np.random.default_rng(seed), deterministic=True)
    violations: list[str] = []
    for _ in range(cfg.duration):
        ep.model.step()
        violations.extend(ep.model.violations())
    m = ep.model.latest_metrics
    return {
        "regime": regime,
        "seed": seed,
        "avg_wait": float(m.avg_wait),
        "p95_wait": float(m.p95_wait),
        "long_wait_pct": float(m.long_wait_pct),
        "energy": float(m.energy),
        "throughput": float(m.throughput),
        "violations": len(set(violations)),
    }
