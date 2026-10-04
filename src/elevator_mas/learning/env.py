"""Gymnasium environment and vectorised wrapper for LiftZero semi-MDP learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.regimes import EvalRegime, load_regimes_config
from elevator_mas.learning.schema import KC, KCAR, KG, MAX_CARS
from elevator_mas.learning.twin.policies import CostGreedyPolicy
from elevator_mas.learning.twin.state import TwinSimulator


@dataclass
class RewardConfig:
    """Coefficients for the team reward shared across cars."""

    w_wait: float = 1.0 / 100.0
    w_ride: float = 0.5 / 100.0
    w_energy: float = 0.02
    w_threshold: float = 0.5
    threshold_sec: int = 60


class ElevatorDecisionEnv(gym.Env):
    """Gymnasium environment for elevator hall call assignment semi-MDP."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        regime: str | EvalRegime | dict[str, Any] | None = None,
        default_seed: int = 42,
    ) -> None:
        super().__init__()
        self.reward_config = reward_config or RewardConfig()
        self.regime_spec = regime or "up_peak"
        self.default_seed = default_seed

        self.action_space = spaces.Discrete(MAX_CARS)
        self.observation_space = spaces.Dict(
            {
                "call": spaces.Box(low=-1.0, high=1.0, shape=(KC,), dtype=np.float32),
                "cars": spaces.Box(low=-1.0, high=1.0, shape=(MAX_CARS, KCAR), dtype=np.float32),
                "glob": spaces.Box(low=0.0, high=1.0, shape=(KG,), dtype=np.float32),
                "mask": spaces.MultiBinary(MAX_CARS),
                "eligible": spaces.MultiBinary(MAX_CARS),
            }
        )

        self.sim: TwinSimulator | None = None
        self.current_ctx: Any = None
        self.current_obs: dict[str, np.ndarray] | None = None
        self.cost_greedy_fallback = CostGreedyPolicy()
        self.crossed_threshold_ids: set[int] = set()

    def _resolve_regime(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        opt_reg = options.get("regime") if options else None
        target = opt_reg or self.regime_spec

        if isinstance(target, dict):
            return target
        if isinstance(target, EvalRegime):
            return {
                "floors": target.floors,
                "cars": target.cars,
                "capacity": target.capacity,
                "duration": target.duration,
                "rate": target.rate,
                "pattern": target.pattern,
            }
        if isinstance(target, str):
            cfg = load_regimes_config()
            if target in cfg.eval_regimes:
                r = cfg.eval_regimes[target]
                return {
                    "floors": r.floors,
                    "cars": r.cars,
                    "capacity": r.capacity,
                    "duration": r.duration,
                    "rate": r.rate,
                    "pattern": r.pattern,
                }
            if target in cfg.scale:
                r = cfg.scale[target]
                return {
                    "floors": r.floors,
                    "cars": r.cars,
                    "capacity": r.capacity,
                    "duration": r.duration,
                    "rate": r.rate,
                    "pattern": r.pattern,
                }

        return {
            "floors": 15,
            "cars": 4,
            "capacity": 10,
            "duration": 900,
            "rate": 0.20,
            "pattern": "up_peak",
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        r_seed = seed if seed is not None else self.default_seed
        params = self._resolve_regime(options)

        self.sim = TwinSimulator(
            floors=params["floors"],
            cars=params["cars"],
            capacity=params["capacity"],
            duration=params["duration"],
            rate=params["rate"],
            pattern=params["pattern"],
            seed=r_seed,
        )
        self.crossed_threshold_ids.clear()

        ctx, term = self.sim.advance()
        self.current_ctx = ctx
        enc = encode_decision(ctx)
        self.current_obs = {
            "call": enc.call,
            "cars": enc.cars,
            "glob": enc.glob,
            "mask": enc.mask.astype(np.int8),
            "eligible": enc.eligible.astype(np.int8),
        }

        info = {
            "action_mask": enc.eligible,
            "invalid_action": False,
        }
        return self.current_obs, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.sim is not None, "Call reset() before step()"
        assert self.current_obs is not None

        mask = self.current_obs["eligible"]
        invalid = False
        act = int(action)

        if act < 0 or act >= MAX_CARS or not mask[act]:
            invalid = True
            act = int(self.cost_greedy_fallback.act(self.current_obs))

        # Snapshot before action
        t_before = self.sim.tick
        floors_before = sum(c.floors_travelled for c in self.sim.cars)

        # Step simulator
        next_ctx, terminated = self.sim.step_action(act)
        t_after = self.sim.tick
        floors_after = sum(c.floors_travelled for c in self.sim.cars)
        dt = max(t_after - t_before, 0)

        # Reward calculation:
        # r = -( Σ_wait Δt + 0.5 * Σ_ride Δt ) / 100 - 0.02 * floors - 0.5 * new_long_waits
        total_waiting = sum(len(q) for q in self.sim.waiting.values())
        total_riding = sum(len(c.riders) for c in self.sim.cars)
        floors_moved = floors_after - floors_before

        new_threshold_crossers = 0
        thresh = self.reward_config.threshold_sec
        for q in self.sim.waiting.values():
            for p in q:
                p_id = id(p)
                is_long = (t_after - p.arrived_tick) >= thresh
                if is_long and (p_id not in self.crossed_threshold_ids):
                    self.crossed_threshold_ids.add(p_id)
                    new_threshold_crossers += 1

        reward = -(
            (total_waiting * dt + 0.5 * total_riding * dt) * self.reward_config.w_wait
            + floors_moved * self.reward_config.w_energy
            + new_threshold_crossers * self.reward_config.w_threshold
        )

        truncated = terminated or (self.sim.tick >= self.sim.duration)

        if not truncated:
            self.current_ctx = next_ctx
            enc = encode_decision(next_ctx)
            self.current_obs = {
                "call": enc.call,
                "cars": enc.cars,
                "glob": enc.glob,
                "mask": enc.mask.astype(np.int8),
                "eligible": enc.eligible.astype(np.int8),
            }
            action_mask = enc.eligible
        else:
            action_mask = np.zeros(MAX_CARS, dtype=bool)

        info: dict[str, Any] = {
            "action_mask": action_mask,
            "invalid_action": invalid,
            "tick": self.sim.tick,
        }

        if truncated:
            info["episode_metrics"] = self.sim.get_metrics()

        return self.current_obs, float(reward), False, truncated, info


class VectorDecisionEnv:
    """Vectorised batch of N parallel elevator decision environments."""

    def __init__(
        self,
        num_envs: int = 64,
        regimes: list[str | dict[str, Any]] | None = None,
        base_seed: int = 1000,
    ) -> None:
        self.num_envs = num_envs
        self.regimes = regimes or ["up_peak"]
        self.base_seed = base_seed
        self.envs = [
            ElevatorDecisionEnv(
                regime=self.regimes[i % len(self.regimes)],
                default_seed=base_seed + i,
            )
            for i in range(num_envs)
        ]

    def reset(self, seeds: list[int] | None = None) -> dict[str, np.ndarray]:
        obs_list = []
        for i, env in enumerate(self.envs):
            s = seeds[i] if seeds else (self.base_seed + i)
            obs, _ = env.reset(seed=s)
            obs_list.append(obs)

        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []

        for i, env in enumerate(self.envs):
            obs, r, term, trunc, info = env.step(int(actions[i]))
            if trunc or term:
                obs, r_info = env.reset()
                info["final_metrics"] = info.get("episode_metrics", {})
                info["action_mask"] = r_info["action_mask"]

            obs_list.append(obs)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)

        batched_obs = {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}
        return (
            batched_obs,
            np.array(rewards, dtype=np.float32),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
            infos,
        )
