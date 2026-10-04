"""Evaluation harness for policies across twin and real simulator backends."""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import pandas as pd

from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.regimes import EvalRegime, load_regimes_config
from elevator_mas.learning.twin.policies import (
    CollectivePolicy,
    CostGreedyPolicy,
    NearestPolicy,
    RandomEligiblePolicy,
)
from elevator_mas.learning.twin.state import TwinSimulator


class Policy(Protocol):
    name: str

    def act(self, obs: dict[str, np.ndarray]) -> int | np.ndarray:
        """Choose action given observation dictionary."""
        ...


BUILTIN_POLICIES: dict[str, type] = {
    "nearest": NearestPolicy,
    "collective": CollectivePolicy,
    "cost_greedy": CostGreedyPolicy,
    "random": RandomEligiblePolicy,
    "random_eligible": RandomEligiblePolicy,
}


def get_policy(name_or_instance: str | Policy) -> Policy:
    if isinstance(name_or_instance, str):
        key = name_or_instance.lower()
        if key in BUILTIN_POLICIES:
            return BUILTIN_POLICIES[key]()
        raise ValueError(f"Unknown policy {name_or_instance}; available: {list(BUILTIN_POLICIES)}")
    return name_or_instance


def _evaluate_twin(
    policy: Policy,
    regime: EvalRegime,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    sim = TwinSimulator(
        floors=regime.floors,
        cars=regime.cars,
        capacity=regime.capacity,
        duration=regime.duration,
        rate=regime.rate,
        pattern=regime.pattern,
        seed=seed,
    )

    ctx, term = sim.advance()
    decisions = 0
    while not term:
        enc = encode_decision(ctx)
        obs = {
            "call": enc.call,
            "cars": enc.cars,
            "glob": enc.glob,
            "mask": enc.mask,
            "eligible": enc.eligible,
        }
        act = policy.act(obs)
        ctx, term = sim.step_action(int(act))
        decisions += 1

    wall_s = time.perf_counter() - t0
    m = sim.get_metrics()
    return {
        "regime": regime.name,
        "backend": "twin",
        "policy": policy.name,
        "seed": seed,
        "avg_wait": m.get("avg_wait", 0.0),
        "p95_wait": m.get("p95_wait", 0.0),
        "max_wait": m.get("max_wait", 0.0),
        "long_wait_pct": m.get("long_wait_pct", 0.0),
        "throughput": m.get("throughput", 0.0),
        "energy": m.get("energy", 0.0),
        "decisions": decisions,
        "wall_s": wall_s,
    }


def _evaluate_real(
    strategy_name: str,
    regime: EvalRegime,
    seed: int,
) -> dict[str, Any]:
    from elevator_mas.config import (
        ArrivalPhase,
        BuildingConfig,
        ScenarioConfig,
        TrafficConfig,
    )
    from elevator_mas.sim import run_scenario

    cfg = ScenarioConfig(
        name=f"eval_{regime.name}_{seed}",
        strategy=strategy_name,
        seed=seed,
        duration=regime.duration,
        building=BuildingConfig(floors=regime.floors, cars=regime.cars, capacity=regime.capacity),
        traffic=TrafficConfig(
            phases=[ArrivalPhase(until=regime.duration, rate=regime.rate, pattern=regime.pattern)]  # type: ignore
        ),
    )

    t0 = time.perf_counter()
    res = run_scenario(cfg)
    wall_s = time.perf_counter() - t0
    m = res.metrics

    return {
        "regime": regime.name,
        "backend": "real",
        "policy": strategy_name,
        "seed": seed,
        "avg_wait": m.avg_wait,
        "p95_wait": m.p95_wait,
        "max_wait": m.max_wait,
        "long_wait_pct": m.long_wait_pct,
        "throughput": m.throughput,
        "energy": m.energy,
        "decisions": m.calls,
        "violations": len(res.violations),
        "wall_s": wall_s,
    }


def evaluate(
    policy_or_strategy: str | Policy,
    regimes: list[str | EvalRegime] | None = None,
    backend: str = "twin",
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate a policy across regimes and seeds."""
    cfg = load_regimes_config()
    seeds = seeds or cfg.val_seeds[:10]

    # Resolve regimes
    target_regimes: list[EvalRegime] = []
    if regimes is None:
        target_regimes = list(cfg.eval_regimes.values())
    else:
        for r in regimes:
            if isinstance(r, EvalRegime):
                target_regimes.append(r)
            elif isinstance(r, str):
                if r in cfg.eval_regimes:
                    target_regimes.append(cfg.eval_regimes[r])
                elif r in cfg.scale:
                    target_regimes.append(cfg.scale[r])
                else:
                    raise ValueError(f"Unknown regime {r}")

    rows: list[dict[str, Any]] = []

    if backend == "twin":
        pol = get_policy(policy_or_strategy)
        for reg in target_regimes:
            for s in seeds:
                rows.append(_evaluate_twin(pol, reg, s))
    elif backend == "real":
        strat_name = str(policy_or_strategy)
        for reg in target_regimes:
            for s in seeds:
                rows.append(_evaluate_real(strat_name, reg, s))
    else:
        raise ValueError(f"Unknown backend {backend}; must be 'twin' or 'real'")

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Compute mean and 95% bootstrap confidence intervals per regime and policy."""
    rng = np.random.default_rng(42)
    records = []

    for (reg, pol, backend), grp in df.groupby(["regime", "policy", "backend"]):
        n = len(grp)
        waits = grp["avg_wait"].to_numpy()

        if n > 1:
            boot_samples = rng.choice(waits, size=(n_boot, n), replace=True)
            boot_means = np.mean(boot_samples, axis=1)
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
        else:
            ci_low = float(waits[0])
            ci_high = float(waits[0])

        records.append(
            {
                "regime": reg,
                "policy": pol,
                "backend": backend,
                "samples": n,
                "avg_wait_mean": float(np.mean(waits)),
                "avg_wait_ci_low": float(ci_low),
                "avg_wait_ci_high": float(ci_high),
                "p95_wait_mean": float(grp["p95_wait"].mean()),
                "long_wait_pct_mean": float(grp["long_wait_pct"].mean()),
                "throughput_mean": float(grp["throughput"].mean()),
                "energy_mean": float(grp["energy"].mean()),
                "wall_s_sum": float(grp["wall_s"].sum()),
            }
        )

    return pd.DataFrame(records)
