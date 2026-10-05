"""Cooperative PPO with a KL anchor to the imitation policy (written from scratch).

One update (``PPOTrainer.iteration``):

1. collect ``decisions_per_iter`` decisions with the current stochastic policy in the real
   simulator (``collector``), across worker processes;
2. SMDP-GAE advantages from the privileged critic ``V_priv`` (``gae``);
3. ``epochs`` passes of shuffled mini-batches minimising::

       L = −L_clip(ε)  +  c_v · Huber(V_priv − R)  +  c_d · MSE(V_pub − sg(V_priv))
           − c_e · H[π]  +  β · KL(π_BC ‖ π)

   with ``c_e`` and ``β`` annealed linearly (entropy 0.01 → 0.001; anchor 0.2 → 0 over the
   first 60 % of training) and the epoch loop stopped early when approx-KL(old‖new) exceeds
   ``target_kl``. Decisions with a single eligible car carry no policy gradient (skipped in
   the policy, entropy and KL terms) but still shape the value targets.

The anchor keeps the policy close to the imitation policy while the critic is still
learning, which prevents the early, noisy advantages from destroying a good warm start.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from elevator_mas.learning.lift.config import CONFIG_DIR, ROOT, ModelConfig
from elevator_mas.learning.lift.model import LiftZeroNet, count_parameters
from elevator_mas.learning.lift.rl import collector
from elevator_mas.learning.lift.rl.gae import gae
from elevator_mas.learning.lift.rl.policy import (
    PrivilegedCritic,
    entropy,
    kl_anchor,
    masked_logits,
)
from elevator_mas.learning.lift.rl.reward import RewardConfig
from elevator_mas.learning.lift.train_bc import load_checkpoint, save_checkpoint
from elevator_mas.learning.recorder import get_git_sha

Log = Callable[[str], None]

EVAL_REGIMES = ("up_peak", "down_peak", "two_way", "interfloor")


@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters of one PPO run (``configs/learning/ppo_*.yaml``)."""

    name: str = "ppo_default"
    init: str = "models/liftzero_bc_v1.ckpt.pt"
    total_decisions: int = 2_500_000
    decisions_per_iter: int = 8192
    workers: int = 7
    epochs: int = 4
    minibatch: int = 2048
    clip: float = 0.2
    c_value: float = 0.5
    c_distill: float = 0.5
    ent_start: float = 0.01
    ent_end: float = 0.001
    beta_start: float = 0.2
    beta_end: float = 0.0
    beta_frac: float = 0.6
    lr_start: float = 3e-4
    lr_end: float = 3e-5
    grad_clip: float = 0.5
    target_kl: float = 0.03
    gamma: float = 0.99
    tau: float = 5.0
    lam: float = 0.95
    reward_clip: float = 20.0
    eval_every: int = 10
    eval_seeds: int = 10
    eval_strategy: str = "liftzero_ppo"
    guard_pct: float = 10.0
    seed: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ppo_config(path: Path | str | None = None, preset: str | None = None) -> PPOConfig:
    if path is None:
        path = CONFIG_DIR / f"ppo_{preset or 'default'}.yaml"
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    reward = RewardConfig(**data.pop("reward", {}))
    known = {f.name for f in fields(PPOConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown PPO config keys: {sorted(unknown)}")
    return PPOConfig(**data, reward=reward)


def linear(start: float, end: float, progress: float, frac: float = 1.0) -> float:
    """Linear schedule from ``start`` to ``end`` over the first ``frac`` of training."""
    p = min(max(progress / max(frac, 1e-9), 0.0), 1.0)
    return start + (end - start) * p


# --------------------------------------------------------------------------- the loss


@dataclass
class LossCoefs:
    clip: float
    c_value: float
    c_distill: float
    c_entropy: float
    beta: float


def ppo_loss(
    net: LiftZeroNet,
    critic: PrivilegedCritic,
    anchor: LiftZeroNet | None,
    b: dict[str, torch.Tensor],
    k: LossCoefs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The PPO objective on one mini-batch (tensors already on the right device)."""
    valid, mask = b["valid"], b["mask"]
    out = net(b["call"], b["cars"], b["glob"], mask, valid)
    logits = masked_logits(out, valid)
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(1, b["action"][:, None]).squeeze(1)
    active = valid.sum(dim=1) >= 2
    w = active.float()
    n_act = w.sum().clamp(min=1.0)

    adv = b["adv"]
    if int(active.sum()) >= 64:
        a = adv[active]
        adv = (adv - a.mean()) / (a.std() + 1e-8)
    ratio = torch.exp(logp - b["logp"])
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - k.clip, 1 + k.clip) * adv)
    l_clip = (surr * w).sum() / n_act

    pooled = out.pooled
    assert pooled is not None
    v_priv = critic(pooled.detach(), b["priv"])
    l_value = F.huber_loss(v_priv, b["ret"])
    v_pub = net.value_head(pooled.detach()).squeeze(-1)
    l_distill = F.mse_loss(v_pub, v_priv.detach())

    ent = (entropy(logits, valid) * w).sum() / n_act
    if anchor is not None and k.beta > 0:
        with torch.no_grad():
            a_out = anchor(b["call"], b["cars"], b["glob"], mask, valid)
            a_logits = masked_logits(a_out, valid)
        l_kl = (kl_anchor(a_logits, logits, valid) * w).sum() / n_act
    else:
        l_kl = torch.zeros((), device=logits.device)

    loss = -l_clip + k.c_value * l_value + k.c_distill * l_distill - k.c_entropy * ent
    loss = loss + k.beta * l_kl
    with torch.no_grad():
        log_ratio = logp - b["logp"]
        approx_kl = (((ratio - 1) - log_ratio) * w).sum() / n_act
        clipfrac = (((ratio - 1).abs() > k.clip).float() * w).sum() / n_act
    return loss, {
        "loss": float(loss.detach()),
        "policy": float(-l_clip.detach()),
        "value": float(l_value.detach()),
        "distill": float(l_distill.detach()),
        "entropy": float(ent.detach()),
        "kl_anchor": float(l_kl.detach()),
        "approx_kl": float(approx_kl),
        "clipfrac": float(clipfrac),
    }


# ------------------------------------------------------------------------ the trainer


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = float(np.var(target))
    return float("nan") if var == 0 else 1.0 - float(np.var(target - pred)) / var


class PPOTrainer:
    """Owns the networks, optimiser, worker pool, logs and checkpoints of one run."""

    def __init__(
        self,
        cfg: PPOConfig,
        out_dir: Path | str | None = None,
        log: Log = print,
        resume: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.log = log
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        init = Path(cfg.init) if Path(cfg.init).is_absolute() else ROOT / cfg.init
        bc, self.bc_ckpt = load_checkpoint(init)
        self.model_cfg: ModelConfig = bc.cfg
        self.net = bc.train()
        self.anchor = copy.deepcopy(bc).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.critic = PrivilegedCritic(3 * self.model_cfg.d_model)
        self.opt = torch.optim.Adam(
            [*self.net.parameters(), *self.critic.parameters()], lr=cfg.lr_start
        )
        self.decisions = 0
        self.iteration = 0
        self.history: list[dict[str, Any]] = []
        self.evals: list[dict[str, Any]] = []
        self.best_score = math.inf
        self.baseline: dict[str, dict[str, float]] = {}
        self.reference: dict[str, dict[str, float]] = {}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(out_dir or ROOT / "runs") / f"{cfg.name}_s{cfg.seed}_{stamp}"
        if resume is not None:
            self._load(Path(resume))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pool: Any = None

    # ---- worker pool
    def __enter__(self) -> PPOTrainer:
        if self.cfg.workers > 0:
            self.pool = mp.get_context("spawn").Pool(
                self.cfg.workers,
                initializer=collector._init_worker,
                initargs=(asdict(self.model_cfg),),
            )
        else:
            collector._init_worker(asdict(self.model_cfg))
        return self

    def __exit__(self, *exc: object) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _map(self, fn: Callable[..., Any], jobs: list[Any]) -> list[Any]:
        if self.pool is None:
            return [fn(j) for j in jobs]
        return self.pool.map(fn, jobs, chunksize=1)

    def _states(self) -> tuple[dict[str, Any], dict[str, Any]]:
        net = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        crit = {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()}
        return net, crit

    # ---- one iteration
    @property
    def progress(self) -> float:
        return min(self.decisions / max(self.cfg.total_decisions, 1), 1.0)

    def collect(self) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        cfg = self.cfg
        n_workers = max(cfg.workers, 1)
        per = math.ceil(cfg.decisions_per_iter / n_workers)
        net, crit = self._states()
        seeds = self.rng.integers(0, 2**31 - 1, size=n_workers)
        jobs = [
            (net, crit, per, int(s), self.progress, cfg.reward.as_dict()) for s in seeds.tolist()
        ]
        results = self._map(collector.collect_task, jobs)
        arrays = {k: np.concatenate([r[0][k] for r in results]) for k in collector.FIELDS}
        stats = [s for r in results for s in r[1]]
        return arrays, stats

    def update(self, data: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = self.cfg
        rewards = np.clip(data["reward"], -cfg.reward_clip, 0.0).astype(np.float64)
        values = data["v_priv"].astype(np.float64)
        adv, ret = gae(
            rewards,
            values,
            data["dt"],
            data["episode_end"],
            data["terminated"],
            data["bootstrap"].astype(np.float64),
            cfg.gamma,
            cfg.tau,
            cfg.lam,
        )
        tensors = {
            "call": torch.from_numpy(data["call"]),
            "cars": torch.from_numpy(data["cars"]),
            "glob": torch.from_numpy(data["glob"]),
            "mask": torch.from_numpy(data["mask"]),
            "valid": torch.from_numpy(data["valid"]),
            "priv": torch.from_numpy(data["priv"]),
            "action": torch.from_numpy(data["action"]),
            "logp": torch.from_numpy(data["logp"]),
            "adv": torch.from_numpy(adv.astype(np.float32)),
            "ret": torch.from_numpy(ret.astype(np.float32)),
        }
        p = self.progress
        coefs = LossCoefs(
            clip=cfg.clip,
            c_value=cfg.c_value,
            c_distill=cfg.c_distill,
            c_entropy=linear(cfg.ent_start, cfg.ent_end, p),
            beta=linear(cfg.beta_start, cfg.beta_end, p, cfg.beta_frac),
        )
        lr = linear(cfg.lr_start, cfg.lr_end, p)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.net.train()
        n = len(rewards)
        sums: dict[str, float] = {}
        count = 0
        stopped = False
        epochs_done = 0
        for _ in range(cfg.epochs):
            order = torch.from_numpy(self.rng.permutation(n))
            kls = []
            for i in range(0, n, cfg.minibatch):
                idx = order[i : i + cfg.minibatch]
                b = {k: v[idx] for k, v in tensors.items()}
                loss, st = ppo_loss(self.net, self.critic, self.anchor, b, coefs)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.net.parameters(), *self.critic.parameters()], cfg.grad_clip
                )
                self.opt.step()
                for kk, vv in st.items():
                    sums[kk] = sums.get(kk, 0.0) + vv
                count += 1
                kls.append(st["approx_kl"])
            epochs_done += 1
            if float(np.mean(kls)) > cfg.target_kl:
                stopped = True
                break
        self.net.eval()
        out = {k: v / max(count, 1) for k, v in sums.items()}
        out.update(
            {
                "lr": lr,
                "beta": coefs.beta,
                "ent_coef": coefs.c_entropy,
                "epochs": epochs_done,
                "kl_stop": float(stopped),
                "explained_var": explained_variance(values, ret),
            }
        )
        return out

    # ---- evaluation and selection
    def evaluate(self, seeds: list[int]) -> dict[str, dict[str, float]]:
        net, crit = self._states()
        jobs = [(net, crit, r, self.cfg.eval_strategy, s) for r in EVAL_REGIMES for s in seeds]
        rows = self._map(collector.eval_task, jobs)
        df = {r: [x for x in rows if x["regime"] == r] for r in EVAL_REGIMES}
        return {
            r: {
                k: float(np.mean([x[k] for x in v]))
                for k in ("avg_wait", "p95_wait", "long_wait_pct", "energy", "violations")
            }
            for r, v in df.items()
        }

    def _teacher_baseline(self, seeds: list[int]) -> dict[str, dict[str, float]]:
        from elevator_mas.learning.lift.eval_sim import run_one

        teacher = "full" if self.cfg.eval_strategy == "liftzero_ppo" else "cnp_astar"
        jobs = [(teacher, r, s) for r in EVAL_REGIMES for s in seeds]
        rows = (
            [run_one(*j) for j in jobs] if self.pool is None else self.pool.starmap(run_one, jobs)
        )
        return {
            r: {
                k: float(np.mean([x[k] for x in rows if x["regime"] == r]))
                for k in ("avg_wait", "p95_wait", "long_wait_pct")
            }
            for r in EVAL_REGIMES
        }

    def score(self, ev: dict[str, dict[str, float]]) -> tuple[float, bool]:
        """Mean relative avg-wait vs the teacher (lower is better) and the fairness guard.

        A checkpoint is *ineligible* if its p95 or long-wait % is more than ``guard_pct``
        worse than the starting (imitation) policy's in any regime.
        """
        rel = [ev[r]["avg_wait"] / max(self.baseline[r]["avg_wait"], 1e-9) - 1 for r in ev]
        ok = True
        for r in ev:
            ref = self.reference[r]
            g = 1 + self.cfg.guard_pct / 100
            if ev[r]["p95_wait"] > ref["p95_wait"] * g:
                ok = False
            if ev[r]["long_wait_pct"] > max(ref["long_wait_pct"] * g, ref["long_wait_pct"] + 1.0):
                ok = False
            if ev[r]["violations"] > 0:
                ok = False
        return float(np.mean(rel)) * 100.0, ok

    # ---- checkpoints
    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        torch.save(
            {
                "net": self.net.state_dict(),
                "critic": self.critic.state_dict(),
                "opt": self.opt.state_dict(),
                "rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "decisions": self.decisions,
                "iteration": self.iteration,
                "history": self.history,
                "evals": self.evals,
                "best_score": self.best_score,
                "baseline": self.baseline,
                "reference": self.reference,
                "config": self.cfg.as_dict(),
                "model_config": asdict(self.model_cfg),
                **(extra or {}),
            },
            path,
        )

    def _load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["net"])
        self.critic.load_state_dict(ck["critic"])
        self.opt.load_state_dict(ck["opt"])
        self.rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        self.decisions = ck["decisions"]
        self.iteration = ck["iteration"]
        self.history = ck["history"]
        self.evals = ck["evals"]
        self.best_score = ck["best_score"]
        self.baseline = ck["baseline"]
        self.reference = ck["reference"]
        self.run_dir = path.parent

    def save_best(self, ev: dict[str, Any], score: float) -> None:
        save_checkpoint(
            self.run_dir / "best.pt",
            self.net,
            {
                "epoch": self.iteration,
                "val": {"closed_loop": ev, "score_pct": score},
                "train_config": {"name": self.cfg.name, "seed": self.cfg.seed, "kind": "ppo"},
                "data_hash": f"ppo:{self.decisions}",
                "critic": self.critic.state_dict(),
            },
        )

    # ---- main loop
    def train(self, max_iterations: int | None = None) -> dict[str, Any]:
        cfg = self.cfg
        t0 = time.perf_counter()
        seeds = list(range(8000, 8000 + cfg.eval_seeds))  # validation seeds only
        if not self.baseline:
            self.baseline = self._teacher_baseline(seeds)
            self.reference = self.evaluate(seeds)
            s, _ = self.score(self.reference)
            self.evals.append(
                {"iteration": 0, "decisions": 0, "score_pct": s, **_flat(self.reference)}
            )
            self.log(f"[ppo] start: imitation policy {s:+.2f} % vs teacher on val")
        self.log(
            f"[ppo] {cfg.name} seed={cfg.seed} params={count_parameters(self.net):,} "
            f"budget={cfg.total_decisions:,} decisions, {cfg.workers} workers"
        )
        iters = 0
        while self.decisions < cfg.total_decisions:
            ti = time.perf_counter()
            data, stats = self.collect()
            tc = time.perf_counter() - ti
            upd = self.update(data)
            self.decisions += len(data["reward"])
            self.iteration += 1
            iters += 1
            row = {
                "iteration": self.iteration,
                "decisions": self.decisions,
                "sps": len(data["reward"]) / max(tc, 1e-9),
                "episodes": len(stats),
                "ep_return": float(np.mean([s["return"] for s in stats])),
                "ep_avg_wait": float(np.mean([s["avg_wait"] for s in stats])),
                "ep_p95_wait": float(np.mean([s["p95_wait"] for s in stats])),
                "ep_long_wait_pct": float(np.mean([s["long_wait_pct"] for s in stats])),
                **upd,
                "wall_s": time.perf_counter() - t0,
            }
            self.history.append(row)
            self.log(
                f"[ppo] it {self.iteration:3d} dec {self.decisions:>9,} sps {row['sps']:6.0f} "
                f"ret {row['ep_return']:8.2f} wait {row['ep_avg_wait']:6.1f} "
                f"kl {upd['approx_kl']:.4f} klbc {upd['kl_anchor']:.4f} H {upd['entropy']:.3f} "
                f"ev {upd['explained_var']:.2f} β {upd['beta']:.3f}"
            )
            last = self.decisions >= cfg.total_decisions
            if self.iteration % cfg.eval_every == 0 or last:
                ev = self.evaluate(seeds)
                s, ok = self.score(ev)
                self.evals.append(
                    {
                        "iteration": self.iteration,
                        "decisions": self.decisions,
                        "score_pct": s,
                        "eligible": ok,
                        **_flat(ev),
                    }
                )
                self.log(f"[ppo] eval it {self.iteration}: {s:+.2f} % vs teacher (eligible={ok})")
                if ok and s < self.best_score:
                    self.best_score = s
                    self.save_best(ev, s)
            self.save(self.run_dir / "last.pt")
            self._write_logs()
            if max_iterations is not None and iters >= max_iterations:
                break
        if not (self.run_dir / "best.pt").exists():
            self.save_best(self.reference, float("nan"))
        info = self._run_info(time.perf_counter() - t0)
        (self.run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
        return info

    def _write_logs(self) -> None:
        for name, rows in (("metrics.csv", self.history), ("evals.csv", self.evals)):
            if not rows:
                continue
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with (self.run_dir / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)

    def _run_info(self, wall: float) -> dict[str, Any]:
        return {
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            "git_sha": get_git_sha(),
            "torch_version": torch.__version__,
            "decisions": self.decisions,
            "iterations": self.iteration,
            "wall_time_s": round(wall, 1),
            "best_score_pct": self.best_score,
            "baseline_teacher": self.baseline,
            "reference_bc": self.reference,
            "config": self.cfg.as_dict(),
        }


def _flat(ev: dict[str, dict[str, float]]) -> dict[str, float]:
    return {f"{r}_{k}": v for r, d in ev.items() for k, v in d.items()}
