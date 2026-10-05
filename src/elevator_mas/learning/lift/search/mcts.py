"""PUCT Monte-Carlo tree search over hall-call assignments.

A **decision node** holds the cars that may take a call, the network's prior over them, and
visit statistics ``N, W`` per car. One *simulation*:

1. take the ``s``-th sampled world (common random numbers, ``worldmodel.build_world``);
2. **select** from the root with PUCT::

       a* = argmax_a  Q̄(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))

   (``Q̄`` = Q min-max normalised to [0, 1] over the node; unvisited actions take the node's
   mean, not 0 — so the prior, not optimism, decides what is tried first);
3. apply the action in the twin and run it to the next decision; the next call is a *chance*
   outcome, so children are keyed by (action, next call);
4. **expand** a new child (priors from the network) and **evaluate** it — by a rollout of the
   cost-greedy base policy to the horizon (``leaf="rollout"``, works with the imitation model)
   or by the network's value head (``leaf="value"``, for the PPO model);
5. **back up** the return (negative team cost) along the path.

The search is *anytime*: it stops after ``sims`` simulations or when ``time_budget_ms`` is
spent, whichever comes first, and plays the most-visited root action (ties → higher Q, then
higher prior, then lower car id). No Dirichlet noise — decisions are deterministic given the
seed. Depth ≤ ``max_depth`` decisions and ≤ ``horizon`` simulated seconds.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.lift.search.worldmodel import SearchTwin
from elevator_mas.learning.twin.policies import CostGreedyPolicy


@dataclass(frozen=True)
class SearchConfig:
    sims: int = 32
    c_puct: float = 1.25
    max_depth: int = 3
    horizon: int = 60
    time_budget_ms: float = 50.0
    top_k: int = 3
    tau_margin: float = 0.5
    leaf: str = "rollout"  # "rollout" | "value"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    chosen: int
    net_choice: int
    candidates: list[int]
    prior: list[float]
    visits: list[int]
    q: list[float]
    root_value: float
    sims_done: int
    depth_max: int
    time_ms: float
    nn_ms: float
    leaf: str
    overridden: bool
    fallback: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Node:
    candidates: list[int]
    prior: np.ndarray
    n: np.ndarray = field(init=False)
    w: np.ndarray = field(init=False)
    children: dict[tuple[int, int, int], Node] = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = len(self.candidates)
        self.n = np.zeros(k)
        self.w = np.zeros(k)

    def q(self) -> np.ndarray:
        visited = self.n > 0
        q = np.zeros(len(self.n))
        q[visited] = self.w[visited] / self.n[visited]
        mean = float(self.w.sum() / self.n.sum()) if self.n.sum() > 0 else 0.0
        q[~visited] = mean
        return q

    def select(self, c_puct: float) -> int:
        q = self.q()
        lo, hi = float(q.min()), float(q.max())
        qn = (q - lo) / (hi - lo) if hi > lo else np.full(len(q), 0.5)
        total = float(self.n.sum())
        u = qn + c_puct * self.prior * math.sqrt(max(total, 1.0)) / (1.0 + self.n)
        return int(np.argmax(u))


Prior = Callable[[Any], tuple[list[int], np.ndarray, float]]


class PUCTSearch:
    """Anytime PUCT over a world builder; deterministic given the builder's seeds."""

    def __init__(
        self,
        cfg: SearchConfig,
        prior_fn: Prior | None = None,
    ) -> None:
        self.cfg = cfg
        self.prior_fn = prior_fn
        self.greedy = CostGreedyPolicy()
        self.nn_ms = 0.0

    def _expand(self, ctx: Any) -> Node:
        if self.prior_fn is not None:
            t0 = time.perf_counter()
            cands, prior, _ = self.prior_fn(ctx)
            self.nn_ms += (time.perf_counter() - t0) * 1000
        else:
            enc = encode_decision(ctx)
            cands = [int(i) for i in np.flatnonzero(enc.mask & enc.eligible)]
            prior = np.full(len(cands), 1.0 / max(len(cands), 1))
        if not cands:
            cands, prior = [0], np.array([1.0])
        k = self.cfg.top_k
        if len(cands) > k:
            keep = np.argsort(-prior)[:k]
            cands = [cands[i] for i in keep]
            prior = prior[keep] / prior[keep].sum()
        return Node(cands, np.asarray(prior, dtype=float))

    def _leaf_value(self, world: SearchTwin, ctx: Any, term: bool) -> float:
        if self.cfg.leaf == "value" and self.prior_fn is not None and ctx is not None and not term:
            t0 = time.perf_counter()
            _, _, value = self.prior_fn(ctx)
            self.nn_ms += (time.perf_counter() - t0) * 1000
            return float(value)
        start = world.cost
        while not term and ctx is not None:
            enc = encode_decision(ctx)
            a = int(self.greedy.act({"cars": enc.cars, "eligible": enc.eligible}))
            ctx, term = world.step_action(a)
        return -(world.cost - start)

    def run(
        self,
        build: Callable[[int], SearchTwin],
        root_candidates: list[int],
        root_prior: np.ndarray,
        net_choice: int,
    ) -> SearchResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        self.nn_ms = 0.0
        root = Node(list(root_candidates), np.asarray(root_prior, dtype=float))
        sims = depth_max = 0
        try:
            while sims < cfg.sims:
                world = build(sims)
                horizon_end = world.duration
                node, depth = root, 0
                path: list[tuple[Node, int, float]] = []
                while True:
                    a = node.select(cfg.c_puct)
                    before = world.cost
                    ctx, term = world.step_action(node.candidates[a])
                    reward = -(world.cost - before)
                    path.append((node, a, reward))
                    depth += 1
                    if term or ctx is None or depth >= cfg.max_depth or world.tick >= horizon_end:
                        value = self._leaf_value(world, ctx, term)
                        break
                    key = (a, ctx.call_floor, ctx.call_direction)
                    child = node.children.get(key)
                    if child is None:
                        node.children[key] = self._expand(ctx)
                        value = self._leaf_value(world, ctx, term)
                        break
                    node = child
                depth_max = max(depth_max, depth)
                g = value
                for nd, a, r in reversed(path):
                    g = r + g
                    nd.n[a] += 1
                    nd.w[a] += g
                sims += 1
                if (time.perf_counter() - t0) * 1000 >= cfg.time_budget_ms:
                    break
        except Exception:  # a broken world must never stall the dispatcher
            return self._result(root, net_choice, sims, depth_max, t0, fallback=True)
        return self._result(root, net_choice, sims, depth_max, t0)

    def _result(
        self, root: Node, net_choice: int, sims: int, depth: int, t0: float, fallback: bool = False
    ) -> SearchResult:
        q = root.q()
        if sims == 0 or fallback:
            chosen = net_choice
        else:
            order = sorted(
                range(len(root.candidates)),
                key=lambda i: (-root.n[i], -q[i], -root.prior[i], root.candidates[i]),
            )
            chosen = root.candidates[order[0]]
        total_n = float(root.n.sum())
        return SearchResult(
            chosen=chosen,
            net_choice=net_choice,
            candidates=list(root.candidates),
            prior=[float(p) for p in root.prior],
            visits=[int(v) for v in root.n],
            q=[float(v) for v in q],
            root_value=float(root.w.sum() / total_n) if total_n else 0.0,
            sims_done=sims,
            depth_max=depth,
            time_ms=(time.perf_counter() - t0) * 1000,
            nn_ms=self.nn_ms,
            leaf=self.cfg.leaf,
            overridden=chosen != net_choice,
            fallback=fallback,
        )
