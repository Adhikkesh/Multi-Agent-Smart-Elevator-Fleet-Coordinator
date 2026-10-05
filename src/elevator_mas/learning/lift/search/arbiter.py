"""The dispatcher-side arbiter: PUCT look-ahead over the cars' learned bids.

Who deliberates, with what information? Cars still bid (reflex, decentralised, public data).
The dispatcher — the only agent that sees every bid — runs a short look-ahead as an *arbiter*
over contested calls. It never recomputes a car's bid and may only choose among cars that
PROPOSEd (refusals stay classical). It searches only when the decision is contested: at least
two candidates and the network's top two scores within ``tau_margin``; otherwise the lowest
bid wins exactly as in Contract Net.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from elevator_mas.domain import Bid, Direction, HallCall
from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.lift.runtime import LiftRuntime
from elevator_mas.learning.lift.search.mcts import PUCTSearch, SearchConfig, SearchResult
from elevator_mas.learning.lift.search.worldmodel import PublicView, build_world


def bid_score(b: Bid) -> float:
    """A learned bid back in the network's log-cost space."""
    return float(np.log1p(max(b.total, 0.0)))


_WARM = False


def warm_up() -> None:
    """Compile the twin's Numba kernels once per process, so no decision pays the JIT."""
    global _WARM
    if _WARM:
        return
    from elevator_mas.learning.twin.state import TwinSimulator

    sim = TwinSimulator(floors=6, cars=2, duration=30, seed=0)
    ctx, term = sim.advance()
    while not term:
        ctx, term = sim.step_action(0)
    _WARM = True


class MCTSArbiter:
    """Chooses the winner of contested auctions by PUCT look-ahead."""

    def __init__(self, cfg: SearchConfig, runtime: LiftRuntime | None, seed: int) -> None:
        warm_up()
        self.cfg = cfg
        self.runtime = runtime
        self.seed = seed
        self.searched = 0
        self.skipped = 0
        self.overrides = 0
        self.fallbacks = 0

    # ---- network prior for deeper nodes
    def _prior(self, ctx: Any) -> tuple[list[int], np.ndarray, float]:
        enc = encode_decision(ctx)
        valid = enc.mask & enc.eligible
        cands = [int(i) for i in np.flatnonzero(valid)]
        if self.runtime is None or not cands:
            return cands, np.full(len(cands), 1.0 / max(len(cands), 1)), 0.0
        out = self.runtime.score(enc)
        s = out.score[0, cands].astype(float)
        logits = -(s - s.min()) * np.exp(-self.runtime.logit_temp)
        p = np.exp(logits - logits.max())
        return cands, p / p.sum(), float(out.value[0])

    def candidates(self, view: PublicView, call: HallCall, viable: list[Bid]) -> list[Bid]:
        """Top-k proposers by bid, plus the nearest proposer as a cheap safety net."""
        ranked = sorted(viable, key=lambda b: (b.total, b.car_id))
        keep = ranked[: self.cfg.top_k]
        floors = {s.car_id: s.floor for s in view.statuses}
        nearest = min(viable, key=lambda b: (abs(floors.get(b.car_id, 0) - call.floor), b.car_id))
        if nearest not in keep:
            keep.append(nearest)
        return keep

    def contested(self, cands: list[Bid]) -> bool:
        if len(cands) < 2:
            return False
        s = sorted(bid_score(b) for b in cands)
        return (s[1] - s[0]) < self.cfg.tau_margin * max(1.0, abs(s[0]))

    def choose(
        self, view: PublicView, call: HallCall, viable: list[Bid]
    ) -> tuple[int, SearchResult | None]:
        ranked = sorted(viable, key=lambda b: (b.total, b.car_id))
        net_choice = ranked[0].car_id
        cands = self.candidates(view, call, viable)
        if not self.contested(cands):
            self.skipped += 1
            return net_choice, None
        scores = np.array([bid_score(b) for b in cands])
        temp = self.runtime.logit_temp if self.runtime is not None else 0.0
        logits = -(scores - scores.min()) * np.exp(-temp)
        prior = np.exp(logits - logits.max())
        prior /= prior.sum()
        root_call = (call.floor, 1 if call.direction is Direction.UP else -1)

        def build(s: int) -> Any:
            ss = np.random.SeedSequence([self.seed, view.tick, call.floor, root_call[1] + 1, s])
            return build_world(view, root_call, ss, self.cfg.horizon)

        search = PUCTSearch(self.cfg, self._prior if self.runtime is not None else None)
        result = search.run(build, [b.car_id for b in cands], prior, net_choice)
        self.searched += 1
        self.overrides += int(result.overridden)
        self.fallbacks += int(result.fallback)
        return result.chosen, result
