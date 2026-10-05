"""The learned bidder: how a LiftZero network lives inside the multi-agent system.

Phase 1's design rule is that agents never call each other's methods: a car answers a CFP
from its inbox using its own program and the *public* status board. LiftZero keeps that
rule. When the strategy's ``bidder`` is ``"learned"``, each car prices the call by running
the (shared) network on public data only:

* the CFP content — call, urgency, number waiting;
* ``model.board.cars()`` — every car's published :class:`CarStatus`;
* ``model.board.policy`` — the traffic monitor's published pattern, demand and weights;
* the static building configuration (floors, lobby, capacity, timing).

It never touches a car's, floor's or passenger's private attributes; a test runs it on a
board-only stub whose every other attribute access raises.

Inference modes (``ScenarioConfig.lift.inference``):

* ``"shared"`` (default) — one forward pass per CFP, cached by conversation id; every car
  reads its own row. The weights are shared and the input is public, so this is
  mathematically the same as each car running the net itself, N times cheaper.
* ``"per_car"`` — every car runs its own forward pass (literal decentralised execution).
  Both modes produce identical bids (tested).

Eligibility is never learned: cars that cannot take the call still answer REFUSE through
the classical ``_refusal`` rule before the bidder is ever consulted.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from elevator_mas.domain import Bid, Direction, HallCall
from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.lift.card import ModelCardError
from elevator_mas.learning.lift.config import ROOT
from elevator_mas.learning.lift.runtime import LiftRuntime, NetOutNp
from elevator_mas.learning.view import from_board

#: Upper clip of a learned bid (the teacher's real bids stay far below this).
MAX_BID = 1.0e5

#: Where a strategy's model lives when it does not name one.
DEFAULT_MODEL = "models/liftzero_bc_v1.onnx"


class StrategyUnavailableError(ValueError):
    """A learned strategy was selected but its model cannot be loaded."""


def resolve_model_path(path: str | None) -> Path:
    p = Path(path or DEFAULT_MODEL)
    return p if p.is_absolute() else ROOT / p


_RUNTIMES: dict[Path, LiftRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_runtime(path: str | None) -> LiftRuntime:
    """A process-wide cached runtime per model file (sessions are thread-safe)."""
    p = resolve_model_path(path)
    with _RUNTIMES_LOCK:
        rt = _RUNTIMES.get(p)
        if rt is None:
            try:
                rt = LiftRuntime(p)
            except ModelCardError as exc:
                raise StrategyUnavailableError(str(exc)) from exc
            _RUNTIMES[p] = rt
        return rt


@lru_cache(maxsize=32)
def _availability(path: str) -> tuple[bool, str]:
    try:
        get_runtime(path)
    except StrategyUnavailableError as exc:
        return False, str(exc)
    return True, ""


def strategy_availability(strategy: Any) -> tuple[bool, str]:
    """``(available, reason)`` for a strategy; classical strategies are always available."""
    if getattr(strategy, "bidder", "classical") != "learned":
        return True, ""
    return _availability(str(resolve_model_path(getattr(strategy, "model_path", None))))


@dataclass
class _Priced:
    out: NetOutNp
    car_ids: list[int]
    encoded: dict[str, np.ndarray] | None = None


class LearnedBidder:
    """Prices CFPs with a LiftZero network from public information only."""

    def __init__(self, model: Any, model_path: str | None = None, inference: str = "shared"):
        if inference not in ("shared", "per_car"):
            raise ValueError(f"unknown inference mode {inference!r}")
        self.model = model
        self.runtime = get_runtime(model_path)
        self.inference = inference
        self._cache: dict[Any, _Priced] = {}
        self.forward_passes = 0

    # The bidder reads *only* these attributes of the model (the board-only contract).
    def _price(self, call: HallCall, urgency: int, waiting: int) -> _Priced:
        board = self.model.board
        statuses = list(board.cars())
        ctx = from_board(
            call_floor=call.floor,
            call_direction=1 if call.direction is Direction.UP else -1,
            call_waiting=waiting,
            call_urgency=urgency,
            cars=statuses,
            policy=board.policy,
            building=self.model.config.building,
            tick=self.model.tick,
        )
        self.forward_passes += 1
        enc = encode_decision(ctx)
        arrays = {k: getattr(enc, k) for k in ("call", "cars", "glob", "mask", "eligible")}
        return _Priced(self.runtime.score(enc), [s.car_id for s in statuses], arrays)

    def priced(self, conversation_id: Any, call: HallCall, urgency: int, waiting: int) -> _Priced:
        """The network's view of one CFP (cached per conversation in shared mode)."""
        if self.inference == "per_car":
            return self._price(call, urgency, waiting)
        hit = self._cache.get(conversation_id)
        if hit is None:
            if len(self._cache) > 64:
                self._cache.clear()
            hit = self._cache[conversation_id] = self._price(call, urgency, waiting)
        return hit

    def bid(
        self,
        car_id: int,
        call: HallCall,
        urgency: int,
        waiting: int,
        conversation_id: Any = None,
    ) -> Bid:
        """Car ``car_id``'s learned PROPOSE for the call."""
        priced = self.priced(conversation_id, call, urgency, waiting)
        row = priced.car_ids.index(car_id)
        score = float(priced.out.score[0, row])
        aux = np.clip(np.expm1(priced.out.aux[0, row].astype(np.float64)), 0.0, MAX_BID)
        total = float(np.clip(np.expm1(score), 0.0, MAX_BID))
        return Bid(
            car_id=car_id,
            total=total,
            wait=float(aux[0]),
            ride=float(aux[1]),
            crowding=float(aux[2]),
            energy=float(aux[3]),
            eta=self._eta(car_id, call),
        )

    def _eta(self, car_id: int, call: HallCall) -> float:
        """Display-only ETA from the public status: finish the plan, then travel."""
        status = self.model.board.car(car_id)
        spf = float(self.model.config.timing.seconds_per_floor)
        if status is None:
            return 0.0
        if status.plan_end_floor is None or status.planned_stops == 0:
            return abs(status.floor - call.floor) * spf
        return float(status.plan_end_eta) + abs(status.plan_end_floor - call.floor) * spf

    def cached(self, conversation_id: Any) -> _Priced | None:
        """The network's view of a CFP, if it was priced in shared mode."""
        return self._cache.get(conversation_id)

    def score_of(self, conversation_id: Any, car_id: int) -> float | None:
        """The raw network score of a car for a cached CFP (for decision traces)."""
        hit = self._cache.get(conversation_id)
        if hit is None or car_id not in hit.car_ids:
            return None
        return float(hit.out.score[0, hit.car_ids.index(car_id)])
