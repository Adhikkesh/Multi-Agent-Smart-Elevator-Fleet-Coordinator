"""CPU inference for a shipped LiftZero model — numpy + onnxruntime only.

This is what runs inside the simulator at demo time; it never imports torch or onnx. One
``InferenceSession`` per runtime, created lazily under a lock (thread-safe), single
intra-op thread (the graph is tiny, so threading overhead dominates) and full graph
optimisation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from elevator_mas.learning.features import Encoded, EncodedBatch
from elevator_mas.learning.lift.card import (
    ModelCard,
    ModelCardError,
    card_path_for,
    load_card,
    validate_card,
)

#: Input names, in the order the graph declares them.
INPUTS: tuple[str, ...] = ("call", "cars", "glob", "mask", "eligible")


@dataclass
class NetOutNp:
    """Numpy mirror of the network outputs (see ``model.NetOut``)."""

    score: np.ndarray  # [B, N]
    aux: np.ndarray  # [B, N, 4]
    value: np.ndarray  # [B]
    attn: np.ndarray  # [B, N]


def _as_batch(enc: Encoded | EncodedBatch | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(enc, dict):
        arrays = {k: np.asarray(enc[k]) for k in INPUTS}
    else:
        arrays = {k: np.asarray(getattr(enc, k)) for k in INPUTS}
    if arrays["call"].ndim == 1:  # a single decision
        arrays = {k: v[None, ...] for k, v in arrays.items()}
    return arrays


class LiftRuntime:
    """A loaded LiftZero ONNX model with its validated model card."""

    def __init__(self, path: Path | str, validate_hash: bool = True) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ModelCardError(f"LiftZero model not found: {self.path}")
        self.card: ModelCard = load_card(self.path)
        validate_card(self.card, self.path if validate_hash else None)
        self._session: Any = None
        self._lock = threading.Lock()

    @property
    def card_path(self) -> Path:
        return card_path_for(self.path)

    @property
    def logit_temp(self) -> float:
        return float(self.card.logit_temp)

    def _get_session(self) -> Any:
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort

                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self._session = ort.InferenceSession(
                        str(self.path), sess_options=opts, providers=["CPUExecutionProvider"]
                    )
        return self._session

    def score(
        self, enc: Encoded | EncodedBatch | dict[str, np.ndarray], trim: bool = True
    ) -> NetOutNp:
        """Run the network on one decision or a batch.

        With ``trim`` the car axis is cut to the last real car (the encoder pads to
        ``MAX_CARS``); outputs are padded back so callers index cars as before. Padding never
        changes real cars' scores (tested), so trimming is purely a speed-up.
        """
        feeds = _as_batch(enc)
        n_full = feeds["mask"].shape[1]
        if trim:
            used = np.flatnonzero(feeds["mask"].any(axis=0))
            n = int(used.max()) + 1 if len(used) else 1
            feeds = {
                k: (v[:, :n] if k in ("cars", "mask", "eligible") else v) for k, v in feeds.items()
            }
        feeds = {
            k: (v.astype(np.bool_) if k in ("mask", "eligible") else v.astype(np.float32))
            for k, v in feeds.items()
        }
        score, aux, value, attn = self._get_session().run(None, feeds)
        if score.shape[1] < n_full:
            pad = n_full - score.shape[1]
            score = np.pad(score, ((0, 0), (0, pad)), constant_values=1.0e4)
            aux = np.pad(aux, ((0, 0), (0, pad), (0, 0)))
            attn = np.pad(attn, ((0, 0), (0, pad)))
        return NetOutNp(score=score, aux=aux, value=value, attn=attn)

    def choose(self, enc: Encoded | EncodedBatch | dict[str, np.ndarray]) -> np.ndarray:
        """argmin decision per row over mask & eligible (ties -> lowest car id; -1 if none)."""
        feeds = _as_batch(enc)
        out = self.score(feeds)
        valid = feeds["mask"].astype(bool) & feeds["eligible"].astype(bool)
        masked = np.where(valid, out.score, np.inf)
        return np.where(valid.any(axis=1), np.argmin(masked, axis=1), -1)


def random_feeds(n: int, rng: np.random.Generator, kc: int, kcar: int, kg: int) -> dict:
    """One valid-looking single decision with ``n`` real, eligible cars (for benchmarks)."""
    return {
        "call": rng.random((1, kc), dtype=np.float32),
        "cars": rng.random((1, n, kcar), dtype=np.float32),
        "glob": rng.random((1, kg), dtype=np.float32),
        "mask": np.ones((1, n), dtype=bool),
        "eligible": np.ones((1, n), dtype=bool),
    }


def bench_latency(
    rt: LiftRuntime, sizes: tuple[int, ...] = (2, 4, 8, 16, 32), reps: int = 2000, seed: int = 0
) -> list[dict[str, float]]:
    """Wall-clock latency of single-decision inference (ms) per fleet size."""
    import time

    from elevator_mas.learning.schema import KC, KCAR, KG

    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        feeds = [random_feeds(n, rng, KC, KCAR, KG) for _ in range(32)]
        for f in feeds[:10]:  # warm-up (session creation, allocator)
            rt.score(f, trim=False)
        times = np.empty(reps)
        for i in range(reps):
            f = feeds[i % len(feeds)]
            t0 = time.perf_counter()
            rt.score(f, trim=False)
            times[i] = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "n": n,
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
            }
        )
    return rows
