"""Expert dataset loader, splits, batch iterator, and dataset statistics.

Shards are written by :class:`~elevator_mas.learning.recorder.DecisionRecorder`. Older (v1)
shards without the ``teacher``/``source``/``executed`` arrays still load: missing arrays are
filled with "unknown teacher", "expert" and the label winner respectively.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from elevator_mas.learning.schema import SOURCES, SPLITS, TEACHERS

#: Per-decision arrays every dataset carries (label arrays included).
ARRAY_FIELDS: tuple[str, ...] = (
    "call",
    "cars",
    "glob",
    "mask",
    "eligible",
    "teacher_cost",
    "teacher_wait",
    "teacher_ride",
    "teacher_crowding",
    "teacher_energy",
    "winner",
    "executed",
    "run_id",
    "tick",
    "split_arr",
    "teacher",
    "source",
)

#: Cost sentinel for refused / padded cars (never ``inf`` in stored labels).
REFUSED_COST = 1e6

#: Car-feature column indices used by the distance/ETA baselines.
_DIST_COL = 20
_ETA_COL = 19


@dataclass
class DatasetStats:
    total_decisions: int
    train_count: int
    val_count: int
    test_count: int
    all_refused_pct: float
    single_eligible_pct: float
    hard_decision_pct: float
    nearest_baseline_acc: float
    lowest_eta_acc: float
    test_large_count: int = 0


def _shard_paths(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.npz"))
    return []


class ExpertDataset:
    """In-memory expert dataset over one or more recorded shards (or shard directories)."""

    def __init__(
        self,
        shards_dir_or_file: Path | str | list[Path | str],
        split: str | None = None,  # "train", "val", "test", "test_large", or None (all)
    ) -> None:
        sources = (
            shards_dir_or_file if isinstance(shards_dir_or_file, list) else [shards_dir_or_file]
        )
        shard_paths: list[Path] = []
        for src in sources:
            shard_paths.extend(_shard_paths(Path(src)))
        if not shard_paths:
            raise FileNotFoundError(f"No .npz shards found at {shards_dir_or_file}")
        self.shard_paths = shard_paths

        parts: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_FIELDS}
        meta_list: list[dict[str, Any]] = []
        for sp in shard_paths:
            with np.load(sp, allow_pickle=False) as data:
                n = len(data["winner"])
                for name in ARRAY_FIELDS:
                    key = "split" if name == "split_arr" else name
                    if key in data:
                        parts[name].append(data[key])
                    else:
                        parts[name].append(_default_array(name, data, n))
                if "meta_json" in data:
                    meta_list.append(json.loads(str(data["meta_json"])))

        for name, arrays in parts.items():
            setattr(self, name, np.concatenate(arrays, axis=0))
        self.meta: dict[str, Any] = meta_list[0] if meta_list else {}
        self.metas = meta_list

        if split is not None:
            code = SPLITS.get(split.lower())
            if code is None:
                raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
            self._filter_indices(np.flatnonzero(self.split_arr == code))

    # Attributes populated dynamically above, declared for type checkers.
    call: np.ndarray
    cars: np.ndarray
    glob: np.ndarray
    mask: np.ndarray
    eligible: np.ndarray
    teacher_cost: np.ndarray
    teacher_wait: np.ndarray
    teacher_ride: np.ndarray
    teacher_crowding: np.ndarray
    teacher_energy: np.ndarray
    winner: np.ndarray
    executed: np.ndarray
    run_id: np.ndarray
    tick: np.ndarray
    split_arr: np.ndarray
    teacher: np.ndarray
    source: np.ndarray

    def _filter_indices(self, idx: np.ndarray) -> None:
        for name in ARRAY_FIELDS:
            setattr(self, name, getattr(self, name)[idx])

    def subset(self, keep: np.ndarray) -> ExpertDataset:
        """A filtered copy (boolean mask or index array); the original is untouched."""
        out = object.__new__(ExpertDataset)
        out.meta = self.meta
        out.metas = self.metas
        out.shard_paths = self.shard_paths
        idx = np.flatnonzero(keep) if keep.dtype == bool else keep
        for name in ARRAY_FIELDS:
            setattr(out, name, getattr(self, name)[idx])
        return out

    def by_teacher(self, teacher: str) -> ExpertDataset:
        """Only the decisions labelled by ``teacher`` (one of ``TEACHERS``)."""
        return self.subset(self.teacher == TEACHERS.index(teacher))

    def by_source(self, source: str) -> ExpertDataset:
        """Only the decisions from ``source`` (one of ``SOURCES``)."""
        return self.subset(self.source == SOURCES.index(source))

    def __len__(self) -> int:
        return len(self.winner)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Every per-decision array for ``indices`` (the split column as ``split``)."""
        out = {name: getattr(self, name)[indices] for name in ARRAY_FIELDS}
        out["split"] = out.pop("split_arr")
        return out

    def iter_batches(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Iterator[dict[str, np.ndarray]]:
        n = len(self)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for i in range(0, n, batch_size):
            b_idx = indices[i : min(i + batch_size, n)]
            yield self.batch(b_idx)

    # ----------------------------------------------------------------- analysis

    @property
    def n_eligible(self) -> np.ndarray:
        """Number of eligible (priced) cars per decision."""
        return np.sum(self.eligible & self.mask, axis=1)

    def nontrivial_mask(self) -> np.ndarray:
        """Decisions with a winner and at least two eligible cars to choose between."""
        return (self.winner >= 0) & (self.n_eligible >= 2)

    def hard_mask(self) -> np.ndarray:
        """Non-trivial decisions whose runner-up is within 10 % of the best cost.

        ``<=`` (not ``<``) so exact ties — including two cars both bidding 0 — count as hard.
        """
        sorted_costs = np.sort(self.teacher_cost, axis=1)
        best = sorted_costs[:, 0]
        second = sorted_costs[:, 1]
        margin = second - best
        return (margin <= 0.10 * best) & (second < REFUSED_COST / 10) & self.nontrivial_mask()

    def baseline_choice(self, column: int) -> np.ndarray:
        """argmin of one car-feature column over eligible cars (ties to the lowest car id)."""
        vals = np.where(self.eligible & self.mask, self.cars[:, :, column], np.inf)
        return np.argmin(vals, axis=1)

    def content_hash(self) -> str:
        """Short hash of the features and labels — identifies the exact training data."""
        h = hashlib.sha256()
        for name in ("call", "cars", "glob", "teacher_cost", "winner", "split_arr"):
            h.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return h.hexdigest()[:16]

    def stats(self) -> DatasetStats:
        n = len(self)
        if n == 0:
            return DatasetStats(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        valid = self.winner >= 0
        n_valid = int(valid.sum())

        def agree(choice: np.ndarray) -> float:
            if n_valid == 0:
                return 0.0
            return float(np.mean(choice[valid] == self.winner[valid]) * 100.0)

        return DatasetStats(
            total_decisions=n,
            train_count=int(np.sum(self.split_arr == SPLITS["train"])),
            val_count=int(np.sum(self.split_arr == SPLITS["val"])),
            test_count=int(np.sum(self.split_arr == SPLITS["test"])),
            all_refused_pct=float(np.mean(~valid) * 100.0),
            single_eligible_pct=float(np.mean(self.n_eligible == 1) * 100.0),
            hard_decision_pct=float(np.mean(self.hard_mask()) * 100.0),
            nearest_baseline_acc=agree(self.baseline_choice(_DIST_COL)),
            lowest_eta_acc=agree(self.baseline_choice(_ETA_COL)),
            test_large_count=int(np.sum(self.split_arr == SPLITS["test_large"])),
        )


def _default_array(name: str, data: Any, n: int) -> np.ndarray:
    """Fill-in for arrays that older shards do not carry."""
    if name == "executed":
        return np.asarray(data["winner"], dtype=np.int16)
    if name == "teacher":
        return np.full(n, -1, dtype=np.int8)
    if name == "source":
        return np.zeros(n, dtype=np.int8)
    raise KeyError(f"shard is missing required array {name!r}")


def load_shards(path: Path | str, split: str | None = None) -> ExpertDataset:
    return ExpertDataset(path, split=split)
