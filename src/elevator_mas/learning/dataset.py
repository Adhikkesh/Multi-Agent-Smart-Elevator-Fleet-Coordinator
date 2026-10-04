"""Expert dataset loader, splits, batch iterator, and dataset statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


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


class ExpertDataset:
    """Memory-mapped or in-memory expert dataset over recorded shards."""

    def __init__(
        self,
        shards_dir_or_file: Path | str,
        split: str | None = None,  # "train", "val", "test", or None (all)
    ) -> None:
        p = Path(shards_dir_or_file)
        if p.is_file():
            shard_paths = [p]
        elif p.is_dir():
            shard_paths = sorted(p.glob("*.npz"))
        else:
            shard_paths = []

        if not shard_paths:
            raise FileNotFoundError(f"No .npz shards found at {p}")

        # Combine arrays across shards
        calls, cars, globs, masks, eligibles = [], [], [], [], []
        t_costs, t_waits, t_rides, t_crowds, t_energies = [], [], [], [], []
        winners, run_ids, ticks, splits = [], [], [], []
        meta_list = []

        for sp in shard_paths:
            data = np.load(sp, allow_pickle=True)
            calls.append(data["call"])
            cars.append(data["cars"])
            globs.append(data["glob"])
            masks.append(data["mask"])
            eligibles.append(data["eligible"])
            t_costs.append(data["teacher_cost"])
            t_waits.append(data["teacher_wait"])
            t_rides.append(data["teacher_ride"])
            t_crowds.append(data["teacher_crowding"])
            t_energies.append(data["teacher_energy"])
            winners.append(data["winner"])
            run_ids.append(data["run_id"])
            ticks.append(data["tick"])
            splits.append(data["split"])
            if "meta_json" in data:
                meta_list.append(str(data["meta_json"]))

        self.call = np.concatenate(calls, axis=0)
        self.cars = np.concatenate(cars, axis=0)
        self.glob = np.concatenate(globs, axis=0)
        self.mask = np.concatenate(masks, axis=0)
        self.eligible = np.concatenate(eligibles, axis=0)
        self.teacher_cost = np.concatenate(t_costs, axis=0)
        self.teacher_wait = np.concatenate(t_waits, axis=0)
        self.teacher_ride = np.concatenate(t_rides, axis=0)
        self.teacher_crowding = np.concatenate(t_crowds, axis=0)
        self.teacher_energy = np.concatenate(t_energies, axis=0)
        self.winner = np.concatenate(winners, axis=0)
        self.run_id = np.concatenate(run_ids, axis=0)
        self.tick = np.concatenate(ticks, axis=0)
        self.split_arr = np.concatenate(splits, axis=0)
        self.meta = json.loads(meta_list[0]) if meta_list else {}

        # Filter by split if requested
        if split is not None:
            split_code = {"train": 0, "val": 1, "test": 2}.get(split.lower(), -1)
            if split_code >= 0:
                idx = np.where(self.split_arr == split_code)[0]
                self._filter_indices(idx)

    def _filter_indices(self, idx: np.ndarray) -> None:
        self.call = self.call[idx]
        self.cars = self.cars[idx]
        self.glob = self.glob[idx]
        self.mask = self.mask[idx]
        self.eligible = self.eligible[idx]
        self.teacher_cost = self.teacher_cost[idx]
        self.teacher_wait = self.teacher_wait[idx]
        self.teacher_ride = self.teacher_ride[idx]
        self.teacher_crowding = self.teacher_crowding[idx]
        self.teacher_energy = self.teacher_energy[idx]
        self.winner = self.winner[idx]
        self.run_id = self.run_id[idx]
        self.tick = self.tick[idx]
        self.split_arr = self.split_arr[idx]

    def __len__(self) -> int:
        return len(self.winner)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "call": self.call[indices],
            "cars": self.cars[indices],
            "glob": self.glob[indices],
            "mask": self.mask[indices],
            "eligible": self.eligible[indices],
            "teacher_cost": self.teacher_cost[indices],
            "winner": self.winner[indices],
        }

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

    def hard_mask(self) -> np.ndarray:
        """Returns boolean mask where winner margin < 10% of best cost."""
        sorted_costs = np.sort(self.teacher_cost, axis=1)
        best = sorted_costs[:, 0]
        second = sorted_costs[:, 1]
        margin = second - best
        is_hard = (margin < (0.10 * best)) & (best < 1e5) & (self.winner >= 0)
        return is_hard

    def stats(self) -> DatasetStats:
        n = len(self)
        if n == 0:
            return DatasetStats(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        n_train = int(np.sum(self.split_arr == 0))
        n_val = int(np.sum(self.split_arr == 1))
        n_test = int(np.sum(self.split_arr == 2))

        all_refused = np.sum(self.winner == -1)
        all_refused_pct = float(all_refused / n * 100.0)

        elig_counts = np.sum(self.eligible, axis=1)
        single_elig = np.sum(elig_counts == 1)
        single_elig_pct = float(single_elig / n * 100.0)

        is_hard = self.hard_mask()
        hard_pct = float(np.sum(is_hard) / n * 100.0)

        # Baseline accuracies on non-refused decisions
        valid_idx = np.where(self.winner >= 0)[0]
        nearest_matches = 0
        lowest_eta_matches = 0

        if len(valid_idx) > 0:
            for i in valid_idx:
                w = self.winner[i]
                elig = self.eligible[i]
                dists = np.where(elig, self.cars[i, :, 20], 1e6)
                if np.argmin(dists) == w:
                    nearest_matches += 1

                etas = np.where(elig, self.cars[i, :, 19], 1e6)
                if np.argmin(etas) == w:
                    lowest_eta_matches += 1

            nearest_acc = float(nearest_matches / len(valid_idx) * 100.0)
            lowest_eta_acc = float(lowest_eta_matches / len(valid_idx) * 100.0)
        else:
            nearest_acc = 0.0
            lowest_eta_acc = 0.0

        return DatasetStats(
            total_decisions=n,
            train_count=n_train,
            val_count=n_val,
            test_count=n_test,
            all_refused_pct=all_refused_pct,
            single_eligible_pct=single_elig_pct,
            hard_decision_pct=hard_pct,
            nearest_baseline_acc=nearest_acc,
            lowest_eta_acc=lowest_eta_acc,
        )


def load_shards(path: Path | str, split: str | None = None) -> ExpertDataset:
    return ExpertDataset(path, split=split)
