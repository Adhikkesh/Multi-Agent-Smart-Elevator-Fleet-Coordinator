"""Decision recorder and parallel expert harvesting for LiftZero imitation learning.

Records expert teacher decisions from real Mesa simulation runs into compressed .npz shards.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from elevator_mas.comms.board import DecisionEvent
from elevator_mas.config import (
    ArrivalPhase,
    BuildingConfig,
    EventConfig,
    ScenarioConfig,
    TimingConfig,
    TrafficConfig,
)
from elevator_mas.domain import Bid
from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.schema import (
    FEATURE_VERSION,
    MAX_CARS,
    PATTERNS,
    SCHEMA_HASH,
    SOURCES,
    SPLITS,
    TEACHERS,
)
from elevator_mas.learning.view import from_event


def get_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()[:10]
    except Exception:
        return "unknown"


def determine_split(seed: int, floors: int, cars: int) -> int:
    """Split code by *run seed* (never by decision): 0 train, 1 val, 2 test, 3 test_large.

    The large 33-40 floor x 7-8 car buildings are only ever sampled for test seeds, and are
    held out as their own split so fleet-size generalisation can be measured separately.
    """
    if floors >= 33 and cars >= 7:
        return SPLITS["test_large"]
    if seed < 8000:
        return 0
    if seed < 8500:
        return 1
    return 2


def sample_random_regime(run_id: int, seed: int, teacher_choice: str) -> ScenarioConfig:
    """Sample a building, traffic schedule, and optional disturbance from domain ranges."""
    rng = np.random.default_rng(seed)

    split = determine_split(seed, 0, 0)
    if split == 2 and rng.random() < 0.35:
        # Scale test split: 33-40 floors, 7-8 cars
        floors = int(rng.integers(33, 41))
        cars = int(rng.integers(7, 9))
    else:
        floors = int(rng.integers(6, 33))
        # Ensure at least 1 car per ~8 floors
        min_cars = max(2, (floors + 7) // 8)
        cars = int(rng.integers(min_cars, min(min_cars + 3, 9)))

    capacity = int(rng.integers(6, 17))
    sec_per_floor = int(rng.integers(1, 4))
    lobby = 0

    u = float(rng.uniform(0.6, 1.8))
    rate = u * cars * 0.04

    # Pattern phases (1 to 4 phases)
    n_phases = int(rng.integers(1, 5))
    duration = int(rng.integers(600, 1801))
    phase_len = max(duration // n_phases, 1)
    phases: list[ArrivalPhase] = []

    curr_until = 0
    for p_idx in range(n_phases):
        p_name = str(rng.choice(PATTERNS))
        curr_until = duration if p_idx == n_phases - 1 else (curr_until + phase_len)
        phases.append(
            ArrivalPhase(
                until=curr_until,
                pattern=p_name,  # type: ignore
                rate=rate * float(rng.uniform(0.8, 1.2)),
            )
        )

    prio_prob = 0.05 if rng.random() < 0.3 else 0.0
    events: list[EventConfig] = []

    if rng.random() < 0.25:
        dist_tick = int(rng.integers(duration // 4, duration * 3 // 4))
        if rng.random() < 0.6:
            c_fault = int(rng.integers(0, cars))
            events.append(EventConfig(tick=dist_tick, kind="car_fault", car=c_fault))
            repair_tick = dist_tick + int(rng.integers(60, 150))
            events.append(EventConfig(tick=repair_tick, kind="car_repair", car=c_fault))
        else:
            events.append(EventConfig(tick=dist_tick, kind="fire_alarm"))
            clear_tick = dist_tick + int(rng.integers(60, 120))
            events.append(EventConfig(tick=clear_tick, kind="fire_clear"))

    if teacher_choice == "mixed":
        strat = "full" if rng.random() < 0.5 else "cnp_astar"
    else:
        strat = teacher_choice

    building_cfg = BuildingConfig(floors=floors, cars=cars, capacity=capacity, lobby=lobby)
    timing_cfg = TimingConfig(seconds_per_floor=sec_per_floor, door_open=2, door_close=2)
    traffic_cfg = TrafficConfig(
        rate=rate,
        pattern=phases[0].pattern,
        phases=phases,
        priority_probability=prio_prob,
        priority_weight=3.0,
    )

    return ScenarioConfig(
        name=f"harvest_{run_id}_{seed}",
        strategy=strat,
        seed=seed,
        duration=duration,
        building=building_cfg,
        timing=timing_cfg,
        traffic=traffic_cfg,
        events=events,
    )


#: Names of the per-decision arrays a shard stores, in write order.
SHARD_ARRAYS: tuple[str, ...] = (
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
    "split",
    "teacher",
    "source",
)

_DTYPES: dict[str, type] = {
    "mask": bool,
    "eligible": bool,
    "winner": np.int16,
    "executed": np.int16,
    "run_id": np.int32,
    "tick": np.int32,
    "split": np.int8,
    "teacher": np.int8,
    "source": np.int8,
}


def teacher_code(strategy: str) -> int:
    """Index of the classical teacher behind ``strategy`` in ``TEACHERS`` (-1 if none).

    A learned strategy is labelled by the classical strategy it shadows (its ``teacher``),
    so DAgger data and expert data share one code space.
    """
    if strategy in TEACHERS:
        return TEACHERS.index(strategy)
    try:
        from elevator_mas.strategies import get_strategy

        teacher = getattr(get_strategy(strategy), "teacher", None)
    except ValueError:
        return -1
    return TEACHERS.index(teacher) if teacher in TEACHERS else -1


def label_winner(bids: tuple[Bid, ...] | list[Bid]) -> int:
    """The teacher's award for a set of bids: lowest total, ties to the lowest car id."""
    viable = [b for b in bids if not b.refused and np.isfinite(b.total)]
    if not viable:
        return -1
    return min(viable, key=lambda b: (b.total, b.car_id)).car_id


class DecisionRecorder:
    """Collects DecisionEvents from ElevatorModel and serializes them to .npz shards.

    Labels always come from the *classical* teacher: the auction's own bids for a classical
    strategy, or the shadow bids for a learned strategy run with ``shadow_teacher`` on (the
    DAgger setting, where the learner chooses the states and the teacher labels them). The
    award that was actually executed is stored separately in ``executed``.
    """

    def __init__(
        self,
        output_dir: Path | str,
        shard_prefix: str = "shard",
        shard_capacity: int = 50000,
        teacher: str = "mixed",
        source: str = "expert",
    ) -> None:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_prefix = shard_prefix
        self.shard_capacity = shard_capacity
        self.teacher = teacher
        self.source = source
        self.shard_index = 0
        self.git_sha = get_git_sha()
        self.skipped_unlabelled = 0
        self.recorded_total = 0
        self._buf: dict[str, list[Any]] = {}
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self._buf = {name: [] for name in SHARD_ARRAYS}

    @property
    def buffered(self) -> int:
        """Decisions recorded since the last flush."""
        return len(self._buf["winner"])

    def __call__(self, event: DecisionEvent) -> None:
        """Hook callback for each auction decision in ElevatorModel."""
        if event.bidder == "learned" and event.shadow_bids is None:
            # A learned auction without shadow labels carries no teacher signal.
            self.skipped_unlabelled += 1
            return
        label_bids = event.shadow_bids if event.shadow_bids is not None else event.bids
        enc = encode_decision(from_event(event))

        costs = np.full(MAX_CARS, 1e6, dtype=np.float32)
        parts = {k: np.zeros(MAX_CARS, dtype=np.float32) for k in _PARTS}
        for b in label_bids:
            if b.car_id < MAX_CARS and not b.refused and np.isfinite(b.total):
                costs[b.car_id] = b.total
                for k in _PARTS:
                    parts[k][b.car_id] = getattr(b, k)

        floors, n_cars = event.building.floors, event.building.cars
        buf = self._buf
        buf["call"].append(enc.call)
        buf["cars"].append(enc.cars)
        buf["glob"].append(enc.glob)
        buf["mask"].append(enc.mask)
        buf["eligible"].append(enc.eligible)
        buf["teacher_cost"].append(costs)
        for k in _PARTS:
            buf[f"teacher_{k}"].append(parts[k])
        buf["winner"].append(label_winner(label_bids))
        buf["executed"].append(-1 if event.winner is None else event.winner)
        buf["run_id"].append(event.seed)
        buf["tick"].append(event.tick)
        buf["split"].append(determine_split(event.seed, floors, n_cars))
        buf["teacher"].append(teacher_code(event.strategy))
        buf["source"].append(SOURCES.index(self.source))
        self.recorded_total += 1

        if self.buffered >= self.shard_capacity:
            self.flush()

    def flush(self) -> Path | None:
        if not self.buffered:
            return None

        out_path = self.output_dir / f"{self.shard_prefix}_{self.shard_index:04d}.npz"
        while out_path.exists():
            self.shard_index += 1
            out_path = self.output_dir / f"{self.shard_prefix}_{self.shard_index:04d}.npz"

        meta = {
            "feature_version": FEATURE_VERSION,
            "teacher": self.teacher,
            "teachers": list(TEACHERS),
            "source": self.source,
            "sources": list(SOURCES),
            "splits": SPLITS,
            "schema_hash": SCHEMA_HASH,
            "git_sha": self.git_sha,
            "max_cars": MAX_CARS,
            "count": self.buffered,
        }
        arrays = {
            name: np.array(values, dtype=_DTYPES.get(name, np.float32))
            for name, values in self._buf.items()
        }
        np.savez_compressed(out_path, meta_json=json.dumps(meta), **arrays)

        self.shard_index += 1
        self._reset_buffer()
        return out_path


#: The four bid components a teacher reports, stored as ``teacher_<part>``.
_PARTS: tuple[str, ...] = ("wait", "ride", "crowding", "energy")


def harvest_run(config: ScenarioConfig, recorder: DecisionRecorder) -> int:
    """Run a single scenario with DecisionRecorder attached, returning decisions recorded."""
    from elevator_mas.model import ElevatorModel

    model = ElevatorModel(config=config, seed=config.seed)
    model.decision_hooks.append(recorder)
    before = recorder.recorded_total
    for _ in range(config.duration):
        model.step()
    return recorder.recorded_total - before


def _worker_harvest(
    worker_id: int,
    target_decisions: int,
    out_dir: Path,
    teacher: str,
    seed_start: int,
    shard_size: int,
) -> None:
    recorder = DecisionRecorder(
        output_dir=out_dir,
        shard_prefix=f"worker_{worker_id}",
        shard_capacity=shard_size,
        teacher=teacher,
    )
    total_decisions = 0
    run_idx = 0
    while total_decisions < target_decisions:
        seed = seed_start + run_idx * 100 + worker_id
        cfg = sample_random_regime(run_id=run_idx, seed=seed, teacher_choice=teacher)
        decisions = harvest_run(cfg, recorder)
        total_decisions += decisions
        run_idx += 1
    recorder.flush()


def record_expert_dataset(
    target_decisions: int,
    out_dir: Path | str,
    workers: int = 1,
    teacher: str = "mixed",
    seed_start: int = 0,
    shard_size: int = 50000,
) -> list[Path]:
    """Harvest expert decisions into compressed .npz shards."""
    import multiprocessing as mp

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        _worker_harvest(
            worker_id=0,
            target_decisions=target_decisions,
            out_dir=out_path,
            teacher=teacher,
            seed_start=seed_start,
            shard_size=shard_size,
        )
    else:
        decisions_per_worker = (target_decisions + workers - 1) // workers
        ctx = mp.get_context("spawn")
        processes = []
        for w in range(workers):
            p = ctx.Process(
                target=_worker_harvest,
                args=(
                    w,
                    decisions_per_worker,
                    out_path,
                    teacher,
                    seed_start,
                    shard_size,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    return sorted(out_path.glob("*.npz"))
