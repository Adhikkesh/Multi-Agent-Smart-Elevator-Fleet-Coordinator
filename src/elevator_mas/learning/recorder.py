"""Decision recorder and parallel expert harvesting for LiftZero imitation learning.

Records expert teacher decisions from real Mesa simulation runs into compressed .npz shards.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from elevator_mas.comms.board import DecisionEvent
from elevator_mas.config import (
    BuildingConfig,
    DisturbanceEvent,
    ScenarioConfig,
    TimingConfig,
    TrafficConfig,
    TrafficPhase,
)
from elevator_mas.learning.features import encode_decision
from elevator_mas.learning.schema import (
    FEATURE_VERSION,
    MAX_CARS,
    PATTERNS,
    SCHEMA_HASH,
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
    """Split allocation: 0 = train, 1 = val, 2 = test."""
    if floors >= 33 and cars >= 7:
        return 2  # Held-out large regime exclusively in test
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
    phase_len = duration // n_phases
    phases: list[TrafficPhase] = []

    for _p_idx in range(n_phases):
        p_name = str(rng.choice(PATTERNS))
        phases.append(
            TrafficPhase(
                pattern=p_name,
                rate=rate * float(rng.uniform(0.8, 1.2)),
                duration=phase_len,
            )
        )

    prio_prob = 0.05 if rng.random() < 0.3 else 0.0
    events: list[DisturbanceEvent] = []

    if rng.random() < 0.25:
        dist_tick = int(rng.integers(duration // 4, duration * 3 // 4))
        if rng.random() < 0.6:
            c_fault = int(rng.integers(0, cars))
            events.append(DisturbanceEvent(tick=dist_tick, type="car_fault", car_id=c_fault))
            repair_tick = dist_tick + int(rng.integers(60, 150))
            events.append(DisturbanceEvent(tick=repair_tick, type="car_repair", car_id=c_fault))
        else:
            events.append(DisturbanceEvent(tick=dist_tick, type="fire_alarm"))
            clear_tick = dist_tick + int(rng.integers(60, 120))
            events.append(DisturbanceEvent(tick=clear_tick, type="fire_clear"))

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


class DecisionRecorder:
    """Collects DecisionEvents from ElevatorModel and serializes to .npz shards."""

    def __init__(
        self,
        output_dir: Path | str,
        shard_prefix: str = "shard",
        shard_capacity: int = 50000,
        teacher: str = "mixed",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_prefix = shard_prefix
        self.shard_capacity = shard_capacity
        self.teacher = teacher
        self.shard_index = 0
        self.git_sha = get_git_sha()

        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self.buf_call = []
        self.buf_cars = []
        self.buf_glob = []
        self.buf_mask = []
        self.buf_eligible = []
        self.buf_teacher_cost = []
        self.buf_teacher_wait = []
        self.buf_teacher_ride = []
        self.buf_teacher_crowd = []
        self.buf_teacher_energy = []
        self.buf_winner = []
        self.buf_run_id = []
        self.buf_tick = []
        self.buf_split = []

    def __call__(self, event: DecisionEvent) -> None:
        """Hook callback for each auction decision in ElevatorModel."""
        ctx = from_event(event)
        enc = encode_decision(ctx)

        n_cars = event.building.cars
        costs = np.full(MAX_CARS, 1e6, dtype=np.float32)
        waits = np.zeros(MAX_CARS, dtype=np.float32)
        rides = np.zeros(MAX_CARS, dtype=np.float32)
        crowds = np.zeros(MAX_CARS, dtype=np.float32)
        energies = np.zeros(MAX_CARS, dtype=np.float32)

        for b in event.bids:
            cid = b.car_id
            if cid < MAX_CARS:
                if not b.refused and np.isfinite(b.total):
                    costs[cid] = b.total
                    waits[cid] = b.wait
                    rides[cid] = b.ride
                    crowds[cid] = b.crowding
                    energies[cid] = b.energy
                else:
                    costs[cid] = 1e6

        split = determine_split(event.seed, event.building.floors, n_cars)

        self.buf_call.append(enc.call)
        self.buf_cars.append(enc.cars)
        self.buf_glob.append(enc.glob)
        self.buf_mask.append(enc.mask)
        self.buf_eligible.append(enc.eligible)
        self.buf_teacher_cost.append(costs)
        self.buf_teacher_wait.append(waits)
        self.buf_teacher_ride.append(rides)
        self.buf_teacher_crowd.append(crowds)
        self.buf_teacher_energy.append(energies)
        self.buf_winner.append(-1 if event.winner is None else event.winner)
        self.buf_run_id.append(event.seed)
        self.buf_tick.append(event.tick)
        self.buf_split.append(split)

        if len(self.buf_winner) >= self.shard_capacity:
            self.flush()

    def flush(self) -> Path | None:
        if not self.buf_winner:
            return None

        out_path = self.output_dir / f"{self.shard_prefix}_{self.shard_index:04d}.npz"
        while out_path.exists():
            self.shard_index += 1
            out_path = self.output_dir / f"{self.shard_prefix}_{self.shard_index:04d}.npz"

        meta = {
            "feature_version": FEATURE_VERSION,
            "teacher": self.teacher,
            "schema_hash": SCHEMA_HASH,
            "git_sha": self.git_sha,
            "max_cars": MAX_CARS,
            "count": len(self.buf_winner),
        }

        np.savez_compressed(
            out_path,
            call=np.array(self.buf_call, dtype=np.float32),
            cars=np.array(self.buf_cars, dtype=np.float32),
            glob=np.array(self.buf_glob, dtype=np.float32),
            mask=np.array(self.buf_mask, dtype=bool),
            eligible=np.array(self.buf_eligible, dtype=bool),
            teacher_cost=np.array(self.buf_teacher_cost, dtype=np.float32),
            teacher_wait=np.array(self.buf_teacher_wait, dtype=np.float32),
            teacher_ride=np.array(self.buf_teacher_ride, dtype=np.float32),
            teacher_crowding=np.array(self.buf_teacher_crowd, dtype=np.float32),
            teacher_energy=np.array(self.buf_teacher_energy, dtype=np.float32),
            winner=np.array(self.buf_winner, dtype=np.int16),
            run_id=np.array(self.buf_run_id, dtype=np.int32),
            tick=np.array(self.buf_tick, dtype=np.int32),
            split=np.array(self.buf_split, dtype=np.int8),
            meta_json=json.dumps(meta),
        )

        self.shard_index += 1
        self._reset_buffer()
        return out_path
