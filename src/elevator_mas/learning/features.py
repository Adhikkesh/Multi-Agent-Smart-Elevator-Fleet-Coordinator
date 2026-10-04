"""Feature encoder: encodes DecisionContext into fixed-shape NumPy arrays.

Produces:
  call: [KC=14] float32
  cars: [MAX_CARS=16, KCAR=26] float32
  glob: [KG=6] float32
  mask: [MAX_CARS=16] bool
  eligible: [MAX_CARS=16] bool
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elevator_mas.learning.schema import KC, KCAR, KG, MAX_CARS, PATTERNS
from elevator_mas.learning.view import DecisionContext


@dataclass(frozen=True)
class Encoded:
    """Encoded observation tensors for a single decision."""

    call: np.ndarray  # [KC] float32
    cars: np.ndarray  # [MAX_CARS, KCAR] float32
    glob: np.ndarray  # [KG] float32
    mask: np.ndarray  # [MAX_CARS] bool
    eligible: np.ndarray  # [MAX_CARS] bool


@dataclass(frozen=True)
class EncodedBatch:
    """Batched encoded observation tensors for B decisions."""

    call: np.ndarray  # [B, KC] float32
    cars: np.ndarray  # [B, MAX_CARS, KCAR] float32
    glob: np.ndarray  # [B, KG] float32
    mask: np.ndarray  # [B, MAX_CARS] bool
    eligible: np.ndarray  # [B, MAX_CARS] bool


def encode_decision(ctx: DecisionContext) -> Encoded:
    """Encode a single DecisionContext into normalized feature tensors."""
    f = ctx.floors
    fm1 = max(float(f - 1), 1.0)
    cap = max(float(ctx.capacity), 1.0)

    # ------------------------------------------------------------- 1. Call token [KC=14]
    call_arr = np.zeros(KC, dtype=np.float32)
    call_arr[0] = np.clip(float(ctx.call_floor) / fm1, 0.0, 1.0)
    call_arr[1] = 1.0 if ctx.call_direction > 0 else -1.0
    call_arr[2] = 1.0 if ctx.call_floor == ctx.lobby else 0.0
    call_arr[3] = np.clip(float(ctx.call_waiting) / 20.0, 0.0, 1.0)
    call_arr[4] = np.clip(float(ctx.call_urgency) / 4.0, 0.0, 1.0)
    call_arr[5] = np.clip(abs(float(ctx.call_floor - ctx.lobby)) / fm1, 0.0, 1.0)

    # Demand share
    if ctx.demand:
        sum_dem = sum(ctx.demand.values())
        if sum_dem > 0.0:
            call_arr[6] = np.clip(ctx.demand.get(ctx.call_floor, 0.0) / sum_dem, 0.0, 1.0)

    # Pattern one-hot [7..11]
    for idx, pat in enumerate(PATTERNS):
        if ctx.pattern == pat:
            call_arr[7 + idx] = 1.0
            break

    call_arr[12] = np.clip(float(ctx.open_calls_count) / 20.0, 0.0, 1.0)
    call_arr[13] = np.clip(float(ctx.fleet.n_cars) / float(MAX_CARS), 0.0, 1.0)

    # ------------------------------------------------------------- 2. Car tokens [MAX_CARS, KCAR=26]
    cars_arr = np.zeros((MAX_CARS, KCAR), dtype=np.float32)
    mask = np.zeros(MAX_CARS, dtype=bool)
    eligible = np.zeros(MAX_CARS, dtype=bool)

    fleet = ctx.fleet
    n_cars = min(fleet.n_cars, MAX_CARS)

    for i in range(n_cars):
        mask[i] = True
        c_floor = float(fleet.floors[i])
        c_dir = int(fleet.directions[i])
        c_load = float(fleet.loads[i])
        c_cap = max(float(fleet.capacities[i]), 1.0)
        c_space = max(c_cap - c_load, 0.0)
        c_avail = bool(fleet.availables[i])
        c_oos = bool(fleet.out_of_service[i])
        c_fire = bool(fleet.fire_mode[i])
        c_door = int(fleet.doors[i])

        # Eligibility: available and not oos and not fire_mode and space > 0
        is_elig = c_avail and (not c_oos) and (not c_fire) and (c_space > 0.0)
        eligible[i] = is_elig

        # 0. Floor
        cars_arr[i, 0] = np.clip(c_floor / fm1, 0.0, 1.0)

        # 1..3. Direction one-hot [UP, DOWN, IDLE]
        if c_dir > 0:
            cars_arr[i, 1] = 1.0
        elif c_dir < 0:
            cars_arr[i, 2] = 1.0
        else:
            cars_arr[i, 3] = 1.0

        # 4..5. Load and Space
        cars_arr[i, 4] = np.clip(c_load / c_cap, 0.0, 1.0)
        cars_arr[i, 5] = np.clip(c_space / c_cap, 0.0, 1.0)

        # 6..8. Flags
        cars_arr[i, 6] = 1.0 if c_avail else 0.0
        cars_arr[i, 7] = 1.0 if c_oos else 0.0
        cars_arr[i, 8] = 1.0 if c_fire else 0.0

        # 9..12. Door one-hot [closed, opening, open, closing]
        if 0 <= c_door <= 3:
            cars_arr[i, 9 + c_door] = 1.0
        else:
            cars_arr[i, 9] = 1.0

        # 13. Door blocked
        cars_arr[i, 13] = np.clip(float(fleet.door_blocked_ticks[i]) / 10.0, 0.0, 1.0)

        # 14..16. Assigned calls, car calls, riders
        cars_arr[i, 14] = np.clip(float(fleet.n_assigned[i]) / 10.0, 0.0, 1.0)
        cars_arr[i, 15] = np.clip(float(fleet.n_car_calls[i]) / 10.0, 0.0, 1.0)
        cars_arr[i, 16] = np.clip(float(fleet.riders[i]) / c_cap, 0.0, 1.0)

        # 17..19. Planned stops, plan end floor, plan end eta
        cars_arr[i, 17] = np.clip(float(fleet.planned_stops[i]) / 12.0, 0.0, 1.0)
        end_fl = float(fleet.plan_end_floors[i])
        cars_arr[i, 18] = np.clip(end_fl / fm1, 0.0, 1.0)
        cars_arr[i, 19] = np.clip(float(fleet.plan_end_etas[i]) / 240.0, 0.0, 1.0)

        # 20..21. Dist to call, signed dist
        d = float(ctx.call_floor) - c_floor
        cars_arr[i, 20] = np.clip(abs(d) / fm1, 0.0, 1.0)
        cars_arr[i, 21] = np.clip(d / fm1, -1.0, 1.0)

        # 22. Heading to call
        if c_dir > 0:
            cars_arr[i, 22] = 1.0 if ctx.call_floor >= c_floor else -1.0
        elif c_dir < 0:
            cars_arr[i, 22] = 1.0 if ctx.call_floor <= c_floor else -1.0
        else:
            cars_arr[i, 22] = 0.0

        # 23. Call on route (free pickup on current sweep)
        on_route = False
        if c_dir > 0 and ctx.call_direction > 0:
            if c_floor <= ctx.call_floor <= end_fl:
                on_route = True
        elif c_dir < 0 and ctx.call_direction < 0:
            if end_fl <= ctx.call_floor <= c_floor:
                on_route = True
        cars_arr[i, 23] = 1.0 if on_route else 0.0

        # 24. Has same call
        cars_arr[i, 24] = 1.0 if fleet.has_same_call[i] else 0.0

        # 25. Park target dist
        pt = float(fleet.park_targets[i])
        if pt >= 0.0:
            cars_arr[i, 25] = np.clip(abs(pt - float(ctx.call_floor)) / fm1, 0.0, 1.0)
        else:
            cars_arr[i, 25] = np.clip(abs(c_floor - float(ctx.call_floor)) / fm1, 0.0, 1.0)

    # ------------------------------------------------------------- 3. Global token [KG=6]
    glob_arr = np.zeros(KG, dtype=np.float32)
    glob_arr[0] = np.clip(float(ctx.tick) / 3600.0, 0.0, 1.0)
    glob_arr[1] = np.clip(float(f) / 40.0, 0.0, 1.0)
    glob_arr[2] = np.clip(cap / 20.0, 0.0, 1.0)

    if n_cars > 0:
        glob_arr[3] = np.clip(float(np.sum(fleet.availables[:n_cars])) / float(n_cars), 0.0, 1.0)
        tot_cap = float(np.sum(fleet.capacities[:n_cars]))
        tot_load = float(np.sum(fleet.loads[:n_cars]))
        glob_arr[4] = np.clip(tot_load / max(tot_cap, 1.0), 0.0, 1.0)

        idle_count = 0
        for i in range(n_cars):
            if fleet.availables[i] and fleet.n_assigned[i] == 0 and fleet.riders[i] == 0:
                idle_count += 1
        glob_arr[5] = np.clip(float(idle_count) / float(n_cars), 0.0, 1.0)

    return Encoded(
        call=call_arr,
        cars=cars_arr,
        glob=glob_arr,
        mask=mask,
        eligible=eligible,
    )


def encode_batch(contexts: list[DecisionContext]) -> EncodedBatch:
    """Vectorized encoding of a list of DecisionContext instances."""
    b = len(contexts)
    call_batch = np.zeros((b, KC), dtype=np.float32)
    cars_batch = np.zeros((b, MAX_CARS, KCAR), dtype=np.float32)
    glob_batch = np.zeros((b, KG), dtype=np.float32)
    mask_batch = np.zeros((b, MAX_CARS), dtype=bool)
    eligible_batch = np.zeros((b, MAX_CARS), dtype=bool)

    for idx, ctx in enumerate(contexts):
        enc = encode_decision(ctx)
        call_batch[idx] = enc.call
        cars_batch[idx] = enc.cars
        glob_batch[idx] = enc.glob
        mask_batch[idx] = enc.mask
        eligible_batch[idx] = enc.eligible

    return EncodedBatch(
        call=call_batch,
        cars=cars_batch,
        glob=glob_batch,
        mask=mask_batch,
        eligible=eligible_batch,
    )
