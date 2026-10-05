"""Plain helper functions shared by the LiftZero tests (importable, unlike conftest)."""

from __future__ import annotations

from typing import Any

import numpy as np

from elevator_mas.comms.board import CarStatus
from elevator_mas.domain import Direction


def make_status(
    car_id: int,
    floor: int,
    direction: Direction = Direction.IDLE,
    load: int = 0,
    capacity: int = 10,
    available: bool = True,
) -> CarStatus:
    return CarStatus(
        car_id=car_id,
        address=f"car-{car_id}",
        tick=0,
        floor=floor,
        direction=direction,
        load=load,
        capacity=capacity,
        space=capacity - load,
        available=available,
        out_of_service=not available,
        fire_mode=False,
        door="closed",
        door_blocked_ticks=0,
    )


def random_batch(
    b: int, n: int, seed: int = 0, kc: int = 14, kcar: int = 26, kg: int = 10
) -> dict[str, Any]:
    """Random torch inputs with car 0 always real and eligible."""
    import torch

    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(b, n, generator=g) > 0.2
    elig = torch.rand(b, n, generator=g) > 0.25
    mask[:, 0] = True
    elig[:, 0] = True
    return {
        "call": torch.rand(b, kc, generator=g),
        "cars": torch.rand(b, n, kcar, generator=g),
        "glob": torch.rand(b, kg, generator=g),
        "mask": mask,
        "eligible": elig & mask,
    }


def numpy_batch(b: int, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = rng.random((b, n)) > 0.2
    elig = rng.random((b, n)) > 0.25
    mask[:, 0] = True
    elig[:, 0] = True
    return {
        "call": rng.random((b, 14), dtype=np.float32),
        "cars": rng.random((b, n, 26), dtype=np.float32),
        "glob": rng.random((b, 10), dtype=np.float32),
        "mask": mask,
        "eligible": elig & mask,
    }
