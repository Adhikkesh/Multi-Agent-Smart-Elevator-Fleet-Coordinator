"""Fast kernels for the twin simulator with optional Numba JIT acceleration.

If numba is available and ELEVATOR_NO_NUMBA != "1", kernels run with @njit.
Otherwise, the exact same Python functions run in pure Python.
"""

import os
from collections.abc import Callable
from typing import Any

import numpy as np

USE_NUMBA: bool = os.environ.get("ELEVATOR_NO_NUMBA", "0") != "1"
if USE_NUMBA:
    try:
        from numba import njit
    except ImportError:
        USE_NUMBA = False

if not USE_NUMBA:

    def njit(*args: Any, **_kwargs: Any) -> Callable[..., Any]:  # type: ignore
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True, fastmath=False)
def look_route(
    current_floor: int,
    current_dir: int,  # +1 UP, -1 DOWN, 0 IDLE
    stops: np.ndarray,  # 1D int32 array
) -> np.ndarray:
    """Compute intra-car stop sequence following the classical LOOK sweep algorithm.

    Sweeps in current_dir to the extreme stop, then reverses to serve stops behind.
    """
    n = len(stops)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n == 1:
        return stops.copy()

    # If IDLE, choose direction toward nearest stop
    if current_dir == 0:
        best_d = 999999
        best_dir = 1
        for i in range(n):
            s = stops[i]
            d = abs(s - current_floor)
            if d < best_d:
                best_d = d
                best_dir = 1 if s >= current_floor else -1
        current_dir = best_dir

    if current_dir >= 0:
        ups = [stops[i] for i in range(n) if stops[i] >= current_floor]
        downs = [stops[i] for i in range(n) if stops[i] < current_floor]
        ups.sort()
        downs.sort()
        downs.reverse()
        res = np.empty(n, dtype=np.int32)
        idx = 0
        for s in ups:
            res[idx] = s
            idx += 1
        for s in downs:
            res[idx] = s
            idx += 1
        return res
    downs = [stops[i] for i in range(n) if stops[i] <= current_floor]
    ups = [stops[i] for i in range(n) if stops[i] > current_floor]
    downs.sort()
    downs.reverse()
    ups.sort()
    res = np.empty(n, dtype=np.int32)
    idx = 0
    for s in downs:
        res[idx] = s
        idx += 1
    for s in ups:
        res[idx] = s
        idx += 1
    return res


@njit(cache=True, fastmath=False)
def route_eta(
    current_floor: int,
    route: np.ndarray,
    seconds_per_floor: int,
    dwell: float,
    move_timer: int,
    dwell_remaining: int,
) -> tuple[int, float]:
    """Compute (plan_end_floor, plan_end_eta) matching ElevatorAgent._eta_to."""
    n = len(route)
    if n == 0:
        return current_floor, 0.0

    end_floor = int(route[-1])
    elapsed = float(move_timer + dwell_remaining)
    pos = current_floor

    for i in range(n):
        stop = int(route[i])
        elapsed += abs(stop - pos) * seconds_per_floor
        pos = stop
        elapsed += dwell

    return end_floor, elapsed
