"""Feature schema definition and constants for LiftZero learning components.

Single source of truth for observation dimensions, token formats, normalization
constants, and feature indices across offline datasets, the twin simulator, and Gym envs.

Version history:

* v1 — 14 call, 26 car and 6 global features.
* v2 — the global token gains the four public cost weights W1-W4 from the fleet policy on
  the status board. The A* teacher prices bids with these weights, and the ``full``
  strategy retunes them per traffic pattern while ``cnp_astar`` keeps the defaults; without
  them, identical feature vectors carried different teacher labels.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

FEATURE_VERSION: Final[int] = 2
MAX_CARS: Final[int] = 16

KC: Final[int] = 14
KCAR: Final[int] = 26
KG: Final[int] = 10

PATTERNS: Final[tuple[str, ...]] = (
    "up_peak",
    "down_peak",
    "two_way",
    "interfloor",
    "light",
)

DOOR_STATES: Final[tuple[str, ...]] = (
    "closed",
    "opening",
    "open",
    "closing",
)

FEATURE_NAMES: Final[dict[str, list[str]]] = {
    "call": [
        "call_floor",
        "call_dir",
        "call_is_lobby",
        "call_waiting",
        "call_urgency",
        "call_dist_to_lobby",
        "call_demand_share",
        "pattern_up_peak",
        "pattern_down_peak",
        "pattern_two_way",
        "pattern_interfloor",
        "pattern_light",
        "n_calls_open_norm",
        "fleet_size_norm",
    ],
    "cars": [
        "floor",
        "dir_up",
        "dir_down",
        "dir_idle",
        "load",
        "space",
        "available",
        "out_of_service",
        "fire_mode",
        "door_closed",
        "door_opening",
        "door_open",
        "door_closing",
        "door_blocked",
        "n_assigned",
        "n_car_calls",
        "riders",
        "plan_stops",
        "plan_end_floor",
        "plan_end_eta",
        "dist_to_call",
        "signed_dist",
        "heading_to_call",
        "call_on_route",
        "has_same_call",
        "park_target_dist",
    ],
    "glob": [
        "tick_in_scenario",
        "floors_norm",
        "capacity_norm",
        "n_available_ratio",
        "fleet_total_load",
        "fleet_idle_frac",
        "weight_wait",
        "weight_ride",
        "weight_crowding",
        "weight_energy",
    ],
}


#: Global-token weights are divided by this before clipping to [0, 1].
WEIGHT_SCALE: Final[float] = 2.0

#: Default cost weights (W1-W4) when a context does not carry the board's policy.
DEFAULT_WEIGHTS: Final[tuple[float, float, float, float]] = (1.0, 0.5, 0.3, 0.2)

#: Split codes stored in the shards. ``test_large`` holds the 33-40 floor x 7-8 car buildings.
SPLITS: Final[dict[str, int]] = {"train": 0, "val": 1, "test": 2, "test_large": 3}

#: Teacher codes stored per decision (index into this tuple; -1 = unknown).
TEACHERS: Final[tuple[str, ...]] = ("cnp_astar", "full")

#: Data-source codes stored per decision (index into this tuple).
SOURCES: Final[tuple[str, ...]] = ("expert", "dagger_r1", "dagger_r2", "dagger_r3")


def get_schema_hash() -> str:
    """Deterministic hash of feature schema for dataset metadata verification."""
    data = {
        "version": FEATURE_VERSION,
        "max_cars": MAX_CARS,
        "kc": KC,
        "kcar": KCAR,
        "kg": KG,
        "names": FEATURE_NAMES,
    }
    dumped = json.dumps(data, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


SCHEMA_HASH: Final[str] = get_schema_hash()
