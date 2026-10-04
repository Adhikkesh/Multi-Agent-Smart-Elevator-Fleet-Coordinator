"""On-the-fly batch augmentation for imitation training (pure torch, vectorised).

* **Random car permutation** — each decision's car axis is shuffled independently, and the
  labels move with it. The network is equivariant by construction; permuting the data also
  removes any accidental ordering in the recorded shards (car 0 is not special).
* **Car dropout** — with probability ``p`` per decision, 1..``max`` cars that are *not* a
  (tied) teacher winner are masked out, as long as two eligible cars remain. This teaches
  variable fleet sizes. The teacher's marginal cost of one car does not depend on the other
  cars, so the remaining labels stay exact. Dropping a winner is never allowed.
* **Feature noise** — Gaussian noise on *continuous* features only; one-hots, flags,
  directions and the masks are never perturbed.

Mirror augmentation (flipping floors, swapping UP/DOWN) is deliberately absent: the lobby is
at the bottom and up-peak/down-peak are not mirror images, so it would create states the
building can never be in.
"""

from __future__ import annotations

import torch
from torch import Tensor

from elevator_mas.learning.lift.config import AugmentConfig
from elevator_mas.learning.lift.losses import SENTINEL, TIE_EPS
from elevator_mas.learning.schema import FEATURE_NAMES

#: Continuous (noisable) features, by name; everything else is categorical or a flag.
_CONT_CALL = (
    "call_floor",
    "call_waiting",
    "call_urgency",
    "call_dist_to_lobby",
    "call_demand_share",
    "n_calls_open_norm",
)
_CONT_CARS = (
    "floor",
    "load",
    "space",
    "door_blocked",
    "n_assigned",
    "n_car_calls",
    "riders",
    "plan_stops",
    "plan_end_floor",
    "plan_end_eta",
    "dist_to_call",
    "signed_dist",
    "park_target_dist",
)
_CONT_GLOB = (
    "tick_in_scenario",
    "fleet_total_load",
    "fleet_idle_frac",
    "n_available_ratio",
)


def continuous_indices(token: str) -> list[int]:
    """Indices of the continuous features of a token ("call", "cars" or "glob")."""
    wanted = {"call": _CONT_CALL, "cars": _CONT_CARS, "glob": _CONT_GLOB}[token]
    names = FEATURE_NAMES[token]
    return [names.index(n) for n in wanted]


_SIGNED_CAR_COL = FEATURE_NAMES["cars"].index("signed_dist")


def permute_cars(batch: dict[str, Tensor], gen: torch.Generator) -> dict[str, Tensor]:
    """Shuffle every decision's car axis independently; labels follow the cars."""
    b, n = batch["mask"].shape
    perm = torch.argsort(torch.rand(b, n, generator=gen, device=batch["mask"].device), dim=1)
    out = dict(batch)
    for key in ("cars", "mask", "eligible", "teacher_cost", "parts"):
        x = batch[key]
        idx = perm if x.dim() == 2 else perm[:, :, None].expand(-1, -1, x.shape[2])
        out[key] = torch.gather(x, 1, idx)
    # winner w moves to position inv[w], where inv is the inverse permutation.
    inv = torch.argsort(perm, dim=1)
    w = batch["winner"]
    moved = torch.gather(inv, 1, w.clamp(min=0)[:, None]).squeeze(1)
    out["winner"] = torch.where(w >= 0, moved, w)
    return out


def drop_cars(
    batch: dict[str, Tensor], gen: torch.Generator, p: float, max_drop: int
) -> dict[str, Tensor]:
    """Mask out 1..max_drop non-winning eligible cars in a fraction ``p`` of decisions."""
    if p <= 0.0 or max_drop <= 0:
        return batch
    mask, elig, cost = batch["mask"], batch["eligible"], batch["teacher_cost"]
    b, _ = mask.shape
    dev = mask.device
    valid = mask & elig
    c = torch.where(valid, cost, torch.full_like(cost, float("inf")))
    best = c.min(dim=1, keepdim=True).values
    winners = valid & (c <= best + TIE_EPS)
    droppable = valid & ~winners

    n_valid = valid.sum(dim=1)
    room = torch.minimum(droppable.sum(dim=1), n_valid - 2).clamp(min=0)
    want = torch.randint(1, max_drop + 1, (b,), generator=gen, device=dev)
    chosen = torch.rand(b, generator=gen, device=dev) < p
    k = torch.where(chosen, torch.minimum(want, room), torch.zeros_like(room))

    # Rank droppable cars by a random key; drop the first k of each row.
    keys = torch.rand(mask.shape, generator=gen, device=dev).masked_fill(~droppable, 2.0)
    rank = torch.argsort(torch.argsort(keys, dim=1), dim=1)
    drop = droppable & (rank < k[:, None])

    out = dict(batch)
    out["mask"] = mask & ~drop
    out["eligible"] = elig & ~drop
    out["teacher_cost"] = torch.where(drop, torch.full_like(cost, SENTINEL * 10), cost)
    out["cars"] = batch["cars"].masked_fill(drop[:, :, None], 0.0)
    out["parts"] = batch["parts"].masked_fill(drop[:, :, None], 0.0)
    return out


def add_noise(batch: dict[str, Tensor], gen: torch.Generator, sigma: float) -> dict[str, Tensor]:
    """Gaussian noise on continuous features of real cars and of the call/global tokens.

    Noisy values are clipped back to the feature's range; categorical columns are copied
    through bit-exact.
    """
    if sigma <= 0.0:
        return batch
    out = dict(batch)
    for key in ("call", "glob", "cars"):
        x = batch[key]
        cols = torch.tensor(continuous_indices(key), device=x.device)
        is_cont = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
        is_cont[cols] = True
        noise = torch.randn(x.shape, generator=gen, device=x.device) * sigma
        if key == "cars":
            noise = noise * batch["mask"][:, :, None].to(x.dtype)
        lo = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        if key == "cars":
            lo[_SIGNED_CAR_COL] = -1.0
        noisy = torch.clamp(x + noise, min=lo, max=torch.ones_like(lo))
        out[key] = torch.where(is_cont, noisy, x)
    return out


def augment(
    batch: dict[str, Tensor], cfg: AugmentConfig, gen: torch.Generator
) -> dict[str, Tensor]:
    """Apply the configured augmentations in order: permute, drop, noise."""
    if cfg.permute:
        batch = permute_cars(batch, gen)
    batch = drop_cars(batch, gen, cfg.car_dropout_p, cfg.car_dropout_max)
    return add_noise(batch, gen, cfg.feature_noise)
