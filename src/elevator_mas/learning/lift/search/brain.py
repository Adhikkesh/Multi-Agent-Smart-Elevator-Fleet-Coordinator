"""The Brain log: a bounded record of the learned system's recent decisions, for the UI.

Kept on the model (``ElevatorModel.brain``) for learned strategies only. Every entry is plain
JSON (no inf/NaN): the call, each candidate's bid / refusal / network score / attention, the
network's choice, the shadow teacher's choice (if enabled), the search statistics (if the
call was searched), and latencies. The encoded inputs of the last ``ENC_KEEP`` decisions are
kept for the occlusion "Why?" explanation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

KEEP = 200
ENC_KEEP = 50

#: Feature groups for occlusion explanations (car-token column ranges and call/global parts).
GROUPS: dict[str, tuple[str, list[int]]] = {
    "position": ("cars", [0, 18, 20, 21]),
    "direction": ("cars", [1, 2, 3, 22, 23]),
    "load": ("cars", [4, 5, 16]),
    "doors": ("cars", [9, 10, 11, 12, 13]),
    "plan": ("cars", [14, 15, 17, 19, 24, 25]),
    "availability": ("cars", [6, 7, 8]),
    "call": ("call", list(range(0, 7))),
    "traffic_pattern": ("call", list(range(7, 12))),
    "fleet_context": ("glob", list(range(0, 10))),
}


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


class BrainLog:
    """Ring buffer of decisions plus running statistics."""

    def __init__(self) -> None:
        self.decisions: deque[dict[str, Any]] = deque(maxlen=KEEP)
        self.encoded: dict[Any, dict[str, np.ndarray]] = {}
        self._enc_order: deque[Any] = deque()
        self.stats = {
            "decisions": 0,
            "agree_net": 0,
            "teacher_compared": 0,
            "agree_teacher": 0,
            "searched": 0,
            "overridden": 0,
            "search_ms": [],
        }

    def record(self, entry: dict[str, Any], encoded: dict[str, np.ndarray] | None = None) -> None:
        st = self.stats
        st["decisions"] += 1
        st["agree_net"] += int(entry["chosen"] == entry["net_choice"])
        if entry.get("teacher_choice") is not None:
            st["teacher_compared"] += 1
            st["agree_teacher"] += int(entry["chosen"] == entry["teacher_choice"])
        if entry.get("search"):
            st["searched"] += 1
            st["overridden"] += int(entry["search"]["overridden"])
            st["search_ms"].append(entry["search"]["time_ms"])
            st["search_ms"] = st["search_ms"][-500:]
        self.decisions.append(entry)
        if encoded is not None:
            cid = entry["conversation_id"]
            self.encoded[cid] = encoded
            self._enc_order.append(cid)
            while len(self._enc_order) > ENC_KEEP:
                self.encoded.pop(self._enc_order.popleft(), None)

    def summary(self) -> dict[str, Any]:
        st = self.stats
        n = max(st["decisions"], 1)
        ms = np.array(st["search_ms"]) if st["search_ms"] else np.zeros(0)
        return {
            "decisions": st["decisions"],
            "agree_net_pct": 100.0 * st["agree_net"] / n,
            "agree_teacher_pct": (
                100.0 * st["agree_teacher"] / st["teacher_compared"]
                if st["teacher_compared"]
                else None
            ),
            "searched_pct": 100.0 * st["searched"] / n,
            "overridden_pct": (100.0 * st["overridden"] / st["searched"])
            if st["searched"]
            else 0.0,
            "search_ms_mean": float(ms.mean()) if len(ms) else None,
            "search_ms_p95": float(np.percentile(ms, 95)) if len(ms) else None,
        }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self.decisions)[-limit:][::-1]


def candidate_rows(
    bids: list[Any], net_scores: dict[int, float], attn: dict[int, float], shadow: dict[int, float]
) -> list[dict[str, Any]]:
    """Per-car rows of a decision (strict-JSON values)."""
    rows = []
    for b in bids:
        rows.append(
            {
                "car_id": b.car_id,
                "refused": bool(b.refused),
                "reason": b.reason or None,
                "bid": _finite(b.total),
                "wait": _finite(b.wait),
                "ride": _finite(b.ride),
                "crowding": _finite(b.crowding),
                "energy": _finite(b.energy),
                "net_score": _finite(net_scores.get(b.car_id)),
                "attention": _finite(attn.get(b.car_id)),
                "teacher_bid": _finite(shadow.get(b.car_id)),
            }
        )
    return rows


def explain(runtime: Any, encoded: dict[str, np.ndarray], chosen: int) -> list[dict[str, Any]]:
    """Occlusion saliency: zero one feature group at a time and re-score.

    Reports, per group, how much the chosen car's and the runner-up's scores move and
    whether the network's choice would flip — "what did the decision depend on?".
    """
    base = runtime.score(encoded, trim=False).score[0]
    valid = encoded["mask"] & encoded["eligible"]
    order = [int(i) for i in np.argsort(np.where(valid, base, np.inf)) if valid[i]]
    runner = next((i for i in order if i != chosen), None)
    out = []
    for name, (token, cols) in GROUPS.items():
        occ = {k: v.copy() for k, v in encoded.items()}
        if token == "cars":
            sel = occ["mask"]
            for c in cols:
                occ["cars"][sel, c] = 0.0
        else:
            occ[token][cols] = 0.0
        s = runtime.score(occ, trim=False).score[0]
        new_choice = int(np.argmin(np.where(valid, s, np.inf)))
        out.append(
            {
                "group": name,
                "delta_score_chosen": float(s[chosen] - base[chosen]),
                "delta_score_runner_up": (
                    float(s[runner] - base[runner]) if runner is not None else None
                ),
                "flips_decision": bool(new_choice != chosen),
            }
        )
    out.sort(key=lambda g: -abs(g["delta_score_chosen"]))
    return out
