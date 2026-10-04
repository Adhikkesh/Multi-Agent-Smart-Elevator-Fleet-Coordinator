"""Loader and utilities for LiftZero learning regimes and evaluation configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REGIMES_FILE = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "learning" / "regimes.yaml"
)


@dataclass(frozen=True)
class EvalRegime:
    name: str
    floors: int
    cars: int
    capacity: int
    duration: int
    rate: float
    pattern: str
    note: str = ""


class RegimesConfig:
    """Loaded configuration from regimes.yaml."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.training = data.get("training", {})
        self.eval_regimes: dict[str, EvalRegime] = {}
        for name, r in data.get("eval_regimes", {}).items():
            self.eval_regimes[name] = EvalRegime(
                name=name,
                floors=r["floors"],
                cars=r["cars"],
                capacity=r["capacity"],
                duration=r["duration"],
                rate=float(r["rate"]),
                pattern=r["pattern"],
                note=r.get("note", ""),
            )

        self.scale: dict[str, EvalRegime] = {}
        for name, r in data.get("scale", {}).items():
            self.scale[name] = EvalRegime(
                name=name,
                floors=r["floors"],
                cars=r["cars"],
                capacity=r["capacity"],
                duration=r["duration"],
                rate=float(r["rate"]),
                pattern=r["pattern"],
                note=r.get("note", ""),
            )

        self.seeds = data.get("seeds", {})

    @property
    def val_seeds(self) -> list[int]:
        s = int(self.seeds.get("val_start", 8000))
        e = int(self.seeds.get("val_end", 8049))
        return list(range(s, e + 1))

    @property
    def test_seeds(self) -> list[int]:
        s = int(self.seeds.get("test_start", 8500))
        e = int(self.seeds.get("test_end", 8599))
        return list(range(s, e + 1))


def load_regimes_config(path: Path | str | None = None) -> RegimesConfig:
    """Load regimes configuration from yaml file."""
    p = Path(path) if path is not None else REGIMES_FILE
    if not p.exists():
        raise FileNotFoundError(f"Regimes config not found at {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return RegimesConfig(data)
