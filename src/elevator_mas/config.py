"""Pydantic configuration models, loaded from the scenario YAML files.

Nothing in the engine reads a magic number directly: floors, cars, timings, traffic
profiles, cost weights and scenario events all arrive through here, so a scenario can
be changed without touching code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

SCENARIO_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "scenarios"

TrafficPattern = Literal["up_peak", "down_peak", "two_way", "interfloor", "light"]


class BuildingConfig(BaseModel):
    """Physical building: how many floors, how many cars, how big the cars are."""

    floors: int = Field(default=15, ge=2, le=200)
    cars: int = Field(default=4, ge=1, le=32)
    capacity: int = Field(default=10, ge=1, le=100)
    lobby: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _lobby_in_building(self) -> BuildingConfig:
        if self.lobby >= self.floors:
            raise ValueError("lobby floor must be inside the building")
        return self


class TimingConfig(BaseModel):
    """Simulated seconds for each physical action. 1 tick == 1 simulated second."""

    seconds_per_floor: int = Field(default=2, ge=1)
    door_open: int = Field(default=2, ge=1)
    door_close: int = Field(default=2, ge=1)
    boarding_per_passenger: int = Field(default=1, ge=1)
    dwell_min: int = Field(default=0, ge=0)


class CostWeights(BaseModel):
    """W1-W4: the utility weights a car's bid is composed of.

    The TrafficMonitorAgent swaps whole weight sets when the traffic pattern changes,
    which is how a "learning agent" visibly changes fleet behaviour.
    """

    wait: float = Field(default=1.0, ge=0)
    ride: float = Field(default=0.5, ge=0)
    crowding: float = Field(default=0.3, ge=0)
    energy: float = Field(default=0.2, ge=0)


class PlannerConfig(BaseModel):
    """Search and local-search knobs."""

    algorithm: Literal["astar", "ucs", "greedy", "bfs"] = "astar"
    max_stops_for_search: int = Field(default=10, ge=1)
    energy_per_floor: float = Field(default=0.5, ge=0)
    energy_per_stop: float = Field(default=1.0, ge=0)
    # 30 s, not the 15 s a first guess suggests: reassigning more often than a car can
    # complete a leg just churns hall lanterns. Measured in docs/TESTING.md.
    reassign_interval: int = Field(default=30, ge=1)
    sa_iterations: int = Field(default=400, ge=1)
    sa_initial_temp: float = Field(default=12.0, gt=0)
    sa_cooling: float = Field(default=0.985, gt=0, lt=1)
    # A reassignment must beat the incumbent by this much to be adopted. Tuned by the
    # sweep recorded in docs/TESTING.md; too low and the fleet oscillates.
    hysteresis: float = Field(default=80.0, ge=0)
    parking_policy: Literal["lobby", "hill_climb", "minimax"] = "hill_climb"
    minimax_depth: int = Field(default=2, ge=1, le=4)
    minimax_candidate_floors: int = Field(default=6, ge=2)


class FairnessConfig(BaseModel):
    """Aging thresholds that stop a call being starved."""

    escalate_after: int = Field(default=60, ge=1)
    long_wait_threshold: int = Field(default=60, ge=1)
    escalation_bonus: float = Field(default=25.0, ge=0)


class ArrivalPhase(BaseModel):
    """A time window with its own arrival rate and traffic pattern.

    `rate` is the whole-building Poisson intensity in passengers per second.
    """

    until: int = Field(ge=1)
    rate: float = Field(ge=0)
    pattern: TrafficPattern = "interfloor"


class TrafficConfig(BaseModel):
    """A time-varying Poisson arrival process."""

    phases: list[ArrivalPhase] = Field(default_factory=lambda: [ArrivalPhase(until=3600, rate=0.2)])
    priority_probability: float = Field(default=0.0, ge=0, le=1)
    priority_weight: float = Field(default=3.0, ge=1)

    def pattern_at(self, tick: int) -> TrafficPattern:
        """The ground-truth traffic pattern at `tick` (the monitor must infer this)."""
        for phase in self.phases:
            if tick < phase.until:
                return phase.pattern
        return self.phases[-1].pattern

    def rate_at(self, tick: int) -> float:
        """The ground-truth arrival intensity at `tick`."""
        for phase in self.phases:
            if tick < phase.until:
                return phase.rate
        return 0.0


class EventConfig(BaseModel):
    """A scripted disturbance, so faults and emergencies are reproducible."""

    tick: int = Field(ge=0)
    kind: Literal["car_fault", "car_repair", "fire_alarm", "fire_clear", "rush"]
    car: int | None = None
    floor: int | None = None
    count: int = Field(default=10, ge=1)


class ScenarioConfig(BaseModel):
    """One complete, reproducible experiment."""

    name: str
    description: str = ""
    duration: int = Field(default=900, ge=1)
    seed: int = 42
    strategy: str = "full"
    building: BuildingConfig = Field(default_factory=BuildingConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    weights: CostWeights = Field(default_factory=CostWeights)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)
    events: list[EventConfig] = Field(default_factory=list)
    expected: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioConfig:
        """Load a scenario from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def load(cls, name: str) -> ScenarioConfig:
        """Load a bundled scenario by name (without the .yaml suffix)."""
        path = SCENARIO_DIR / f"{name}.yaml"
        if not path.exists():
            available = ", ".join(sorted(available_scenarios()))
            raise FileNotFoundError(f"unknown scenario {name!r}; available: {available}")
        return cls.from_yaml(path)


def available_scenarios() -> list[str]:
    """Names of every bundled scenario YAML."""
    if not SCENARIO_DIR.exists():
        return []
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
