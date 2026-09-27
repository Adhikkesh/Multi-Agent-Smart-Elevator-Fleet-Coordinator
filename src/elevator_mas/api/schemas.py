"""Pydantic request bodies for the control endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    """Rebuild the simulation, optionally with new settings."""

    scenario: str | None = None
    strategy: str | None = None
    seed: int | None = None
    floors: int | None = Field(default=None, ge=2, le=200)
    cars: int | None = Field(default=None, ge=1, le=32)


class SpeedRequest(BaseModel):
    """Set the playback multiplier."""

    speed: float = Field(ge=0.1, le=50.0)


class StepRequest(BaseModel):
    """Advance a fixed number of ticks while paused."""

    ticks: int = Field(default=1, ge=1, le=600)


class PassengerRequest(BaseModel):
    """Add one passenger by hand (clicking a floor in the dashboard)."""

    origin: int = Field(ge=0)
    destination: int | None = None
    priority: bool = False


class InjectRequest(BaseModel):
    """Inject a disturbance."""

    kind: str
    car: int | None = None
    floor: int | None = None
    count: int = Field(default=10, ge=1, le=100)


class BenchmarkRequest(BaseModel):
    """Run a benchmark from the dashboard."""

    scenarios: list[str] | None = None
    strategies: list[str] | None = None
    seeds: int = Field(default=3, ge=1, le=10)
    ticks: int = Field(default=600, ge=60, le=3600)
