"""Headless single-scenario runs.

Completely independent of the API and the web layer: the engine can be driven from a
script, a test or the CLI without a server ever starting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from elevator_mas.config import ScenarioConfig
from elevator_mas.metrics import Metrics
from elevator_mas.model import ElevatorModel


@dataclass
class RunResult:
    """Everything one headless run produced."""

    scenario: str
    strategy: str
    seed: int
    ticks: int
    metrics: Metrics
    runtime_seconds: float
    violations: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    detected_pattern: str = ""
    delivered_all: bool = False

    def as_row(self) -> dict[str, Any]:
        """A flat row for a pandas DataFrame."""
        row: dict[str, Any] = {
            "scenario": self.scenario,
            "strategy": self.strategy,
            "seed": self.seed,
            "ticks": self.ticks,
            "runtime_s": round(self.runtime_seconds, 3),
            "violations": len(self.violations),
            "detected_pattern": self.detected_pattern,
        }
        row.update(self.metrics.as_dict())
        return row

    def summary(self) -> str:
        """A human-readable block, printed by `elevator run`."""
        m = self.metrics
        lines = [
            f"Scenario      {self.scenario}  (strategy={self.strategy}, seed={self.seed})",
            f"Simulated     {self.ticks} s in {self.runtime_seconds:.2f} s wall clock "
            f"({m.compute_ms_per_tick:.2f} ms/tick)",
            "",
            f"  passengers        {m.delivered} delivered / {m.arrived} arrived "
            f"({m.waiting} waiting, {m.riding} riding)",
            f"  average wait      {m.avg_wait:8.1f} s",
            f"  95th pct wait     {m.p95_wait:8.1f} s",
            f"  maximum wait      {m.max_wait:8.1f} s",
            f"  long waits        {m.long_wait_pct:8.1f} %  of waits exceed the threshold",
            f"  average ride      {m.avg_ride:8.1f} s",
            f"  average system    {m.avg_system:8.1f} s",
            f"  throughput        {m.throughput:8.0f} passengers/hour",
            "",
            f"  energy            {m.energy:8.0f}  "
            f"({m.floors_travelled} floors, {m.stops} stops, {m.reversals} reversals)",
            f"  messages          {m.messages:8d}  ({m.messages_per_call:.1f} per call, "
            f"{m.calls} calls)",
            f"  search nodes      {m.nodes_expanded:8d}  ({m.replans} replans)",
            f"  rules fired       {m.rules_fired:8d}",
            "",
            f"  traffic detected  {self.detected_pattern}",
            f"  safety rules      {', '.join(self.rules_fired) or 'none'}",
            f"  invariants        {'OK' if not self.violations else 'VIOLATED'}",
        ]
        if self.violations:
            lines.extend(f"    ! {v}" for v in self.violations[:10])
        return "\n".join(lines)


def run_scenario(
    scenario: str | ScenarioConfig,
    strategy: str | None = None,
    seed: int | None = None,
    ticks: int | None = None,
    csv_path: str | Path | None = None,
    drain: bool = False,
) -> RunResult:
    """Run one scenario headlessly and collect its metrics.

    With `drain=True` the run continues after the scenario's duration, with arrivals
    switched off, until everyone already in the building has been delivered. That is what
    makes the no-starvation assertion meaningful.
    """
    config = ScenarioConfig.load(scenario) if isinstance(scenario, str) else scenario
    if strategy is not None:
        config = config.model_copy(update={"strategy": strategy})
    if seed is not None:
        config = config.model_copy(update={"seed": seed})

    model = ElevatorModel(config)
    horizon = ticks if ticks is not None else config.duration

    started = time.perf_counter()
    violations: list[str] = []
    for _ in range(horizon):
        model.step()
        # Check the invariants every tick, not just at the end: a violation that is later
        # repaired would otherwise go unnoticed.
        violations.extend(model.violations())
    if drain:
        model.drain()
        violations.extend(model.violations())
    runtime = time.perf_counter() - started

    records = model.collector.records.values()
    result = RunResult(
        scenario=config.name,
        strategy=config.strategy,
        seed=config.seed,
        ticks=model.tick,
        metrics=model.latest_metrics,
        runtime_seconds=runtime,
        violations=sorted(set(violations)),
        rules_fired=sorted({r.rule for r in model.safety.log}),
        detected_pattern=model.monitor.pattern,
        delivered_all=all(r.delivered for r in records),
    )

    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.datacollector.get_model_vars_dataframe().to_csv(path, index=False)

    return result
