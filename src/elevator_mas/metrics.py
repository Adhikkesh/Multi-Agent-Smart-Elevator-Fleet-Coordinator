"""Metric definitions and aggregation.

These are the performance measures from the PEAS formulation, so what the dashboard
shows, what the benchmark tabulates and what `P` in the PEAS table promises are all the
same numbers computed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from statistics import mean

from elevator_mas.domain import PassengerRecord


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (`q` in [0, 1]); 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


@dataclass
class Metrics:
    """A snapshot of every performance measure, for one moment or one whole run."""

    tick: int = 0
    arrived: int = 0
    delivered: int = 0
    waiting: int = 0
    riding: int = 0
    avg_wait: float = 0.0
    p95_wait: float = 0.0
    max_wait: float = 0.0
    avg_ride: float = 0.0
    avg_system: float = 0.0
    long_wait_pct: float = 0.0
    throughput: float = 0.0
    energy: float = 0.0
    floors_travelled: int = 0
    stops: int = 0
    reversals: int = 0
    messages: int = 0
    messages_per_call: float = 0.0
    calls: int = 0
    nodes_expanded: int = 0
    replans: int = 0
    compute_ms_per_tick: float = 0.0
    rules_fired: int = 0

    def as_dict(self) -> dict[str, float | int]:
        """JSON-friendly form, floats rounded for display."""
        out: dict[str, float | int] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = round(value, 3) if isinstance(value, float) else value
        return out


@dataclass
class EnergyCounters:
    """Physical-effort counters the cars increment as they move."""

    floors_travelled: int = 0
    stops: int = 0
    reversals: int = 0
    per_floor: float = 1.0
    per_stop: float = 2.0
    per_reversal: float = 1.0

    @property
    def total(self) -> float:
        """Weighted energy total, the `energy` term of the performance measure."""
        return (
            self.floors_travelled * self.per_floor
            + self.stops * self.per_stop
            + self.reversals * self.per_reversal
        )


@dataclass
class MetricsCollector:
    """Turns passenger records and counters into a `Metrics` snapshot."""

    long_wait_threshold: int = 60
    energy: EnergyCounters = field(default_factory=EnergyCounters)
    records: dict[int, PassengerRecord] = field(default_factory=dict)
    total_messages: int = 0
    total_calls: int = 0
    total_nodes_expanded: int = 0
    total_replans: int = 0
    total_rules_fired: int = 0
    compute_ms: float = 0.0
    ticks: int = 0

    def register(self, record: PassengerRecord) -> None:
        """Start tracking a passenger."""
        self.records[record.passenger_id] = record

    def snapshot(self, tick: int) -> Metrics:
        """Compute every metric from the records collected so far.

        Waits are measured for boarded passengers; a passenger still waiting has their
        wait counted *so far*, so the live KPIs cannot hide a starving call behind a
        flattering average of only the lucky ones.
        """
        records = list(self.records.values())
        boarded = [r for r in records if r.board_tick is not None]
        delivered = [r for r in records if r.delivered]
        waits = [float(r.wait_time) for r in boarded if r.wait_time is not None]
        pending_waits = [float(tick - r.arrival_tick) for r in records if r.board_tick is None]
        all_waits = waits + pending_waits
        rides = [float(r.ride_time) for r in delivered if r.ride_time is not None]
        systems = [float(r.system_time) for r in delivered if r.system_time is not None]
        long_waits = [w for w in all_waits if w > self.long_wait_threshold]

        return Metrics(
            tick=tick,
            arrived=len(records),
            delivered=len(delivered),
            waiting=sum(1 for r in records if r.board_tick is None),
            riding=sum(1 for r in records if r.board_tick is not None and not r.delivered),
            avg_wait=mean(all_waits) if all_waits else 0.0,
            p95_wait=percentile(all_waits, 0.95),
            max_wait=max(all_waits) if all_waits else 0.0,
            avg_ride=mean(rides) if rides else 0.0,
            avg_system=mean(systems) if systems else 0.0,
            long_wait_pct=100.0 * len(long_waits) / len(all_waits) if all_waits else 0.0,
            throughput=3600.0 * len(delivered) / tick if tick > 0 else 0.0,
            energy=self.energy.total,
            floors_travelled=self.energy.floors_travelled,
            stops=self.energy.stops,
            reversals=self.energy.reversals,
            messages=self.total_messages,
            messages_per_call=(self.total_messages / self.total_calls if self.total_calls else 0.0),
            calls=self.total_calls,
            nodes_expanded=self.total_nodes_expanded,
            replans=self.total_replans,
            compute_ms_per_tick=self.compute_ms / self.ticks if self.ticks else 0.0,
            rules_fired=self.total_rules_fired,
        )
