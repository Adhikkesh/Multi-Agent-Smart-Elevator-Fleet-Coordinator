"""Origin-destination profiles for the five traffic patterns.

These are the standard lift-traffic regimes from elevator engineering, and they matter
for the case study because each one rewards a different dispatch policy: up-peak is won
by parking cars at the lobby, down-peak by spreading them through the building.
"""

from __future__ import annotations

from dataclasses import dataclass

from elevator_mas.config import TrafficPattern


@dataclass(frozen=True)
class OriginDestinationProfile:
    """How likely an arrival is to start at, or be bound for, the lobby.

    `lobby_origin` is P(trip starts in the lobby) and `lobby_destination` is
    P(trip ends in the lobby | it did not start there). The remainder is inter-floor.
    """

    lobby_origin: float
    lobby_destination: float
    label: str

    def __post_init__(self) -> None:
        for value in (self.lobby_origin, self.lobby_destination):
            if not 0.0 <= value <= 1.0:
                raise ValueError("profile probabilities must lie in [0, 1]")


PATTERN_PROFILES: dict[TrafficPattern, OriginDestinationProfile] = {
    # Morning: nearly everyone enters at the lobby and rides up.
    "up_peak": OriginDestinationProfile(0.90, 0.05, "Up peak (morning arrival)"),
    # Evening: nearly everyone rides down to leave.
    "down_peak": OriginDestinationProfile(0.05, 0.90, "Down peak (evening departure)"),
    # Lunch: heavy in both directions at once, the hardest case for a dispatcher.
    "two_way": OriginDestinationProfile(0.45, 0.45, "Two-way (lunch)"),
    # Mid-morning: floor-to-floor trips that mostly skip the lobby.
    "interfloor": OriginDestinationProfile(0.20, 0.20, "Inter-floor"),
    "light": OriginDestinationProfile(0.30, 0.30, "Light / off-peak"),
}


def profile_for(pattern: TrafficPattern) -> OriginDestinationProfile:
    """The origin-destination profile for a named pattern."""
    return PATTERN_PROFILES[pattern]
