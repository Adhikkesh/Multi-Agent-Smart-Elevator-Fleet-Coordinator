"""Pluggable dispatch strategies, behind a registry."""

from elevator_mas.strategies.registry import (
    STRATEGIES,
    DispatchStrategy,
    get_strategy,
    register_strategy,
    strategy_names,
)

__all__ = [
    "STRATEGIES",
    "DispatchStrategy",
    "get_strategy",
    "register_strategy",
    "strategy_names",
]
