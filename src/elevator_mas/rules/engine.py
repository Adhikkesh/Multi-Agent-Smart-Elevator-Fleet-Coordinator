"""A small forward-chaining production system (AIMA 4e §9.3).

The SafetyAgent is a knowledge-based agent, so its behaviour is not coded as `if`
statements inside the agent — it is *data*: a rule base of `IF conditions THEN actions`
productions, plus this engine that runs them to a fixed point.

Forward chaining here is the data-driven variety: assert what is observed as facts, then
repeatedly fire every rule whose premises are satisfied, adding the facts its conclusion
asserts, until a whole pass adds nothing new. Rules are tried in salience order (highest
first) so that fire recall dominates overload, which dominates routine faults.

Every firing is recorded in `fired`, which is what the dashboard's rules-fired log shows
and what makes the agent's reasoning auditable rather than a black box.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

Fact = tuple[Any, ...]


@dataclass(frozen=True)
class Rule:
    """One production: `IF condition(facts) THEN actions`.

    `condition` inspects working memory and returns the bindings it matched (each a dict
    of variables), or an empty list if it does not apply. `conclude` turns one binding
    into the new facts to assert; `act` performs the external side effect on the world.
    """

    name: str
    salience: int
    condition: Callable[[set[Fact], Any], list[dict[str, Any]]]
    conclude: Callable[[dict[str, Any], Any], Iterable[Fact]] | None = None
    act: Callable[[dict[str, Any], Any], str] | None = None
    description: str = ""


@dataclass
class FiredRule:
    """A record of one rule firing, for the UI log and for the tests."""

    tick: int
    rule: str
    binding: dict[str, Any]
    effect: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the dashboard."""
        return {
            "tick": self.tick,
            "rule": self.rule,
            "binding": {k: _plain(v) for k, v in self.binding.items()},
            "effect": self.effect,
        }


def _plain(value: Any) -> Any:
    """Coerce a binding value into something JSON can carry."""
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


class ForwardChainingEngine:
    """Runs a rule base over working memory until no new fact can be derived."""

    def __init__(self, rules: Iterable[Rule], max_passes: int = 16) -> None:
        # Highest salience first; the name breaks ties so ordering is deterministic.
        self.rules: list[Rule] = sorted(rules, key=lambda r: (-r.salience, r.name))
        self.max_passes = max_passes
        self.working_memory: set[Fact] = set()
        self.fired: list[FiredRule] = []
        self.total_fired: int = 0

    def assert_fact(self, fact: Fact) -> None:
        """Add one fact to working memory."""
        self.working_memory.add(fact)

    def assert_facts(self, facts: Iterable[Fact]) -> None:
        """Add many facts to working memory."""
        self.working_memory.update(facts)

    def retract(self, fact: Fact) -> None:
        """Remove a fact, so a cleared alarm really does change the conclusions."""
        self.working_memory.discard(fact)

    def holds(self, fact: Fact) -> bool:
        """Whether a fact is currently believed."""
        return fact in self.working_memory

    def run(self, world: Any, tick: int, log: list[FiredRule] | None = None) -> list[FiredRule]:
        """Forward-chain to a fixed point and return the firings from this call.

        A rule fires at most once per binding per call (tracked in `seen`), which is what
        stops a rule that re-asserts its own premise from looping forever.
        """
        fired_now: list[FiredRule] = []
        seen: set[tuple[str, str]] = set()

        for _ in range(self.max_passes):
            new_facts: set[Fact] = set()
            progressed = False

            for rule in self.rules:
                for binding in rule.condition(self.working_memory, world):
                    key = (rule.name, repr(sorted(binding.items(), key=lambda kv: kv[0])))
                    if key in seen:
                        continue
                    seen.add(key)
                    effect = ""
                    if rule.act is not None:
                        effect = rule.act(binding, world) or ""
                    if rule.conclude is not None:
                        for fact in rule.conclude(binding, world):
                            if fact not in self.working_memory:
                                new_facts.add(fact)
                    record = FiredRule(tick=tick, rule=rule.name, binding=binding, effect=effect)
                    fired_now.append(record)
                    self.fired.append(record)
                    self.total_fired += 1
                    progressed = True

            if new_facts:
                self.working_memory.update(new_facts)
                progressed = True
            if not progressed:
                break

        if log is not None:
            log.extend(fired_now)
        # Keep the in-engine history bounded for long headless runs.
        if len(self.fired) > 500:
            del self.fired[:-500]
        return fired_now

    def explain(self, limit: int = 20) -> list[dict[str, Any]]:
        """The most recent firings, for the dashboard's rules-fired panel."""
        return [f.as_dict() for f in self.fired[-limit:]]
