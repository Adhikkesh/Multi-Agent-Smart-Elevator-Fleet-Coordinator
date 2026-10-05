"""Compare BFS, UCS, Greedy and A* on random car-routing problems (the search rubric item).

Each instance is a car on a 15-floor building with 7 pending stops (pickups with a direction
and a waiting weight, or drop-offs), solved by all four algorithms on the *same* problem.
Reports, per algorithm: how often it found the optimal plan, its mean cost gap to optimal,
nodes expanded and runtime. Writes reports/search_benchmark.csv (+ printed table).

    uv run python scripts/search_benchmark.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from elevator_mas.domain import Direction, Stop, StopKind
from elevator_mas.planning import ALGORITHMS, CarRoutingProblem
from elevator_mas.planning.routing import RoutingCosts

ROOT = Path(__file__).resolve().parent.parent
FLOORS, STOPS, INSTANCES = 15, 7, 200


def random_problem(rng: random.Random) -> CarRoutingProblem:
    stops = []
    floors = rng.sample(range(FLOORS), STOPS + 1)
    start, targets = floors[0], floors[1:]
    for f in targets:
        if rng.random() < 0.5:
            stops.append(
                Stop(
                    f,
                    StopKind.PICKUP,
                    rng.choice([Direction.UP, Direction.DOWN]),
                    float(rng.randint(1, 3)),
                )
            )
        else:
            stops.append(Stop(f, StopKind.DROPOFF, Direction.IDLE, float(rng.randint(1, 3))))
    costs = RoutingCosts(
        seconds_per_floor=2.0, dwell=4.0, energy_per_floor=1.0, energy_per_stop=2.0
    )
    return CarRoutingProblem(start, stops, costs, boarding_per_passenger=1.0)


def main() -> None:
    rng = random.Random(7)
    rows = []
    i = skipped = 0
    while i < INSTANCES:
        problem = random_problem(rng)
        res = {name: ALGORITHMS[name](problem) for name in ("bfs", "ucs", "greedy", "astar")}
        if not any(r.found for r in res.values()):
            skipped += 1  # no plan obeys collective control for this random stop set
            continue
        best = min(r.cost for r in res.values() if r.found)
        for name, r in res.items():
            rows.append(
                {
                    "instance": i,
                    "algorithm": name,
                    "found": r.found,
                    "cost": r.cost,
                    "optimal": bool(r.found and abs(r.cost - best) < 1e-6),
                    "gap_pct": (r.cost / best - 1) * 100 if best else 0.0,
                    "nodes_expanded": r.nodes_expanded,
                    "runtime_ms": r.runtime_ms,
                }
            )
        i += 1
    df = pd.DataFrame(rows)
    out = ROOT / "reports" / "search_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summ = df.groupby("algorithm").agg(
        optimal_pct=("optimal", lambda x: 100 * x.mean()),
        mean_gap_pct=("gap_pct", "mean"),
        nodes_mean=("nodes_expanded", "mean"),
        runtime_ms_mean=("runtime_ms", "mean"),
    )
    summ = summ.loc[["bfs", "ucs", "greedy", "astar"]].round(2)
    print(f"{INSTANCES} solvable random instances ({skipped} unsolvable skipped), {STOPS} stops")
    print(summ.to_string())
    summ.to_csv(ROOT / "reports" / "search_benchmark_summary.csv")


if __name__ == "__main__":
    main()
