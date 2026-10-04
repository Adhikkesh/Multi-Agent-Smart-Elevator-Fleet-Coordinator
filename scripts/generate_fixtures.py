"""Generate 20 deterministic routing problem fixtures for client-side A* parity testing."""

import json
from pathlib import Path

from elevator_mas.api.session import SimulationSession


def generate():
    fixtures = []
    # Use different scenarios and ticks to get diverse routing problems
    scenarios = ["demo_story", "morning_up_peak", "evening_down_peak", "lunch_two_way"]

    count = 0
    for scen in scenarios:
        session = SimulationSession(scen)
        # Advance simulation to create realistic pending stops
        for _step in [20, 40, 60, 80, 100]:
            session.model.run(20)
            for car in session.model.cars:
                if len(car.pending_stops()) >= 2:
                    data = session.snapshot_routing_problem(car.car_id)
                    astar_res = next(
                        (r for r in data["results"] if r["algorithm"] == "astar"), None
                    )
                    if astar_res and astar_res["found"]:
                        fixtures.append(
                            {
                                "id": count,
                                "scenario": scen,
                                "car_id": data["car_id"],
                                "current_floor": data["current_floor"],
                                "stops": data["stops"],
                                "h_at_start": data["h_at_start"],
                                "astar": astar_res,
                            }
                        )
                        count += 1
                        if count >= 20:
                            break
            if count >= 20:
                break
        if count >= 20:
            break

    # If we need more to reach 20, use _random_stops
    while count < 20:
        session = SimulationSession("demo_story")
        data = session.snapshot_routing_problem(None)
        astar_res = next((r for r in data["results"] if r["algorithm"] == "astar"), None)
        if astar_res and astar_res["found"]:
            fixtures.append(
                {
                    "id": count,
                    "scenario": "random",
                    "car_id": data["car_id"],
                    "current_floor": data["current_floor"],
                    "stops": data["stops"],
                    "h_at_start": data["h_at_start"],
                    "astar": astar_res,
                }
            )
            count += 1

    out_file = Path("frontend/src/lab/fixtures/routing_fixtures.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(fixtures, indent=2))
    print(f"Generated {len(fixtures)} fixtures in {out_file}")


if __name__ == "__main__":
    generate()
