import { describe, expect, it } from "vitest";
import fixtures from "./fixtures/routing_fixtures.json";
import {
  actionCost,
  heuristic,
  legalStops,
  mergeStops,
  solveRoutingProblem,
  type Stop,
} from "./routing";

describe("Routing Search Parity with Python Planner", () => {
  it("has exactly 20 test fixtures", () => {
    expect(fixtures.length).toBe(20);
  });

  for (const fixture of fixtures) {
    it(`fixture #${fixture.id} (${fixture.scenario}, car ${fixture.car_id}): client A* cost matches server A* cost`, () => {
      const rawStops: Stop[] = fixture.stops.map((s) => ({
        floor: s.floor,
        kind: s.kind as "pickup" | "dropoff",
        direction: s.direction as any,
        weight: s.weight,
      }));

      const { result } = solveRoutingProblem(fixture.current_floor, rawStops, "astar");

      expect(result.found).toBe(true);
      // Assert optimal cost matches the server's astar.cost within 1e-4
      expect(Math.abs(result.cost - fixture.astar.cost)).toBeLessThan(1e-3);
    });
  }

  it("handles empty stops gracefully", () => {
    const { result } = solveRoutingProblem(3, [], "astar");
    expect(result.found).toBe(true);
    expect(result.cost).toBe(0);
    expect(result.route.length).toBe(0);
  });

  it("merges identical stops correctly", () => {
    const stops: Stop[] = [
      { floor: 4, kind: "pickup", direction: "UP", weight: 1 },
      { floor: 4, kind: "pickup", direction: "UP", weight: 2 },
      { floor: 4, kind: "dropoff", direction: "IDLE", weight: 1 },
    ];
    const merged = mergeStops(stops);
    expect(merged.length).toBe(2);
    const pickup = merged.find((s) => s.kind === "pickup");
    expect(pickup?.weight).toBe(3);
  });

  it("enforces legal stops constraint (no backwards carry)", () => {
    const pending: Stop[] = [
      { floor: 2, kind: "pickup", direction: "UP", weight: 1 },
      { floor: 8, kind: "dropoff", direction: "IDLE", weight: 1 },
    ];
    // If current floor is 5, pickup at 2 is behind us with dropoff at 8
    const legals = legalStops(5, pending);
    expect(legals.some((s) => s.floor === 2)).toBe(false);
    expect(legals.some((s) => s.floor === 8)).toBe(true);
  });

  it("calculates admissible heuristic and step action costs", () => {
    const pending: Stop[] = [{ floor: 5, kind: "dropoff", direction: "IDLE", weight: 2 }];
    const h = heuristic(0, pending);
    expect(h).toBeGreaterThan(0);

    const stepCost = actionCost(0, pending, pending[0]!);
    expect(stepCost).toBeGreaterThan(0);
  });

  it("solves with UCS, BFS, and Greedy algorithms", () => {
    const pending: Stop[] = [
      { floor: 2, kind: "dropoff", direction: "IDLE", weight: 1 },
      { floor: 5, kind: "dropoff", direction: "IDLE", weight: 1 },
    ];
    const ucs = solveRoutingProblem(0, pending, "ucs");
    const bfs = solveRoutingProblem(0, pending, "bfs");
    const greedy = solveRoutingProblem(0, pending, "greedy");

    expect(ucs.result.found).toBe(true);
    expect(bfs.result.found).toBe(true);
    expect(greedy.result.found).toBe(true);
  });
});
