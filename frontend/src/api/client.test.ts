import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./client";
import {
  AnnealingLabSchema,
  BenchmarkSchema,
  BoardResponseSchema,
  MetaSchema,
  MinimaxLabSchema,
  RunResponseSchema,
  SearchLabRoutingSchema,
  validateSnapshotShape,
} from "./schemas";

describe("API client & schemas", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("creates ApiError with status and detail", () => {
    const err = new ApiError(404, "Car not found");
    expect(err.name).toBe("ApiError");
    expect(err.status).toBe(404);
    expect(err.detail).toBe("Car not found");
    expect(err.message).toContain("404");
  });

  it("validates snapshot shape cheaply and accurately", () => {
    expect(validateSnapshotShape(null)).toBe(false);
    expect(validateSnapshotShape("string")).toBe(false);
    expect(validateSnapshotShape({})).toBe(false);

    const valid = {
      tick: 1,
      cars: [],
      floors: [],
      metrics: {},
      building: {},
    };
    expect(validateSnapshotShape(valid)).toBe(true);

    const invalid = {
      tick: "not-a-number",
      cars: [],
      floors: [],
    };
    expect(validateSnapshotShape(invalid)).toBe(false);
  });

  it("validates MetaSchema", () => {
    const meta = {
      scenarios: ["demo_story"],
      strategies: [{ name: "full", description: "all features" }],
      agents: [
        {
          name: "ElevatorAgent",
          agent_type: "utility-based",
          peas: {
            performance: "wait time",
            environment: "building",
            actuators: "motor",
            sensors: "encoder",
          },
          docstring: "car agent",
        },
      ],
      environment: [{ property: "Agents", value: "Multi-agent", justification: "cars" }],
      system_peas: {
        performance: "wait",
        environment: "building",
        actuators: "motors",
        sensors: "buttons",
      },
      rules: [{ name: "R1", salience: 100, description: "fire" }],
    };
    expect(() => MetaSchema.parse(meta)).not.toThrow();
  });

  it("validates SearchLabRoutingSchema", () => {
    const routing = {
      source: "live",
      car_id: 0,
      current_floor: 2,
      stops: [{ floor: 4, kind: "dropoff", direction: "IDLE", weight: 1 }],
      h_at_start: 12.0,
      results: [
        {
          algorithm: "astar",
          found: true,
          cost: 15.0,
          nodes_expanded: 2,
          nodes_generated: 4,
          max_frontier: 2,
          runtime_ms: 0.1,
          depth: 1,
          route: ["4D"],
        },
      ],
    };
    expect(() => SearchLabRoutingSchema.parse(routing)).not.toThrow();
  });

  it("validates AnnealingLabSchema", () => {
    const annealing = {
      available: true,
      synthetic: false,
      simulated_annealing: {
        initial_cost: 100,
        final_cost: 70,
        improvement: 30,
        iterations: 400,
        accepted: 25,
        curve: [100, 90, 70],
        assignment: { "1U": 0 },
      },
    };
    expect(() => AnnealingLabSchema.parse(annealing)).not.toThrow();
  });

  it("validates MinimaxLabSchema", () => {
    const minimax = {
      idle_cars: 2,
      cars_placed: 2,
      candidate_spots: [0, 7, 14],
      likely_floors: [0, 14],
      best_parking: { "0": 0, "1": 7 },
      value: 12.5,
      minimax_nodes: 50,
      alphabeta_nodes: 25,
      pruning_saving_pct: 50.0,
    };
    expect(() => MinimaxLabSchema.parse(minimax)).not.toThrow();
  });

  it("validates BenchmarkSchema", () => {
    const bench = {
      runs: 1,
      scenarios: ["morning_up_peak"],
      strategies: ["full"],
      seeds: [1],
      ticks: 600,
      summary: [
        {
          scenario: "morning_up_peak",
          strategy: "full",
          runs: 1,
          avg_wait_mean: 15.0,
          avg_wait_std: 0.0,
          p95_wait_mean: 25.0,
          p95_wait_std: 0.0,
          long_wait_pct_mean: 0.0,
          long_wait_pct_std: 0.0,
          energy_mean: 200.0,
          energy_std: 0.0,
          throughput_mean: 300.0,
          throughput_std: 0.0,
          violations: 0,
        },
      ],
      comparison: [
        {
          scenario: "morning_up_peak",
          strategy: "full",
          avg_wait_delta_pct: -25.0,
          p95_wait_delta_pct: -20.0,
          long_wait_pct_delta_pct: 0.0,
          energy_delta_pct: 5.0,
          wins_on: ["avg_wait"],
        },
      ],
      wins: { full: 1 },
    };
    expect(() => BenchmarkSchema.parse(bench)).not.toThrow();
  });

  it("validates RunResponseSchema", () => {
    const runRes = {
      scenario: "interfloor_light",
      strategy: "collective",
      seed: 42,
      ticks: 60,
      runtime_s: 0.05,
      final: { avg_wait: 10.0 },
      series: {
        tick: [10, 20],
        avg_wait: [5, 10],
        p95_wait: [10, 15],
        waiting: [2, 1],
        riding: [1, 2],
        delivered: [1, 3],
        energy: [10, 20],
        long_wait_pct: [0, 0],
        messages: [10, 25],
      },
      rules_fired: [],
      violations: [],
    };
    expect(() => RunResponseSchema.parse(runRes)).not.toThrow();
  });

  it("validates BoardResponseSchema", () => {
    const board = {
      cars: [
        {
          car_id: 0,
          tick: 1,
          floor: 0,
          direction: "UP",
          load: 2,
          capacity: 10,
          space: 8,
          available: true,
          out_of_service: false,
          fire_mode: false,
          door: "closed",
          assigned_calls: ["3U"],
          car_calls: [7],
          riders: 2,
          plan_end_floor: 7,
          plan_end_eta: 14.5,
          planned_stops: 2,
          park_target: null,
        },
      ],
      policy: {
        pattern: "two_way",
        weights: { wait: 1.0, ride: 0.5, crowding: 0.3, energy: 0.2 },
        published_by: "TrafficMonitor",
        tick: 1,
      },
      writes: 5,
    };
    expect(() => BoardResponseSchema.parse(board)).not.toThrow();
  });

  it("invokes play, pause, step, speed, reset via fetch", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ running: true, speed: 5, tick: 10 }),
    });
    vi.stubGlobal("fetch", mockFetch);

    await api.play();
    expect(mockFetch).toHaveBeenCalledWith("/api/play", expect.objectContaining({ method: "POST" }));

    await api.pause();
    expect(mockFetch).toHaveBeenCalledWith("/api/pause", expect.objectContaining({ method: "POST" }));

    await api.step(10);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/step",
      expect.objectContaining({ body: JSON.stringify({ ticks: 10 }) }),
    );

    await api.setSpeed(5);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/speed",
      expect.objectContaining({ body: JSON.stringify({ speed: 5 }) }),
    );

    await api.reset({ scenario: "morning_up_peak" });
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/reset",
      expect.objectContaining({ body: JSON.stringify({ scenario: "morning_up_peak" }) }),
    );
  });

  it("throws ApiError on non-2xx response with detail message", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      statusText: "Bad Request",
      json: async () => ({ detail: "Unknown scenario" }),
    });
    vi.stubGlobal("fetch", mockFetch);

    await expect(api.reset({ scenario: "invalid" })).rejects.toThrow(ApiError);
  });
});
