import { z } from "zod";

export const DirectionSchema = z.enum(["UP", "DOWN", "IDLE"]);
export const DoorStateSchema = z.enum(["closed", "opening", "open", "closing"]);

export const MetaSchema = z.object({
  scenarios: z.array(z.string()),
  strategies: z.array(
    z.object({
      name: z.string(),
      description: z.string(),
      learning: z.boolean().optional(),
    }),
  ),
  agents: z.array(
    z.object({
      name: z.string(),
      agent_type: z.string(),
      peas: z.object({
        performance: z.string(),
        environment: z.string(),
        actuators: z.string(),
        sensors: z.string(),
      }),
      docstring: z.string(),
    }),
  ),
  environment: z.array(
    z.object({
      property: z.string(),
      value: z.string(),
      justification: z.string(),
    }),
  ),
  system_peas: z.object({
    performance: z.string(),
    environment: z.string(),
    actuators: z.string(),
    sensors: z.string(),
  }),
  rules: z.array(
    z.object({
      name: z.string(),
      salience: z.number(),
      description: z.string(),
    }),
  ),
});

export const SearchLabRoutingSchema = z.object({
  source: z.string(),
  car_id: z.number(),
  current_floor: z.number(),
  stops: z.array(
    z.object({
      floor: z.number(),
      kind: z.string(),
      direction: z.string(),
      weight: z.number(),
    }),
  ),
  h_at_start: z.number(),
  results: z.array(
    z.object({
      algorithm: z.string(),
      found: z.boolean(),
      cost: z.number(),
      nodes_expanded: z.number(),
      nodes_generated: z.number(),
      max_frontier: z.number(),
      runtime_ms: z.number(),
      depth: z.number(),
      route: z.array(z.string()),
      optimal: z.boolean().optional(),
    }),
  ),
});

export const AnnealingLabSchema = z.object({
  available: z.boolean(),
  synthetic: z.boolean().optional(),
  reason: z.string().optional(),
  calls: z.number().optional(),
  cars: z.number().optional(),
  simulated_annealing: z
    .object({
      initial_cost: z.number(),
      final_cost: z.number(),
      improvement: z.number(),
      iterations: z.number(),
      accepted: z.number(),
      curve: z.array(z.number()),
      best_curve: z.array(z.number()).optional(),
      assignment: z.record(z.string(), z.number()),
    })
    .optional(),
  hill_climbing: z
    .object({
      initial_cost: z.number(),
      final_cost: z.number(),
      improvement: z.number(),
      iterations: z.number(),
      accepted: z.number(),
      curve: z.array(z.number()),
      best_curve: z.array(z.number()).optional(),
      assignment: z.record(z.string(), z.number()),
    })
    .optional(),
});

export const MinimaxLabSchema = z.object({
  idle_cars: z.number(),
  cars_placed: z.number(),
  candidate_spots: z.array(z.number()),
  likely_floors: z.array(z.number()),
  best_parking: z.record(z.string(), z.number()),
  value: z.number(),
  minimax_nodes: z.number(),
  alphabeta_nodes: z.number(),
  pruning_saving_pct: z.number(),
});

export const BenchmarkSchema = z.object({
  runs: z.number(),
  scenarios: z.array(z.string()),
  strategies: z.array(z.string()),
  seeds: z.array(z.number()),
  ticks: z.number(),
  summary: z.array(
    z.object({
      scenario: z.string(),
      strategy: z.string(),
      runs: z.number(),
      avg_wait_mean: z.number(),
      avg_wait_std: z.number(),
      p95_wait_mean: z.number(),
      p95_wait_std: z.number(),
      long_wait_pct_mean: z.number(),
      long_wait_pct_std: z.number(),
      energy_mean: z.number(),
      energy_std: z.number(),
      throughput_mean: z.number(),
      throughput_std: z.number(),
      violations: z.number(),
    }),
  ),
  comparison: z.array(
    z.object({
      scenario: z.string(),
      strategy: z.string(),
      avg_wait_delta_pct: z.number(),
      p95_wait_delta_pct: z.number(),
      long_wait_pct_delta_pct: z.number(),
      energy_delta_pct: z.number(),
      wins_on: z.array(z.string()),
    }),
  ),
  wins: z.record(z.string(), z.number()),
});

export const RunResponseSchema = z.object({
  scenario: z.string(),
  strategy: z.string(),
  seed: z.number(),
  ticks: z.number(),
  runtime_s: z.number(),
  final: z.record(z.string(), z.any()),
  series: z.object({
    tick: z.array(z.number()),
    avg_wait: z.array(z.number()),
    p95_wait: z.array(z.number()),
    waiting: z.array(z.number()),
    riding: z.array(z.number()),
    delivered: z.array(z.number()),
    energy: z.array(z.number()),
    long_wait_pct: z.array(z.number()),
    messages: z.array(z.number()),
  }),
  rules_fired: z.array(z.string()),
  violations: z.array(z.string()),
});

export const BoardResponseSchema = z.object({
  cars: z.array(
    z.object({
      car_id: z.number(),
      tick: z.number(),
      floor: z.number(),
      direction: DirectionSchema,
      load: z.number(),
      capacity: z.number(),
      space: z.number(),
      available: z.boolean(),
      out_of_service: z.boolean(),
      fire_mode: z.boolean(),
      door: z.string(),
      assigned_calls: z.array(z.string()),
      car_calls: z.array(z.number()),
      riders: z.number(),
      plan_end_floor: z.number().nullable(),
      plan_end_eta: z.number(),
      planned_stops: z.number(),
      park_target: z.number().nullable(),
    }),
  ),
  policy: z.object({
    pattern: z.string(),
    weights: z.object({
      wait: z.number(),
      ride: z.number(),
      crowding: z.number(),
      energy: z.number(),
    }),
    published_by: z.string(),
    tick: z.number(),
  }),
  writes: z.number(),
});

/**
 * Cheap top-level validation for 50fps WebSocket snapshots.
 */
export function validateSnapshotShape(data: unknown): boolean {
  if (typeof data !== "object" || data === null) return false;
  const d = data as Record<string, unknown>;
  return (
    typeof d.tick === "number" &&
    Array.isArray(d.cars) &&
    Array.isArray(d.floors) &&
    typeof d.metrics === "object" &&
    d.metrics !== null &&
    typeof d.building === "object" &&
    d.building !== null
  );
}

// ---------------------------------------------------------------- Brain panel (Phase 6)

export const BrainCandidateSchema = z.object({
  car_id: z.number(),
  refused: z.boolean(),
  reason: z.string().nullable(),
  bid: z.number().nullable(),
  wait: z.number().nullable(),
  ride: z.number().nullable(),
  crowding: z.number().nullable(),
  energy: z.number().nullable(),
  net_score: z.number().nullable(),
  attention: z.number().nullable(),
  teacher_bid: z.number().nullable(),
});

export const SearchResultSchema = z.object({
  chosen: z.number(),
  net_choice: z.number(),
  candidates: z.array(z.number()),
  prior: z.array(z.number()),
  visits: z.array(z.number()),
  q: z.array(z.number()),
  root_value: z.number(),
  sims_done: z.number(),
  depth_max: z.number(),
  time_ms: z.number(),
  nn_ms: z.number(),
  leaf: z.string(),
  overridden: z.boolean(),
  fallback: z.boolean(),
});

export const BrainDecisionSchema = z.object({
  tick: z.number(),
  conversation_id: z.union([z.string(), z.number()]),
  call: z.object({ floor: z.number(), direction: z.string() }),
  candidates: z.array(BrainCandidateSchema),
  chosen: z.number(),
  net_choice: z.number().nullable(),
  teacher_choice: z.number().nullable(),
  search: SearchResultSchema.nullable(),
});

export const BrainDecisionsSchema = z.object({ decisions: z.array(BrainDecisionSchema) });

export const BrainSchema = z.object({
  available: z.boolean(),
  strategies: z.array(
    z.object({
      name: z.string(),
      label: z.string(),
      available: z.boolean(),
      reason: z.string().optional(),
      arbiter: z.string().optional(),
    }),
  ),
  active: z.object({
    strategy: z.string(),
    label: z.string(),
    bidder: z.string(),
    arbiter: z.string(),
    model: z
      .object({
        name: z.string(),
        version: z.string(),
        param_count: z.number(),
        feature_version: z.number(),
      })
      .nullable(),
    search: z
      .object({
        sims: z.number(),
        time_budget_ms: z.number(),
        tau_margin: z.number(),
        top_k: z.number(),
        horizon: z.number(),
        leaf: z.string(),
      })
      .nullable(),
    shadow_teacher: z.boolean(),
  }),
  stats: z
    .object({
      decisions: z.number(),
      agree_net_pct: z.number(),
      agree_teacher_pct: z.number().nullable(),
      searched_pct: z.number(),
      overridden_pct: z.number(),
      search_ms_mean: z.number().nullable(),
      search_ms_p95: z.number().nullable(),
    })
    .nullable(),
});

export const ExplainSchema = z.object({
  conversation_id: z.union([z.string(), z.number()]),
  method: z.string(),
  groups: z.array(
    z.object({
      group: z.string(),
      delta_score_chosen: z.number(),
      delta_score_runner_up: z.number().nullable(),
      flips_decision: z.boolean(),
    }),
  ),
});

export type BrainResponse = z.infer<typeof BrainSchema>;
export type BrainDecision = z.infer<typeof BrainDecisionSchema>;
export type ExplainResponse = z.infer<typeof ExplainSchema>;
export type BrainConfigUpdate = {
  sims?: number;
  time_budget_ms?: number;
  tau_margin?: number;
  top_k?: number;
  shadow_teacher?: boolean;
};
