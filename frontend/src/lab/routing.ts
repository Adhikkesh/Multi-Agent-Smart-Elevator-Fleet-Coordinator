/**
 * Client-side replication of elevator routing state-space search (A*, BFS, UCS, Greedy).
 * Mirrors elevator_mas.planning.routing and search.py exactly.
 */

import type { Direction } from "../api/types";

export interface Stop {
  floor: number;
  kind: "pickup" | "dropoff";
  direction: Direction;
  weight: number;
}

export interface RoutingCosts {
  seconds_per_floor: number;
  dwell: number;
  energy_per_floor: number;
  energy_per_stop: number;
  boarding_per_passenger: number;
}

export const DEFAULT_COSTS: RoutingCosts = {
  seconds_per_floor: 2.0,
  dwell: 4.0,
  energy_per_floor: 0.5,
  energy_per_stop: 1.0,
  boarding_per_passenger: 1.0,
};

export function stopKey(s: Stop): string {
  return `${s.floor}:${s.kind}:${s.direction}`;
}

export function formatStopLabel(s: Stop): string {
  const d = s.direction === "UP" ? "▲" : s.direction === "DOWN" ? "▼" : "●";
  const k = s.kind === "pickup" ? "P" : "D";
  return `${s.floor}${k}${d}`;
}

export function mergeStops(stops: Stop[]): Stop[] {
  const map = new Map<string, Stop>();
  for (const s of stops) {
    const key = stopKey(s);
    const existing = map.get(key);
    if (existing) {
      existing.weight += s.weight;
    } else {
      map.set(key, { ...s });
    }
  }
  return Array.from(map.values());
}

const DIR_ORDER: Record<string, number> = { DOWN: -1, IDLE: 0, UP: 1 };

export function legalStops(current: number, pending: Stop[]): Stop[] {
  const dropoffs = pending.filter((s) => s.kind === "dropoff").map((s) => s.floor);
  const pickups = pending.filter((s) => s.kind === "pickup");
  const out: Stop[] = [];

  for (const stop of pending) {
    if (stop.kind === "dropoff") {
      out.push(stop);
      continue;
    }

    // Constraint 1: never carry a committed rider backwards
    if (dropoffs.length > 0) {
      if (stop.direction === "UP") {
        if (!(current <= stop.floor && dropoffs.every((d) => stop.floor <= d))) {
          continue;
        }
      } else if (stop.direction === "DOWN") {
        if (!(current >= stop.floor && dropoffs.every((d) => stop.floor >= d))) {
          continue;
        }
      }
    }

    // Constraint 2: take call only at the turning point of the sweep
    if (stop.direction === "DOWN") {
      if (pickups.some((other) => other.floor > stop.floor)) {
        continue;
      }
    } else if (pickups.some((other) => other.floor < stop.floor)) {
      continue;
    }

    out.push(stop);
  }

  // Deterministic order: floor asc, kind.value asc, direction.value asc
  out.sort((a, b) => {
    if (a.floor !== b.floor) return a.floor - b.floor;
    if (a.kind !== b.kind) return a.kind.localeCompare(b.kind);
    return (DIR_ORDER[a.direction] ?? 0) - (DIR_ORDER[b.direction] ?? 0);
  });

  return out;
}

export function travelTime(a: number, b: number, costs = DEFAULT_COSTS): number {
  return Math.abs(a - b) * costs.seconds_per_floor;
}

export function dwellFor(stop: Stop, costs = DEFAULT_COSTS): number {
  return costs.dwell + costs.boarding_per_passenger * Math.max(0.0, stop.weight - 1.0);
}

export function actionCost(
  current: number,
  pending: Stop[],
  action: Stop,
  costs = DEFAULT_COSTS,
): number {
  const duration = travelTime(current, action.floor, costs) + dwellFor(action, costs);
  const remainingWeight = pending.reduce((sum, s) => sum + s.weight, 0);
  const energy =
    costs.energy_per_floor * Math.abs(action.floor - current) + costs.energy_per_stop;
  return duration * remainingWeight + energy;
}

export function heuristic(current: number, pending: Stop[], costs = DEFAULT_COSTS): number {
  if (pending.length === 0) return 0.0;
  const travelBound = pending.reduce(
    (sum, s) => sum + s.weight * travelTime(current, s.floor, costs),
    0,
  );
  const floors = pending.map((s) => s.floor);
  const lo = Math.min(...floors);
  const hi = Math.max(...floors);
  const span = hi - lo + Math.min(Math.abs(current - lo), Math.abs(current - hi));
  const energyBound = costs.energy_per_floor * span + costs.energy_per_stop * pending.length;
  return travelBound + energyBound;
}

export function stateKey(floor: number, pending: Stop[]): string {
  const sortedKeys = pending.map((s) => `${s.floor}:${s.kind}:${s.direction}:${s.weight}`).sort();
  return `${floor}|${sortedKeys.join(";")}`;
}

export interface SearchNode {
  id: number;
  floor: number;
  pending: Stop[];
  f: number;
  g: number;
  h: number;
  order: number;
  depth: number;
  parent: SearchNode | null;
  action: Stop | null;
}

export interface RoutingSearchResult {
  algorithm: string;
  found: boolean;
  cost: number;
  nodes_expanded: number;
  nodes_generated: number;
  max_frontier: number;
  route: Stop[];
  routeLabels: string[];
}

export interface StepTraceEvent {
  step: number;
  type: "expand" | "generate" | "goal";
  node: {
    id: number;
    floor: number;
    f: number;
    g: number;
    h: number;
    depth: number;
    pendingCount: number;
    actionLabel: string | null;
  };
  openCount: number;
  closedCount: number;
}

export function solveRoutingProblem(
  startFloor: number,
  rawStops: Stop[],
  algorithm: "astar" | "ucs" | "bfs" | "greedy" = "astar",
  costs = DEFAULT_COSTS,
): { result: RoutingSearchResult; traces: StepTraceEvent[] } {
  const merged = mergeStops(rawStops);
  let orderCounter = 0;
  let nodeIdCounter = 0;

  const initialH = heuristic(startFloor, merged, costs);
  const root: SearchNode = {
    id: nodeIdCounter++,
    floor: startFloor,
    pending: merged,
    g: 0.0,
    h: initialH,
    f: 0.0,
    order: orderCounter++,
    depth: 0,
    parent: null,
    action: null,
  };

  const evalFn = (node: SearchNode, hVal: number): number => {
    switch (algorithm) {
      case "ucs":
        return node.g;
      case "greedy":
        return hVal;
      case "bfs":
        return node.depth;
      case "astar":
      default:
        return node.g + hVal;
    }
  };

  root.f = evalFn(root, initialH);

  if (merged.length === 0) {
    return {
      result: {
        algorithm,
        found: true,
        cost: 0.0,
        nodes_expanded: 0,
        nodes_generated: 1,
        max_frontier: 1,
        route: [],
        routeLabels: [],
      },
      traces: [],
    };
  }

  const frontier: SearchNode[] = [root];
  const reached = new Map<string, number>();
  reached.set(stateKey(startFloor, merged), 0.0);

  let expanded = 0;
  let generated = 1;
  let maxFrontier = 1;
  const traces: StepTraceEvent[] = [];

  const earlyGoalTest = algorithm === "bfs";

  while (frontier.length > 0) {
    // Pop node with smallest f, tie-break by order
    frontier.sort((a, b) => {
      if (Math.abs(a.f - b.f) > 1e-9) return a.f - b.f;
      return a.order - b.order;
    });

    const node = frontier.shift()!;
    const key = stateKey(node.floor, node.pending);

    // Stale check
    if (node.g > (reached.get(key) ?? Infinity)) {
      continue;
    }

    if (!earlyGoalTest && node.pending.length === 0) {
      // Goal found!
      traces.push({
        step: traces.length,
        type: "goal",
        node: {
          id: node.id,
          floor: node.floor,
          f: node.f,
          g: node.g,
          h: node.h,
          depth: node.depth,
          pendingCount: node.pending.length,
          actionLabel: node.action ? formatStopLabel(node.action) : null,
        },
        openCount: frontier.length,
        closedCount: expanded,
      });

      const route: Stop[] = [];
      let curr: SearchNode | null = node;
      while (curr && curr.action) {
        route.unshift(curr.action);
        curr = curr.parent;
      }

      return {
        result: {
          algorithm,
          found: true,
          cost: Math.round(node.g * 1000) / 1000,
          nodes_expanded: expanded,
          nodes_generated: generated,
          max_frontier: maxFrontier,
          route,
          routeLabels: route.map((s) => {
            const kindChar = s.kind === "dropoff" ? "D" : s.direction === "UP" ? "▲" : "▼";
            return `${s.floor}${kindChar}`;
          }),
        },
        traces,
      };
    }

    expanded++;
    if (traces.length < 500) {
      traces.push({
        step: traces.length,
        type: "expand",
        node: {
          id: node.id,
          floor: node.floor,
          f: node.f,
          g: node.g,
          h: node.h,
          depth: node.depth,
          pendingCount: node.pending.length,
          actionLabel: node.action ? formatStopLabel(node.action) : null,
        },
        openCount: frontier.length,
        closedCount: expanded,
      });
    }

    const legals = legalStops(node.floor, node.pending);
    for (const action of legals) {
      const childPending = node.pending.filter((s) => s !== action);
      const step = actionCost(node.floor, node.pending, action, costs);
      const childCost = node.g + step;
      const childKey = stateKey(action.floor, childPending);

      if (childCost >= (reached.get(childKey) ?? Infinity)) {
        continue;
      }

      const childH = heuristic(action.floor, childPending, costs);
      const childNode: SearchNode = {
        id: nodeIdCounter++,
        floor: action.floor,
        pending: childPending,
        g: childCost,
        h: childH,
        f: 0.0,
        order: orderCounter++,
        depth: node.depth + 1,
        parent: node,
        action,
      };
      childNode.f = evalFn(childNode, childH);

      reached.set(childKey, childCost);
      generated++;

      if (earlyGoalTest && childPending.length === 0) {
        const route: Stop[] = [];
        let curr: SearchNode | null = childNode;
        while (curr && curr.action) {
          route.unshift(curr.action);
          curr = curr.parent;
        }
        return {
          result: {
            algorithm,
            found: true,
            cost: Math.round(childCost * 1000) / 1000,
            nodes_expanded: expanded,
            nodes_generated: generated,
            max_frontier: maxFrontier,
            route,
            routeLabels: route.map((s) => `${s.floor}${s.kind === "dropoff" ? "D" : s.direction}`),
          },
          traces,
        };
      }

      frontier.push(childNode);
      maxFrontier = Math.max(maxFrontier, frontier.length);
    }
  }

  return {
    result: {
      algorithm,
      found: false,
      cost: 0.0,
      nodes_expanded: expanded,
      nodes_generated: generated,
      max_frontier: maxFrontier,
      route: [],
      routeLabels: [],
    },
    traces,
  };
}
