/**
 * Theme & categorical colors for performatives, utility terms, and car states.
 */

import type { Direction, Performative } from "../api/types";

export const PERFORMATIVE_COLORS: Record<Performative, string> = {
  REQUEST: "var(--perf-request, #38bdf8)",
  CFP: "var(--perf-cfp, #fbbf24)",
  PROPOSE: "var(--perf-propose, #a78bfa)",
  REFUSE: "var(--perf-refuse, #f87171)",
  ACCEPT_PROPOSAL: "var(--perf-accept, #34d399)",
  REJECT_PROPOSAL: "var(--perf-reject, #fb923c)",
  INFORM: "var(--perf-inform, #60a5fa)",
  CANCEL: "var(--perf-cancel, #94a3b8)",
  FAILURE: "var(--perf-failure, #ef4444)",
};

export const UTILITY_COLORS = {
  wait: "var(--term-wait, #38bdf8)",
  ride: "var(--term-ride, #a78bfa)",
  crowding: "var(--term-crowding, #fbbf24)",
  energy: "var(--term-energy, #34d399)",
};

export function getPerformativeColor(perf: string): string {
  return PERFORMATIVE_COLORS[perf as Performative] || "#94a3b8";
}

export function getCarStateBorder(state: string, outOfService: boolean, fireMode: boolean): string {
  if (fireMode) return "border-red-500 animate-pulse";
  if (outOfService) return "border-slate-500 opacity-60";
  switch (state) {
    case "moving_up":
    case "moving_down":
      return "border-primary shadow-sm shadow-primary/20";
    case "doors":
      return "border-emerald-500 shadow-sm shadow-emerald-500/20";
    case "idle":
    default:
      return "border-border";
  }
}

export function getDirectionIcon(dir: Direction): string {
  switch (dir) {
    case "UP":
      return "▲";
    case "DOWN":
      return "▼";
    case "IDLE":
    default:
      return "—";
  }
}
