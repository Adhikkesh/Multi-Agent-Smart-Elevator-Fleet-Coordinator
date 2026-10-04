/**
 * Pure formatting helpers for numbers, timings, deltas, and units.
 */

export function formatTime(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function formatNumber(val: number | null | undefined, decimals = 1): string {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return val.toFixed(decimals);
}

export function formatInt(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return Math.round(val).toLocaleString();
}

export function formatPercent(val: number | null | undefined, decimals = 1): string {
  if (val === null || val === undefined || isNaN(val)) return "—%";
  return `${val.toFixed(decimals)}%`;
}

export function formatDelta(current: number, baseline: number, invert = false): {
  text: string;
  isGood: boolean;
  isNeutral: boolean;
} {
  if (baseline === 0) {
    return { text: "0.0%", isGood: false, isNeutral: true };
  }
  const pct = ((current - baseline) / Math.abs(baseline)) * 100;
  if (Math.abs(pct) < 0.1) {
    return { text: "0.0%", isGood: false, isNeutral: true };
  }
  const sign = pct > 0 ? "+" : "";
  const isGood = invert ? pct < 0 : pct > 0;
  return {
    text: `${sign}${pct.toFixed(1)}%`,
    isGood,
    isNeutral: false,
  };
}
