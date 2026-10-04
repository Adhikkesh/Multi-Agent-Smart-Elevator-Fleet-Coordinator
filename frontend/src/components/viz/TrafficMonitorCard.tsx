import { Activity, AlertCircle, CheckCircle2 } from "lucide-react";
import React from "react";
import { UTILITY_COLORS } from "../../lib/colors";
import { useLiveStore } from "../../store/liveStore";

export const TrafficMonitorCard: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const traffic = snapshot?.traffic;
  const parking = snapshot?.parking ?? {};

  if (!traffic) return null;

  const isMatched = traffic.pattern === traffic.true_pattern;

  return (
    <div className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">Traffic Monitor (Learning Agent)</h2>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          {isMatched ? (
            <span className="flex items-center gap-1 font-semibold text-emerald-500">
              <CheckCircle2 className="h-3.5 w-3.5" /> Pattern Identified
            </span>
          ) : (
            <span
              className="flex items-center gap-1 font-semibold text-amber-500"
              title="Partial observability: learning agent has not yet seen enough arrivals to match ground truth"
            >
              <AlertCircle className="h-3.5 w-3.5" /> Learning...
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3 text-xs">
        {/* Pattern comparison: Inferred vs Ground Truth */}
        <div className="grid grid-cols-2 gap-2 rounded-lg border border-border/60 bg-muted/20 p-2.5">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Inferred Pattern
            </div>
            <div className="mt-0.5 font-mono text-sm font-bold text-foreground">
              {traffic.pattern.toUpperCase()}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Hidden Ground Truth
            </div>
            <div className="mt-0.5 font-mono text-sm font-medium text-muted-foreground">
              {traffic.true_pattern.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Reason */}
        <div className="rounded bg-muted/30 p-2 italic text-muted-foreground">
          "{traffic.reason || "Observing Poisson arrivals across floors"}"
        </div>

        {/* Live Weight Bars */}
        <div className="space-y-1.5">
          <div className="text-[11px] font-medium text-muted-foreground">
            Current Dispatch Utility Weights:
          </div>
          <div className="grid grid-cols-4 gap-2">
            <WeightBar label="Wait" val={traffic.weights.wait} color={UTILITY_COLORS.wait} />
            <WeightBar label="Ride" val={traffic.weights.ride} color={UTILITY_COLORS.ride} />
            <WeightBar label="Crowd" val={traffic.weights.crowding} color={UTILITY_COLORS.crowding} />
            <WeightBar label="Energy" val={traffic.weights.energy} color={UTILITY_COLORS.energy} />
          </div>
        </div>

        {/* Strategic Parking */}
        {Object.keys(parking).length > 0 && (
          <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-border/60 text-[11px]">
            <span className="text-muted-foreground">Strategic Parking:</span>
            {Object.entries(parking).map(([carId, fl]) => (
              <span
                key={carId}
                className="rounded border border-primary/30 bg-primary/10 px-1.5 py-0.5 font-mono text-primary font-semibold"
              >
                C{carId} → F{fl}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const WeightBar: React.FC<{ label: string; val: number; color: string }> = ({
  label,
  val,
  color,
}) => (
  <div className="rounded border border-border/40 bg-muted/40 p-1.5">
    <div className="flex justify-between text-[10px] text-muted-foreground">
      <span>{label}</span>
      <span className="font-mono font-bold text-foreground">{val.toFixed(2)}</span>
    </div>
    <div className="mt-1 h-1.5 w-full rounded bg-muted overflow-hidden">
      <div
        style={{ width: `${Math.min(100, (val / 2.0) * 100)}%`, background: color }}
        className="h-full transition-all duration-300"
      />
    </div>
  </div>
);
