import { CheckCircle2, Flame, Shield, ShieldAlert } from "lucide-react";
import React from "react";
import { useLiveStore } from "../../store/liveStore";

export const SafetyCard: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const rules = snapshot?.rules ?? [];
  const fireAlarm = snapshot?.fire_alarm ?? false;

  return (
    <div className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <Shield className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">Safety Agent (Knowledge-Based)</h2>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span className="flex items-center gap-1 font-semibold text-emerald-500">
            <CheckCircle2 className="h-3.5 w-3.5" /> Invariants OK
          </span>
        </div>
      </div>

      {/* Emergency Fire Alarm Banner */}
      {fireAlarm && (
        <div className="mb-3 flex items-center justify-between rounded-lg border border-destructive/50 bg-destructive/15 p-3 text-destructive animate-pulse">
          <div className="flex items-center gap-2">
            <Flame className="h-5 w-5" />
            <div>
              <div className="text-xs font-bold uppercase tracking-wider">
                Emergency Fire Alarm Active
              </div>
              <div className="text-[11px] text-destructive/90">
                Rule R1 fired · Hall calls blocked · Cars recalled to lobby
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Safety Rules Timeline */}
      <div className="space-y-2">
        <div className="text-[11px] font-medium text-muted-foreground">
          Recent Forward-Chaining Production Rules Fired:
        </div>
        {rules.length > 0 ? (
          <div className="max-h-40 space-y-1.5 overflow-y-auto pr-1">
            {rules.slice(-10).map((r, i) => (
              <div
                key={i}
                className="flex flex-col gap-0.5 rounded-lg border border-border/60 bg-muted/20 p-2 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 font-mono font-bold text-foreground">
                    <ShieldAlert className="h-3.5 w-3.5 text-amber-500" />
                    {r.rule}
                  </span>
                  <span className="font-mono text-[10px] text-muted-foreground">t={r.tick}s</span>
                </div>
                <div className="text-[11px] text-muted-foreground">{r.effect}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex h-20 items-center justify-center rounded-lg border border-dashed border-border/60 text-xs text-muted-foreground">
            No safety rules fired yet. Building operating normally.
          </div>
        )}
      </div>
    </div>
  );
};
