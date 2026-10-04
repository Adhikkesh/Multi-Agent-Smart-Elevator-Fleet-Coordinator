import React from "react";
import { formatTime } from "../../lib/format";
import { useLiveStore } from "../../store/liveStore";

export const FooterProgressBar: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const tick = snapshot?.tick ?? 0;
  const duration = snapshot?.session.duration ?? 330;
  const scenario = snapshot?.scenario ?? "demo_story";

  const progressPct = Math.min(100, Math.max(0, (tick / (duration || 1)) * 100));

  // Phase segments for demo_story (YAML boundaries: 60/150/240/330)
  const isDemoStory = scenario === "demo_story";

  return (
    <footer
      aria-label="Simulation Scenario Timeline"
      className="flex h-9 w-full shrink-0 items-center justify-between border-t border-border bg-card px-4 text-xs select-none"
    >
      <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
        <span className="font-semibold text-foreground">Timeline:</span>
        <span>
          {formatTime(tick)} / {formatTime(duration)}
        </span>
      </div>

      {/* Progress Bar Container with phases */}
      <div className="relative mx-4 flex-1 h-3 rounded-full bg-muted/60 overflow-hidden border border-border/40">
        {/* Phase markers for demo_story */}
        {isDemoStory && (
          <div className="absolute inset-0 flex">
            {/* Phase 1: Calm (0..60s) */}
            <div
              style={{ width: `${(60 / 330) * 100}%` }}
              className="h-full border-r border-border/40 bg-sky-500/10"
              title="Phase 1: Calm traffic & Strategic parking"
            />
            {/* Phase 2: Rush (60..150s) */}
            <div
              style={{ width: `${((150 - 60) / 330) * 100}%` }}
              className="h-full border-r border-border/40 bg-amber-500/10"
              title="Phase 2: Up-peak morning rush at lobby"
            />
            {/* Phase 3: Fault (150..240s) */}
            <div
              style={{ width: `${((240 - 150) / 330) * 100}%` }}
              className="h-full border-r border-border/40 bg-rose-500/10"
              title="Phase 3: Car breakdown & re-auction"
            />
            {/* Phase 4: Fire (240..330s) */}
            <div
              style={{ width: `${((330 - 240) / 330) * 100}%` }}
              className="h-full bg-red-600/15"
              title="Phase 4: Emergency fire alarm drill"
            />
          </div>
        )}

        {/* Live Elapsed Progress indicator */}
        <div
          style={{ width: `${progressPct}%` }}
          className="relative h-full bg-primary/80 transition-all duration-150"
        />
      </div>

      <div className="flex items-center gap-2 text-[11px] text-muted-foreground font-mono">
        <span>{progressPct.toFixed(0)}%</span>
      </div>
    </footer>
  );
};
