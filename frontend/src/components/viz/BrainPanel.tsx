import { Brain, Cpu, Network, Sparkles } from "lucide-react";
import React from "react";
import { useLiveStore } from "../../store/liveStore";

export const BrainPanel: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const brain = snapshot?.brain;

  return (
    <div className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-purple-400" />
          <h2 className="text-sm font-semibold text-foreground">
            LiftZero Neural Bidder (Phase 6)
          </h2>
        </div>
        <span className="rounded-full bg-purple-500/10 px-2 py-0.5 text-[10px] font-semibold text-purple-400">
          {brain ? "Active Model" : "Placeholder"}
        </span>
      </div>

      {brain ? (
        <div className="space-y-3 text-xs">
          <div className="flex justify-between font-mono">
            <span className="text-muted-foreground">Model: {brain.model}</span>
            <span className="text-muted-foreground">Params: {brain.params.toLocaleString()}</span>
            <span className="text-muted-foreground">Latency: {brain.latency_ms}ms</span>
          </div>
          {brain.decision && (
            <div className="rounded bg-muted/40 p-2 font-mono">
              Chosen Car: {brain.decision.chosen}
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-4 text-center text-xs">
          <div className="mb-3 flex items-center gap-4 text-muted-foreground/60">
            <div className="flex flex-col items-center gap-1">
              <Network className="h-6 w-6 text-primary/70" />
              <span className="text-[10px]">Fleet State</span>
            </div>
            <span className="font-mono text-xs">→</span>
            <div className="flex flex-col items-center gap-1">
              <Cpu className="h-7 w-7 text-purple-400/80" />
              <span className="text-[10px]">Deep Policy</span>
            </div>
            <span className="font-mono text-xs">→</span>
            <div className="flex flex-col items-center gap-1">
              <Sparkles className="h-6 w-6 text-amber-400/80" />
              <span className="text-[10px]">Learned Bids</span>
            </div>
          </div>
          <p className="font-medium text-foreground">
            LiftZero model not loaded — running the classical Contract-Net bidder.
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground max-w-sm">
            Phase 6 will train a neural network against A* expert demonstrations to replace the
            heuristic evaluation function with a learned valuation function.
          </p>
        </div>
      )}
    </div>
  );
};
