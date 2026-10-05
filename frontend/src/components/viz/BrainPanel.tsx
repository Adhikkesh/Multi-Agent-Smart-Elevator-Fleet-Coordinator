import { BrainCircuit, ChevronRight, Cpu } from "lucide-react";
import React from "react";
import { Link } from "react-router-dom";
import { useLiveStore } from "../../store/liveStore";

/** Mission Control card: which bidder the cars use right now, and a link to the Brain page. */
export const BrainPanel: React.FC = () => {
  const strategy = useLiveStore((s) => s.snapshot?.strategy) ?? "";
  const learned = strategy.startsWith("liftzero");
  const lookahead = strategy.endsWith("mcts");

  return (
    <div className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <BrainCircuit className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">How the cars bid</h2>
        </div>
        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
            learned ? "bg-primary/15 text-primary" : "bg-secondary text-secondary-foreground"
          }`}
        >
          {learned ? (lookahead ? "Learned + look-ahead" : "Learned (LiftZero)") : "Classical A*"}
        </span>
      </div>
      <div className="flex items-start gap-3 text-xs">
        <Cpu className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
        <p className="text-muted-foreground">
          {learned
            ? "Each car prices the call with the LiftZero neural network (trained by imitating the A* bid)" +
              (lookahead ? "; close calls are settled by a short look-ahead search." : ".")
            : "Each car bids the marginal cost of adding the call to its own A*-planned route. Switch the strategy to a LiftZero option to see learned bids."}
        </p>
      </div>
      <Link
        to="/brain"
        className="mt-3 inline-flex items-center gap-1 self-start rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground hover:bg-secondary"
      >
        Open LiftZero Brain <ChevronRight className="h-3.5 w-3.5" />
      </Link>
    </div>
  );
};
