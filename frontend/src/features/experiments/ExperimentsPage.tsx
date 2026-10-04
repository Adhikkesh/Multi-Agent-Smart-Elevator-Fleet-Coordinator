import { BarChart3, Split } from "lucide-react";
import React from "react";
import { useSearchParams } from "react-router-dom";
import { BenchmarkRunner } from "./BenchmarkRunner";
import { CompareMode } from "./CompareMode";

export const ExperimentsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const mode = searchParams.get("mode") || "benchmark";

  return (
    <div className="flex h-full flex-col overflow-y-auto p-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h1 className="text-lg font-bold text-foreground">Experiments & Empirical Validation</h1>
          <p className="text-xs text-muted-foreground">
            Headless Monte-Carlo benchmarks and synchronized strategy replays.
          </p>
        </div>

        <div className="flex rounded-lg border border-border bg-card p-1">
          <button
            onClick={() => setSearchParams({ mode: "benchmark" })}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              mode === "benchmark"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>Monte-Carlo Benchmark</span>
          </button>

          <button
            onClick={() => setSearchParams({ mode: "compare" })}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              mode === "compare"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Split className="h-3.5 w-3.5" />
            <span>Compare Mode (Synchronised Replay)</span>
          </button>
        </div>
      </div>

      {mode === "benchmark" ? <BenchmarkRunner /> : <CompareMode />}
    </div>
  );
};
