import { useQuery } from "@tanstack/react-query";
import { AlertCircle, RefreshCw } from "lucide-react";
import React, { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../api/client";

export const AnnealingTab: React.FC = () => {
  const { data, refetch, isFetching } = useQuery({
    queryKey: ["search-lab-annealing"],
    queryFn: () => api.getAnnealingLab(),
  });

  const sa = data?.simulated_annealing;
  const hc = data?.hill_climbing;

  // Build iteration chart points
  const chartData = useMemo(() => {
    if (!sa?.curve) return [];
    return sa.curve.map((cost, idx) => ({
      iteration: idx,
      current: Math.round(cost * 10) / 10,
      best: sa.best_curve ? Math.round((sa.best_curve[idx] ?? cost) * 10) / 10 : cost,
    }));
  }, [sa]);

  return (
    <div className="space-y-4">
      {/* Explainer */}
      <div className="rounded-lg border border-border/80 bg-muted/20 p-3 text-xs">
        <div className="font-semibold text-foreground flex items-center justify-between">
          <span>What am I looking at?</span>
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-1 text-primary hover:underline"
          >
            <RefreshCw className={`h-3 w-3 ${isFetching ? "animate-spin" : ""}`} />
            Run Reassignment Search
          </button>
        </div>
        <p className="mt-1 text-muted-foreground">
          Simulated Annealing and Hill Climbing optimise global fleet assignment by swapping
          unserved calls between cars. While Hill Climbing easily gets trapped in local minima,
          Simulated Annealing probabilistically accepts worse moves early (governed by Boltzmann
          temperature decay), allowing it to escape suboptimal basins and find lower fleet costs.
        </p>
      </div>

      {/* Synthetic flag notice */}
      {data?.synthetic && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5 text-xs text-amber-500">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>
            Constructed instance — traffic was calm so representative calls were generated from the
            building geometry. The optimisation algorithms and objective cost function are the real
            ones.
          </span>
        </div>
      )}

      {/* Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-xs font-mono">
          <span className="text-[10px] text-muted-foreground uppercase font-sans">
            Initial Fleet Cost
          </span>
          <div className="mt-1 text-lg font-bold text-foreground">
            {sa?.initial_cost.toFixed(1) ?? "—"}
          </div>
        </div>

        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-xs font-mono">
          <span className="text-[10px] text-muted-foreground uppercase font-sans">
            Hill Climbing Final
          </span>
          <div className="mt-1 text-lg font-bold text-foreground">
            {hc?.final_cost.toFixed(1) ?? "—"}
          </div>
          <span className="text-[10px] text-muted-foreground font-sans">
            Greedy descent only
          </span>
        </div>

        <div className="rounded-lg border border-primary/40 bg-primary/5 p-3 shadow-xs font-mono">
          <span className="text-[10px] text-primary uppercase font-sans font-semibold">
            Simulated Annealing Final
          </span>
          <div className="mt-1 text-lg font-bold text-primary">
            {sa?.final_cost.toFixed(1) ?? "—"}
          </div>
          <span className="text-[10px] text-emerald-500 font-sans font-semibold">
            {sa?.improvement ? `-${sa.improvement.toFixed(1)} cost reduction` : ""}
          </span>
        </div>

        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-xs font-mono">
          <span className="text-[10px] text-muted-foreground uppercase font-sans">
            Accepted Worse Moves
          </span>
          <div className="mt-1 text-lg font-bold text-amber-500">
            {sa?.accepted ?? 0} moves
          </div>
          <span className="text-[10px] text-muted-foreground font-sans">
            Escaped local minima
          </span>
        </div>
      </div>

      {/* Annealing Convergence Curve Line Chart */}
      <div className="rounded-lg border border-border/60 bg-card p-4 shadow-xs">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-xs font-semibold text-foreground">
            Simulated Annealing Cost Convergence Curve (Temperature Cooling)
          </h3>
          <span className="text-[11px] font-mono text-muted-foreground">
            {sa?.iterations ?? 400} iterations
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
              <XAxis dataKey="iteration" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--card)",
                  borderColor: "var(--border)",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Line
                type="monotone"
                dataKey="current"
                name="Current Energy"
                stroke="#f59e0b"
                strokeWidth={1}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="best"
                name="Best-So-Far Energy"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
