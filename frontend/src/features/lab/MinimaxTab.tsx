import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import React from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../../api/client";

export const MinimaxTab: React.FC = () => {
  const { data, refetch, isFetching } = useQuery({
    queryKey: ["search-lab-minimax"],
    queryFn: () => api.getMinimaxLab(),
  });

  const chartData = [
    {
      name: "Node Expansions",
      Minimax: data?.minimax_nodes ?? 0,
      "Alpha-Beta": data?.alphabeta_nodes ?? 0,
    },
  ];

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
            Run Adversarial Solver
          </button>
        </div>
        <p className="mt-1 text-muted-foreground">
          Idle cars choose parking spots against an adversary who places worst-case demand spikes
          at the building's high-traffic floors. By formulating parking as a zero-sum game, Minimax
          minimises maximum passenger pickup latency. Alpha-Beta pruning prunes search branches that
          cannot influence the final decision, producing the identical optimal placement in a
          fraction of the node expansions.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs font-mono">
        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-xs">
          <span className="text-[10px] text-muted-foreground uppercase font-sans">
            Idle Cars Placed
          </span>
          <div className="mt-1 text-lg font-bold text-foreground">
            {data?.cars_placed ?? 0} cars
          </div>
        </div>

        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-xs">
          <span className="text-[10px] text-muted-foreground uppercase font-sans">
            Minimax Full Tree
          </span>
          <div className="mt-1 text-lg font-bold text-foreground">
            {data?.minimax_nodes ?? 0} nodes
          </div>
        </div>

        <div className="rounded-lg border border-primary/40 bg-primary/5 p-3 shadow-xs">
          <span className="text-[10px] text-primary uppercase font-sans font-semibold">
            Alpha-Beta Pruned
          </span>
          <div className="mt-1 text-lg font-bold text-primary">
            {data?.alphabeta_nodes ?? 0} nodes
          </div>
        </div>

        <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 shadow-xs">
          <span className="text-[10px] text-emerald-600 uppercase font-sans font-semibold">
            Pruning Savings
          </span>
          <div className="mt-1 text-lg font-bold text-emerald-500">
            {data?.pruning_saving_pct.toFixed(1) ?? "0.0"}%
          </div>
        </div>
      </div>

      {/* Visual Building Diagram for Parking & Pruning Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Node Comparison Bar Chart */}
        <div className="rounded-lg border border-border/60 bg-card p-4 shadow-xs">
          <h3 className="mb-2 text-xs font-semibold text-foreground">
            Minimax vs Alpha-Beta Node Evaluation
          </h3>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--card)",
                    borderColor: "var(--border)",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Bar dataKey="Minimax" fill="#f87171" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Alpha-Beta" fill="#34d399" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Strategic Floor Layout */}
        <div className="rounded-lg border border-border/60 bg-card p-4 shadow-xs flex flex-col justify-between">
          <h3 className="mb-2 text-xs font-semibold text-foreground">
            Optimal Strategic Parking Allocation
          </h3>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between border-b border-border/60 pb-1.5 font-sans">
              <span className="text-muted-foreground">Game Value (Worst-Case Latency):</span>
              <span className="font-bold text-primary">{data?.value.toFixed(1) ?? "0.0"}s</span>
            </div>

            <div className="flex flex-col gap-1 font-sans">
              <span className="text-muted-foreground">Likely Demand Floors (Adversary Focus):</span>
              <div className="flex flex-wrap gap-1">
                {data?.likely_floors.map((fl) => (
                  <span key={fl} className="rounded bg-rose-500/10 px-2 py-0.5 text-rose-500 font-mono font-bold">
                    F{fl}
                  </span>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-1 pt-1 font-sans">
              <span className="text-muted-foreground">Optimal Chosen Parking Spots:</span>
              <div className="flex flex-wrap gap-1.5">
                {data?.best_parking &&
                  Object.entries(data.best_parking).map(([carId, fl]) => (
                    <span
                      key={carId}
                      className="rounded border border-primary bg-primary/10 px-2.5 py-1 font-mono font-bold text-primary"
                    >
                      Car {carId} → Floor {fl}
                    </span>
                  ))}
              </div>
            </div>
          </div>

          <div className="mt-4 rounded bg-muted/30 p-2 text-[11px] text-muted-foreground italic font-sans">
            "Candidate parking spots: {data?.candidate_spots.join(", ") ?? "0, 7, 14"}."
          </div>
        </div>
      </div>
    </div>
  );
};
