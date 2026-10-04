import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Pause, Play, RefreshCw, Sparkles } from "lucide-react";
import React, { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../api/client";
import { solveRoutingProblem, type StepTraceEvent, type Stop } from "../../lab/routing";

export const RoutingTab: React.FC<{ selectedCar?: number }> = ({ selectedCar }) => {
  const { data, refetch, isFetching } = useQuery({
    queryKey: ["search-lab-routing", selectedCar],
    queryFn: () => api.getSearchLabRouting(selectedCar),
  });

  // Step-through visualizer state
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playSpeedMs = 500;

  // Compute client-side A* step-through traces
  const traces: StepTraceEvent[] = useMemo(() => {
    if (!data || !data.stops || data.stops.length === 0) return [];
    const clientStops: Stop[] = data.stops.slice(0, 6).map((s) => ({
      floor: s.floor,
      kind: s.kind as "pickup" | "dropoff",
      direction: s.direction as any,
      weight: s.weight,
    }));
    const { traces } = solveRoutingProblem(data.current_floor, clientStops, "astar");
    return traces;
  }, [data]);

  const activeTrace = traces[currentStepIdx] || traces[0];

  // Auto-play timer for step-through
  React.useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentStepIdx((prev) => {
        if (prev >= traces.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, playSpeedMs);
    return () => clearInterval(interval);
  }, [isPlaying, traces.length, playSpeedMs]);

  // Chart data for nodes
  const chartData = useMemo(() => {
    if (!data?.results) return [];
    return data.results.map((r) => ({
      algorithm: r.algorithm.toUpperCase(),
      expanded: r.nodes_expanded,
      generated: r.nodes_generated,
      frontier: r.max_frontier,
    }));
  }, [data]);

  return (
    <div className="space-y-4">
      {/* What am I looking at explainer */}
      <div className="rounded-lg border border-border/80 bg-muted/20 p-3 text-xs">
        <div className="font-semibold text-foreground flex items-center justify-between">
          <span>What am I looking at?</span>
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-1 text-primary hover:underline"
          >
            <RefreshCw className={`h-3 w-3 ${isFetching ? "animate-spin" : ""}`} />
            Run on Live Car
          </button>
        </div>
        <p className="mt-1 text-muted-foreground">
          We snapshot the active call routing problem for Car {data?.car_id ?? 0} and run BFS, UCS,
          Greedy Best-First, and A* side by side on the exact same state space. A* uses an admissible
          and consistent heuristic (weighted passenger travel + line span energy bound), guaranteeing
          optimal sequence while expanding far fewer nodes than Uniform-Cost Search.
        </p>
      </div>

      {/* Results Comparison Table */}
      <div className="overflow-x-auto rounded-lg border border-border/60 bg-card p-3 shadow-xs">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs font-semibold text-foreground">
            Search Algorithms Comparison (Start: Floor {data?.current_floor ?? 0}, h(n₀) ={" "}
            {data?.h_at_start ?? 0})
          </span>
          <span className="text-[11px] text-muted-foreground font-mono">Source: {data?.source}</span>
        </div>

        <table className="w-full text-left text-xs font-mono">
          <thead className="border-b border-border bg-muted/40 font-sans font-semibold text-foreground">
            <tr>
              <th className="p-2">Algorithm</th>
              <th className="p-2">Cost (Wait+Energy)</th>
              <th className="p-2">Nodes Expanded</th>
              <th className="p-2">Nodes Generated</th>
              <th className="p-2">Max Frontier</th>
              <th className="p-2">Runtime</th>
              <th className="p-2">Optimal Route</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {data?.results.map((r) => (
              <tr key={r.algorithm} className="hover:bg-muted/20">
                <td className="p-2 font-bold text-foreground flex items-center gap-1.5">
                  {r.optimal && (
                    <span className="rounded bg-emerald-500/20 px-1 text-[9px] font-bold text-emerald-500">
                      OPTIMAL
                    </span>
                  )}
                  {r.algorithm.toUpperCase()}
                </td>
                <td className="p-2 font-bold text-primary">{r.cost.toFixed(1)}</td>
                <td className="p-2">{r.nodes_expanded}</td>
                <td className="p-2">{r.nodes_generated}</td>
                <td className="p-2">{r.max_frontier}</td>
                <td className="p-2 text-muted-foreground">{r.runtime_ms.toFixed(3)} ms</td>
                <td className="p-2">
                  <div className="flex flex-wrap gap-1">
                    {r.route.map((st, i) => (
                      <span key={i} className="rounded bg-secondary px-1.5 py-0.5 text-[10px]">
                        {st}
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Nodes Expanded Chart */}
      <div className="rounded-lg border border-border/60 bg-card p-4 shadow-xs">
        <h3 className="mb-2 text-xs font-semibold text-foreground">
          Node Expansion & Frontier Comparison
        </h3>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
              <XAxis dataKey="algorithm" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--card)",
                  borderColor: "var(--border)",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Bar dataKey="expanded" name="Nodes Expanded" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="generated" name="Nodes Generated" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="frontier" name="Max Frontier" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Interactive Step-Through A* Visualizer */}
      <div className="rounded-lg border border-border/60 bg-card p-4 shadow-xs space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <h3 className="text-xs font-semibold text-foreground">
              A* Step-Through Search Trace (Open & Closed Lists)
            </h3>
          </div>

          {/* Player controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentStepIdx((p) => Math.max(0, p - 1))}
              disabled={currentStepIdx === 0}
              className="rounded p-1 hover:bg-muted disabled:opacity-30"
              title="Previous Step"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center gap-1 rounded bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90"
            >
              {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
              {isPlaying ? "Pause" : "Play"}
            </button>

            <button
              onClick={() => setCurrentStepIdx((p) => Math.min(traces.length - 1, p + 1))}
              disabled={currentStepIdx >= traces.length - 1}
              className="rounded p-1 hover:bg-muted disabled:opacity-30"
              title="Next Step"
            >
              <ChevronRight className="h-4 w-4" />
            </button>

            <span className="font-mono text-xs text-muted-foreground ml-2">
              Step {currentStepIdx + 1} / {Math.max(1, traces.length)}
            </span>
          </div>
        </div>

        {activeTrace ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            {/* Current Selected Node */}
            <div className="rounded-lg border border-primary/30 bg-primary/5 p-3 space-y-1.5 font-mono">
              <div className="font-semibold font-sans text-primary">Current Node Being Evaluated</div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">State Floor:</span>
                <span className="font-bold">Floor {activeTrace.node.floor}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Action Taken:</span>
                <span className="font-bold text-foreground">
                  {activeTrace.node.actionLabel || "Initial State"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Evaluation f = g + h:</span>
                <span className="font-bold text-foreground">
                  {activeTrace.node.f.toFixed(1)} = {activeTrace.node.g.toFixed(1)} +{" "}
                  {activeTrace.node.h.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Search Depth:</span>
                <span>{activeTrace.node.depth}</span>
              </div>
            </div>

            {/* Open List Status */}
            <div className="rounded-lg border border-border/60 bg-muted/20 p-3 space-y-1 font-mono">
              <div className="font-semibold font-sans text-foreground">
                Frontier (Open List)
              </div>
              <div className="text-muted-foreground">
                Queued candidate nodes in min-heap:
              </div>
              <div className="text-xl font-bold text-primary">{activeTrace.openCount} nodes</div>
            </div>

            {/* Closed List Status */}
            <div className="rounded-lg border border-border/60 bg-muted/20 p-3 space-y-1 font-mono">
              <div className="font-semibold font-sans text-foreground">
                Explored (Closed List)
              </div>
              <div className="text-muted-foreground">Expanded states so far:</div>
              <div className="text-xl font-bold text-emerald-500">
                {activeTrace.closedCount} nodes
              </div>
            </div>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">No stops to step through.</div>
        )}
      </div>
    </div>
  );
};
