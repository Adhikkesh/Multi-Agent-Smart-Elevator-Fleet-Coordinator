import { useQuery } from "@tanstack/react-query";
import { toPng } from "html-to-image";
import { Download, Play, StopCircle, Trophy } from "lucide-react";
import React, { useRef, useState } from "react";
import { api } from "../../api/client";
import type { BenchmarkResponse } from "../../api/types";

export const BenchmarkRunner: React.FC = () => {
  const { data: meta } = useQuery({
    queryKey: ["meta"],
    queryFn: () => api.getMeta(),
  });

  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([
    "morning_up_peak",
    "evening_down_peak",
    "lunch_two_way",
  ]);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([
    "nearest_car",
    "collective",
    "cnp_astar",
    "full",
  ]);
  const [seedsCount, setSeedsCount] = useState(3);
  const [ticksCount, setTicksCount] = useState(600);

  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<BenchmarkResponse | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const exportRef = useRef<HTMLDivElement>(null);

  const handleRun = async () => {
    setIsRunning(true);
    const controller = new AbortController();
    setAbortController(controller);

    try {
      const res = await api.benchmark(
        {
          scenarios: selectedScenarios,
          strategies: selectedStrategies,
          seeds: seedsCount,
          ticks: ticksCount,
        },
        controller.signal,
      );
      setResults(res);
    } catch (err: any) {
      if (err.name !== "AbortError") {
        console.error("Benchmark failed:", err);
      }
    } finally {
      setIsRunning(false);
      setAbortController(null);
    }
  };

  const handleCancel = () => {
    if (abortController) {
      abortController.abort();
    }
    setIsRunning(false);
  };

  const handleExportCSV = () => {
    if (!results) return;
    const headers = [
      "scenario",
      "strategy",
      "avg_wait_mean",
      "avg_wait_std",
      "p95_wait_mean",
      "p95_wait_std",
      "long_wait_pct_mean",
      "energy_mean",
      "throughput_mean",
    ];
    const rows = results.summary.map((r) => [
      r.scenario,
      r.strategy,
      r.avg_wait_mean,
      r.avg_wait_std,
      r.p95_wait_mean,
      r.p95_wait_std,
      r.long_wait_pct_mean,
      r.energy_mean,
      r.throughput_mean,
    ]);
    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `liftzero_benchmark_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportPNG = async () => {
    if (!exportRef.current) return;
    try {
      const dataUrl = await toPng(exportRef.current, { backgroundColor: "#0f172a" });
      const link = document.createElement("a");
      link.download = `liftzero_benchmark_${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error("PNG export error:", err);
    }
  };

  return (
    <div className="space-y-4">
      {/* Controls Card */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm space-y-3">
        <h2 className="text-sm font-semibold text-foreground">Configure Headless Benchmark Suite</h2>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          {/* Scenarios (multi) */}
          <div className="space-y-1">
            <span className="font-medium text-muted-foreground">Scenarios:</span>
            <div className="max-h-28 overflow-y-auto space-y-1 rounded border border-border p-2">
              {meta?.scenarios.map((scen) => (
                <label key={scen} className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedScenarios.includes(scen)}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedScenarios([...selectedScenarios, scen]);
                      else setSelectedScenarios(selectedScenarios.filter((s) => s !== scen));
                    }}
                    className="rounded border-border text-primary focus:ring-1 focus:ring-primary"
                  />
                  <span>{scen}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Strategies (multi) */}
          <div className="space-y-1">
            <span className="font-medium text-muted-foreground">Strategies:</span>
            <div className="max-h-28 overflow-y-auto space-y-1 rounded border border-border p-2">
              {meta?.strategies.map((st) => (
                <label key={st.name} className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedStrategies.includes(st.name)}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedStrategies([...selectedStrategies, st.name]);
                      else setSelectedStrategies(selectedStrategies.filter((s) => s !== st.name));
                    }}
                    className="rounded border-border text-primary focus:ring-1 focus:ring-primary"
                  />
                  <span>{st.name}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Seeds & Ticks */}
          <div className="space-y-2">
            <div>
              <span className="font-medium text-muted-foreground">Seeds (1..10):</span>
              <input
                aria-label="Number of random seeds"
                type="number"
                min={1}
                max={10}
                value={seedsCount}
                onChange={(e) => setSeedsCount(parseInt(e.target.value) || 1)}
                className="mt-1 w-full rounded border border-border bg-background p-1.5 font-mono"
              />
            </div>
            <div>
              <span className="font-medium text-muted-foreground">Ticks (60..3600):</span>
              <input
                aria-label="Simulation duration in ticks"
                type="number"
                min={60}
                max={3600}
                step={60}
                value={ticksCount}
                onChange={(e) => setTicksCount(parseInt(e.target.value) || 60)}
                className="mt-1 w-full rounded border border-border bg-background p-1.5 font-mono"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-col justify-end space-y-2">
            {!isRunning ? (
              <button
                onClick={handleRun}
                disabled={selectedScenarios.length === 0 || selectedStrategies.length === 0}
                className="flex items-center justify-center gap-2 rounded-lg bg-primary py-2.5 font-semibold text-primary-foreground shadow-xs hover:bg-primary/90 disabled:opacity-50"
              >
                <Play className="h-4 w-4" /> Run Benchmark
              </button>
            ) : (
              <button
                onClick={handleCancel}
                className="flex items-center justify-center gap-2 rounded-lg bg-destructive py-2.5 font-semibold text-destructive-foreground hover:bg-destructive/90"
              >
                <StopCircle className="h-4 w-4" /> Cancel Execution
              </button>
            )}

            <div className="flex gap-2">
              <button
                onClick={handleExportCSV}
                disabled={!results}
                className="flex flex-1 items-center justify-center gap-1 rounded border border-border bg-secondary py-1 text-[11px] font-medium text-secondary-foreground hover:bg-secondary/80 disabled:opacity-40"
              >
                <Download className="h-3 w-3" /> Export CSV
              </button>
              <button
                onClick={handleExportPNG}
                disabled={!results}
                className="flex flex-1 items-center justify-center gap-1 rounded border border-border bg-secondary py-1 text-[11px] font-medium text-secondary-foreground hover:bg-secondary/80 disabled:opacity-40"
              >
                <Download className="h-3 w-3" /> Export PNG
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Progress / Status banner */}
      {isRunning && (
        <div className="flex items-center justify-between rounded-lg border border-primary/30 bg-primary/10 p-3 text-xs">
          <div className="flex items-center gap-2 text-primary font-semibold">
            <span className="h-2.5 w-2.5 rounded-full bg-primary animate-ping" />
            Executing Monte-Carlo Benchmark in headless background thread...
          </div>
          <span className="font-mono text-muted-foreground">
            {selectedScenarios.length * selectedStrategies.length * seedsCount} total runs
          </span>
        </div>
      )}

      {/* Results Container */}
      {results && (
        <div ref={exportRef} className="space-y-4">
          {/* Wins Banner */}
          {results.wins && (
            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-card p-3 text-xs">
              <div className="flex items-center gap-1.5 font-bold text-amber-500">
                <Trophy className="h-4 w-4" />
                <span>Strategy Win Counts:</span>
              </div>
              {Object.entries(results.wins).map(([st, wins]) => (
                <span
                  key={st}
                  className="rounded bg-secondary px-2.5 py-1 font-mono font-bold text-foreground"
                >
                  {st}: {wins} wins
                </span>
              ))}
            </div>
          )}

          {/* Comparison Table vs Baseline (nearest_car) */}
          <div className="rounded-lg border border-border/60 bg-card p-4 shadow-xs">
            <h3 className="mb-2 text-xs font-semibold text-foreground">
              Comparison vs Nearest-Car Baseline (% Improvement, Losses in Red)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="border-b border-border bg-muted/40 font-sans font-semibold text-foreground">
                  <tr>
                    <th className="p-2">Scenario</th>
                    <th className="p-2">Strategy</th>
                    <th className="p-2">Avg Wait Δ%</th>
                    <th className="p-2">P95 Wait Δ%</th>
                    <th className="p-2">Long Wait Δ%</th>
                    <th className="p-2">Energy Δ%</th>
                    <th className="p-2 font-sans">Wins On</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {results.comparison.map((r, i) => {
                    const waitGood = r.avg_wait_delta_pct < 0;
                    const p95Good = r.p95_wait_delta_pct < 0;

                    return (
                      <tr key={i} className="hover:bg-muted/20">
                        <td className="p-2 font-sans font-medium text-foreground">{r.scenario}</td>
                        <td className="p-2 font-bold text-primary">{r.strategy}</td>
                        <td
                          className={`p-2 font-bold ${
                            waitGood ? "text-emerald-500" : "text-rose-500"
                          }`}
                        >
                          {r.avg_wait_delta_pct > 0 ? "+" : ""}
                          {r.avg_wait_delta_pct.toFixed(1)}%
                        </td>
                        <td
                          className={`p-2 font-bold ${
                            p95Good ? "text-emerald-500" : "text-rose-500"
                          }`}
                        >
                          {r.p95_wait_delta_pct > 0 ? "+" : ""}
                          {r.p95_wait_delta_pct.toFixed(1)}%
                        </td>
                        <td className="p-2">
                          {r.long_wait_pct_delta_pct > 0 ? "+" : ""}
                          {r.long_wait_pct_delta_pct.toFixed(1)}%
                        </td>
                        <td className="p-2">
                          {r.energy_delta_pct > 0 ? "+" : ""}
                          {r.energy_delta_pct.toFixed(1)}%
                        </td>
                        <td className="p-2 font-sans text-muted-foreground">
                          {r.wins_on.join(", ") || "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Full Aggregate Summary Table */}
          <div className="rounded-lg border border-border/60 bg-card p-4 shadow-xs">
            <h3 className="mb-2 text-xs font-semibold text-foreground">
              Statistical Aggregate Summary (Mean ± Std Across Seeds)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="border-b border-border bg-muted/40 font-sans font-semibold text-foreground">
                  <tr>
                    <th className="p-2">Scenario</th>
                    <th className="p-2">Strategy</th>
                    <th className="p-2">Avg Wait (s)</th>
                    <th className="p-2">P95 Wait (s)</th>
                    <th className="p-2">Long Wait %</th>
                    <th className="p-2">Energy</th>
                    <th className="p-2">Throughput (/h)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {results.summary.map((r, i) => (
                    <tr key={i} className="hover:bg-muted/20">
                      <td className="p-2 font-sans font-medium text-foreground">{r.scenario}</td>
                      <td className="p-2 font-bold text-primary">{r.strategy}</td>
                      <td className="p-2">
                        {r.avg_wait_mean.toFixed(1)} ± {r.avg_wait_std.toFixed(1)}
                      </td>
                      <td className="p-2">
                        {r.p95_wait_mean.toFixed(1)} ± {r.p95_wait_std.toFixed(1)}
                      </td>
                      <td className="p-2">{r.long_wait_pct_mean.toFixed(1)}%</td>
                      <td className="p-2">{Math.round(r.energy_mean)}</td>
                      <td className="p-2">{Math.round(r.throughput_mean)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
