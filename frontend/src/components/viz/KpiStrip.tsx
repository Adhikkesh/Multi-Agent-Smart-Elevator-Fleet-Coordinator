import React from "react";
import { formatNumber, formatPercent } from "../../lib/format";
import { useLiveStore } from "../../store/liveStore";
import { KpiTile } from "./KpiTile";

export const KpiStrip: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const metricsHistory = useLiveStore((s) => s.metricsHistory);

  const m = snapshot?.metrics;
  const recentHistory = metricsHistory.slice(-120);

  // Sample from ~60 ticks ago for deltas
  const baselineIndex = Math.max(0, recentHistory.length - 60);
  const baseline = recentHistory[baselineIndex];

  return (
    <div
      aria-label="Key Performance Indicators Strip"
      className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-5 xl:grid-cols-10"
    >
      <KpiTile
        label="Avg Wait"
        value={formatNumber(m?.avg_wait, 1)}
        unit="s"
        history={recentHistory.map((h) => h.avg_wait)}
        currentNumeric={m?.avg_wait}
        baseline={baseline?.avg_wait}
        invertDelta={true}
      />
      <KpiTile
        label="P95 Wait"
        value={formatNumber(m?.p95_wait, 1)}
        unit="s"
        history={recentHistory.map((h) => h.p95_wait)}
        currentNumeric={m?.p95_wait}
        baseline={baseline?.p95_wait}
        invertDelta={true}
      />
      <KpiTile
        label="Max Wait"
        value={formatNumber(m?.max_wait, 1)}
        unit="s"
        history={recentHistory.map((h) => h.max_wait)}
        currentNumeric={m?.max_wait}
        baseline={baseline?.max_wait}
        invertDelta={true}
      />
      <KpiTile
        label="Long Waits"
        value={formatPercent(m?.long_wait_pct, 1)}
        history={recentHistory.map((h) => h.long_wait_pct)}
        currentNumeric={m?.long_wait_pct}
        baseline={baseline?.long_wait_pct}
        invertDelta={true}
      />
      <KpiTile
        label="Throughput"
        value={Math.round(m?.throughput ?? 0)}
        unit="/h"
        history={recentHistory.map((h) => h.throughput)}
        currentNumeric={m?.throughput}
        baseline={baseline?.throughput}
      />
      <KpiTile
        label="Delivered"
        value={`${m?.delivered ?? 0} / ${m?.arrived ?? 0}`}
        history={recentHistory.map((h) => h.delivered)}
      />
      <KpiTile
        label="In System"
        value={`${m?.waiting ?? 0}w · ${m?.riding ?? 0}r`}
        history={recentHistory.map((h) => h.waiting + h.riding)}
      />
      <KpiTile
        label="Energy"
        value={Math.round(m?.energy ?? 0)}
        history={recentHistory.map((h) => h.energy)}
        currentNumeric={m?.energy}
        baseline={baseline?.energy}
        invertDelta={true}
      />
      <KpiTile
        label="Msgs / Call"
        value={formatNumber(m?.messages_per_call, 1)}
        history={recentHistory.map((h) => h.messages_per_call)}
      />
      <KpiTile
        label="Compute"
        value={formatNumber(m?.compute_ms_per_tick, 2)}
        unit="ms"
        history={recentHistory.map((h) => h.compute_ms_per_tick)}
      />
    </div>
  );
};
