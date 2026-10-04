import { TrendingDown, TrendingUp } from "lucide-react";
import React from "react";
import { formatDelta } from "../../lib/format";

interface KpiTileProps {
  label: string;
  value: string | number;
  unit?: string;
  history?: number[];
  baseline?: number;
  currentNumeric?: number;
  invertDelta?: boolean;
}

export const KpiTile: React.FC<KpiTileProps> = ({
  label,
  value,
  unit,
  history = [],
  baseline,
  currentNumeric,
  invertDelta = false,
}) => {
  const delta =
    baseline !== undefined && currentNumeric !== undefined
      ? formatDelta(currentNumeric, baseline, invertDelta)
      : null;

  return (
    <div className="flex flex-col justify-between rounded-lg border border-border bg-card p-2.5 shadow-xs transition-shadow hover:shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider truncate">
          {label}
        </span>
        {delta && !delta.isNeutral && (
          <span
            className={`flex items-center gap-0.5 text-[10px] font-semibold ${
              delta.isGood ? "text-emerald-500" : "text-rose-500"
            }`}
          >
            {delta.text.startsWith("+") ? (
              <TrendingUp className="h-3 w-3" />
            ) : (
              <TrendingDown className="h-3 w-3" />
            )}
            {delta.text}
          </span>
        )}
      </div>

      <div className="my-1 flex items-baseline gap-1">
        <span className="font-mono text-lg font-bold text-foreground num-tabular">{value}</span>
        {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
      </div>

      {/* Inline Sparkline */}
      {history.length > 1 && (
        <div className="mt-1 h-5 w-full">
          <Sparkline points={history} />
        </div>
      )}
    </div>
  );
};

export const Sparkline: React.FC<{ points: number[]; color?: string }> = ({
  points,
  color = "var(--primary, #3b82f6)",
}) => {
  if (points.length < 2) return null;

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;

  const width = 100;
  const height = 20;

  const pathData = points
    .map((val, i) => {
      const x = (i / (points.length - 1)) * width;
      const y = height - ((val - min) / range) * (height - 4) - 2;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full overflow-visible">
      <path
        d={pathData}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};
