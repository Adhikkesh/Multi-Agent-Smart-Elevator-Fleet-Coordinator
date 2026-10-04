import { Trophy, XCircle } from "lucide-react";
import React from "react";
import type { Auction, TrafficWeights } from "../../api/types";
import { UTILITY_COLORS } from "../../lib/colors";
import { formatNumber } from "../../lib/format";

interface BidBarsProps {
  auction: Auction;
  weights?: TrafficWeights;
}

export const BidBars: React.FC<BidBarsProps> = ({ auction, weights }) => {
  const bids = auction.bids;

  // Filter out refused bids with null total when computing max scale
  const validBids = bids.filter((b) => !b.refused && b.total !== null && typeof b.total === "number");
  const maxTotal = validBids.length > 0 ? Math.max(...validBids.map((b) => b.total as number), 10) : 10;

  return (
    <div className="space-y-2">
      {/* Legend showing live weights */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 pb-2 text-[11px] text-muted-foreground">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-xs" style={{ background: UTILITY_COLORS.wait }} />
            Wait (w={weights?.wait.toFixed(1) ?? "1.0"})
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-xs" style={{ background: UTILITY_COLORS.ride }} />
            Ride (w={weights?.ride.toFixed(1) ?? "0.5"})
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-xs" style={{ background: UTILITY_COLORS.crowding }} />
            Crowd (w={weights?.crowding.toFixed(1) ?? "0.3"})
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-xs" style={{ background: UTILITY_COLORS.energy }} />
            Energy (w={weights?.energy.toFixed(1) ?? "0.2"})
          </span>
        </div>
        <span className="font-mono text-[10px]">Call: F{auction.floor} {auction.direction}</span>
      </div>

      {/* Stacked bars per car */}
      <div className="space-y-2 pt-1">
        {bids.map((bid) => {
          const isWinner = bid.car_id === auction.winner;

          if (bid.refused || bid.total === null) {
            return (
              <div
                key={bid.car_id}
                className="flex items-center gap-2 rounded border border-border/40 bg-muted/30 p-1.5 opacity-70"
              >
                <span className="w-10 font-mono text-xs font-bold text-muted-foreground">
                  Car {bid.car_id}
                </span>
                <div className="flex flex-1 items-center gap-1 rounded bg-muted/60 px-2 py-1 text-xs text-rose-400">
                  <XCircle className="h-3.5 w-3.5" />
                  <span className="font-medium">REFUSED — {bid.reason || "Full / Out of service"}</span>
                </div>
              </div>
            );
          }

          // Width percentage of the bar relative to maxTotal
          const barWidthPct = Math.min(100, Math.max(5, ((bid.total || 0) / maxTotal) * 100));

          // Component proportions
          const sumTerms = bid.wait + bid.ride + bid.crowding + bid.energy || 1;
          const waitPct = (bid.wait / sumTerms) * 100;
          const ridePct = (bid.ride / sumTerms) * 100;
          const crowdPct = (bid.crowding / sumTerms) * 100;
          const energyPct = (bid.energy / sumTerms) * 100;

          return (
            <div
              key={bid.car_id}
              className={`flex items-center gap-2 rounded-lg border p-1.5 transition-colors ${
                isWinner
                  ? "border-primary/50 bg-primary/10 shadow-xs"
                  : "border-border/60 bg-card/60"
              }`}
            >
              <div className="flex w-12 items-center gap-1 font-mono text-xs font-bold text-foreground">
                {isWinner && <Trophy className="h-3 w-3 text-amber-400" />}
                Car {bid.car_id}
              </div>

              {/* Stacked Bar Container */}
              <div className="flex flex-1 items-center gap-2">
                <div className="h-4.5 w-full rounded overflow-hidden bg-muted/40">
                  <div
                    style={{ width: `${barWidthPct}%` }}
                    className="flex h-full transition-all duration-300"
                  >
                    <div
                      style={{ width: `${waitPct}%`, background: UTILITY_COLORS.wait }}
                      title={`Wait cost: ${bid.wait.toFixed(1)}`}
                    />
                    <div
                      style={{ width: `${ridePct}%`, background: UTILITY_COLORS.ride }}
                      title={`Ride cost: ${bid.ride.toFixed(1)}`}
                    />
                    <div
                      style={{ width: `${crowdPct}%`, background: UTILITY_COLORS.crowding }}
                      title={`Crowding cost: ${bid.crowding.toFixed(1)}`}
                    />
                    <div
                      style={{ width: `${energyPct}%`, background: UTILITY_COLORS.energy }}
                      title={`Energy cost: ${bid.energy.toFixed(1)}`}
                    />
                  </div>
                </div>

                <div className="flex w-24 shrink-0 items-baseline justify-end gap-1 font-mono text-xs">
                  <span className="font-bold text-foreground">{formatNumber(bid.total, 1)}</span>
                  {bid.eta !== null && (
                    <span className="text-[10px] text-muted-foreground">({bid.eta}s)</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
