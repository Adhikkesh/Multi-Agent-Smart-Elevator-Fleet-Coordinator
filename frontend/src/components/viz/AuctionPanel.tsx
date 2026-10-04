import { Gavel, History } from "lucide-react";
import React from "react";
import { useLiveStore } from "../../store/liveStore";
import { BidBars } from "./BidBars";

export const AuctionPanel: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const auctionHistory = useLiveStore((s) => s.auctionHistory);
  const selectedAuctionIndex = useLiveStore((s) => s.selectedAuctionIndex);
  const selectAuction = useLiveStore((s) => s.selectAuction);

  // Active auction: selected one from history or latest live auction
  const currentAuction =
    selectedAuctionIndex !== null && auctionHistory[selectedAuctionIndex]
      ? auctionHistory[selectedAuctionIndex]
      : snapshot?.auction;

  const weights = snapshot?.traffic.weights;

  return (
    <div
      data-testid="auction-panel"
      className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm"
    >
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <Gavel className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">
            Contract Net Auction: Why Did Winner Get Call?
          </h2>
        </div>

        {/* History rounds selector */}
        {auctionHistory.length > 0 && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <History className="h-3.5 w-3.5" />
            <select
              aria-label="Past Auctions"
              value={selectedAuctionIndex ?? "latest"}
              onChange={(e) => {
                const val = e.target.value;
                selectAuction(val === "latest" ? null : parseInt(val, 10));
              }}
              className="rounded border border-border bg-background px-1.5 py-0.5 text-xs text-foreground"
            >
              <option value="latest">Latest Round</option>
              {auctionHistory.map((a, idx) => (
                <option key={a.conversation_id || idx} value={idx}>
                  t={a.tick} · F{a.floor}{a.direction} · winner C{a.winner ?? "none"}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {currentAuction ? (
        <div className="space-y-3">
          {/* Decision Trace Quote Block */}
          <blockquote
            aria-live="polite"
            className="rounded-lg border-l-4 border-primary bg-primary/5 p-2.5 text-xs text-foreground/90 italic"
          >
            "{currentAuction.reason || "Dispatching call to lowest cost bidder."}"
          </blockquote>

          {/* Bid Breakdown Stacked Bars */}
          <BidBars auction={currentAuction} weights={weights} />
        </div>
      ) : (
        <div className="flex h-32 flex-col items-center justify-center text-xs text-muted-foreground">
          <span>No Contract Net auction round yet.</span>
          <span className="text-[11px] text-muted-foreground/70">
            Auctions trigger dynamically when hall calls arrive.
          </span>
        </div>
      )}
    </div>
  );
};
