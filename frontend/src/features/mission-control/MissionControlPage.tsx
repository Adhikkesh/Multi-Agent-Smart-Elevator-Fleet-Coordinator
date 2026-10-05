import React from "react";
import { AuctionPanel } from "../../components/viz/AuctionPanel";
import { BrainPanel } from "../../components/viz/BrainPanel";
import { Building } from "../../components/viz/Building";
import { ChaosPanel } from "../../components/viz/ChaosPanel";
import { FooterProgressBar } from "../../components/viz/FooterProgressBar";
import { KpiStrip } from "../../components/viz/KpiStrip";
import { MessageStream } from "../../components/viz/MessageStream";
import { SafetyCard } from "../../components/viz/SafetyCard";
import { TrafficMonitorCard } from "../../components/viz/TrafficMonitorCard";

export const MissionControlPage: React.FC = () => {
  return (
    <div className="flex h-full flex-col justify-between overflow-y-auto p-4 space-y-4">
      {/* 1. Live KPI Strip */}
      <section aria-label="Key Performance Indicators" className="shrink-0">
        <KpiStrip />
      </section>

      {/* 2. Main Floor Grid: Building in Center, Auctions & Messages flanking */}
      {/* shrink-0: never let the column flexbox squeeze this row below its content, or the
          building overflows onto the panels underneath on laptop-height screens. */}
      <div className="grid shrink-0 grid-cols-1 gap-4 lg:grid-cols-12 min-h-[520px]">
        {/* Left Column: Auction & Decision Trace (4 cols) */}
        <div className="space-y-4 lg:col-span-4 flex flex-col justify-between">
          <AuctionPanel />
          <BrainPanel />
        </div>

        {/* Center Column: Building View (5 cols) */}
        <div className="lg:col-span-5 flex flex-col">
          <Building />
        </div>

        {/* Right Column: Live FIPA-ACL Messages (3 cols) */}
        <div className="lg:col-span-3 flex flex-col">
          <MessageStream />
        </div>
      </div>

      {/* 3. Intelligence, Safety & Disturbance Controls Grid */}
      <div className="grid shrink-0 grid-cols-1 gap-4 md:grid-cols-3">
        <TrafficMonitorCard />
        <SafetyCard />
        <ChaosPanel />
      </div>

      {/* 4. Footer timeline */}
      <FooterProgressBar />
    </div>
  );
};
