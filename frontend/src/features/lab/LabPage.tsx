import { Activity, Cpu, ShieldCheck } from "lucide-react";
import React from "react";
import { useSearchParams } from "react-router-dom";
import { AnnealingTab } from "./AnnealingTab";
import { MinimaxTab } from "./MinimaxTab";
import { RoutingTab } from "./RoutingTab";

export const LabPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const currentTab = searchParams.get("tab") || "routing";

  const setTab = (tab: string) => {
    setSearchParams({ tab });
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto p-4 space-y-4">
      {/* Header and Tab Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h1 className="text-lg font-bold text-foreground">Algorithm & State-Space Search Lab</h1>
          <p className="text-xs text-muted-foreground">
            AIMA 4e Classical Search, Local Search, and Adversarial Games running on live fleet
            instances.
          </p>
        </div>

        {/* Tab Buttons */}
        <div className="flex rounded-lg border border-border bg-card p-1">
          <button
            onClick={() => setTab("routing")}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              currentTab === "routing"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Cpu className="h-3.5 w-3.5" />
            <span>1. Car Routing (BFS/UCS/Greedy/A*)</span>
          </button>

          <button
            onClick={() => setTab("annealing")}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              currentTab === "annealing"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            <span>2. Assignment (Simulated Annealing)</span>
          </button>

          <button
            onClick={() => setTab("minimax")}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              currentTab === "minimax"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <ShieldCheck className="h-3.5 w-3.5" />
            <span>3. Adversarial Parking (Minimax/Alpha-Beta)</span>
          </button>
        </div>
      </div>

      {/* Tab Panels */}
      {currentTab === "routing" && <RoutingTab />}
      {currentTab === "annealing" && <AnnealingTab />}
      {currentTab === "minimax" && <MinimaxTab />}
    </div>
  );
};
