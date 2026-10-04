import { useQuery } from "@tanstack/react-query";
import { Battery, Cpu, Route, X } from "lucide-react";
import React from "react";
import { api } from "../../api/client";
import { useUiStore } from "../../store/uiStore";

export const AgentInspector: React.FC = () => {
  const isOpen = useUiStore((s) => s.inspectorOpen);
  const address = useUiStore((s) => s.inspectorAgentAddress);
  const close = useUiStore((s) => s.closeInspector);

  const { data: agent, isLoading } = useQuery({
    queryKey: ["agent", address],
    queryFn: () => (address ? api.inspectAgent(address) : null),
    enabled: isOpen && !!address,
    refetchInterval: isOpen ? 500 : false, // Poll only while drawer is open
  });

  if (!isOpen || !address) return null;

  const isCar = address.startsWith("car-");
  const isDispatcher = address === "dispatcher";

  return (
    <div
      role="dialog"
      aria-label="Agent Inspector"
      className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-border bg-card shadow-2xl transition-transform"
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground font-mono">{address}</h2>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">
              {agent?.agent_type || "Autonomous Agent"}
            </div>
          </div>
        </div>
        <button
          onClick={close}
          aria-label="Close inspector drawer"
          className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 space-y-4 overflow-y-auto p-4 text-xs">
        {isLoading && !agent ? (
          <div className="flex h-32 items-center justify-center text-muted-foreground">
            Inspecting internal memory...
          </div>
        ) : agent ? (
          <>
            {/* PEAS Formulation Card */}
            {agent.peas && (
              <div className="rounded-lg border border-border/80 bg-muted/20 p-3 space-y-2">
                <span className="font-semibold text-foreground text-xs uppercase tracking-wider">
                  PEAS Specification (AIMA 4e)
                </span>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div>
                    <span className="font-bold text-primary">Performance:</span>
                    <p className="text-muted-foreground mt-0.5">{agent.peas.performance}</p>
                  </div>
                  <div>
                    <span className="font-bold text-primary">Environment:</span>
                    <p className="text-muted-foreground mt-0.5">{agent.peas.environment}</p>
                  </div>
                  <div>
                    <span className="font-bold text-primary">Actuators:</span>
                    <p className="text-muted-foreground mt-0.5">{agent.peas.actuators}</p>
                  </div>
                  <div>
                    <span className="font-bold text-primary">Sensors:</span>
                    <p className="text-muted-foreground mt-0.5">{agent.peas.sensors}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Car-Specific Inspector */}
            {isCar && (
              <div className="space-y-3">
                <div className="grid grid-cols-3 gap-2">
                  <div className="rounded border border-border/60 bg-muted/30 p-2 text-center">
                    <span className="text-[10px] text-muted-foreground uppercase">Load</span>
                    <div className="font-mono text-base font-bold text-foreground">
                      {agent.load ?? 0} / {agent.capacity ?? 10}
                    </div>
                  </div>
                  <div className="rounded border border-border/60 bg-muted/30 p-2 text-center">
                    <span className="text-[10px] text-muted-foreground uppercase">Direction</span>
                    <div className="font-mono text-base font-bold text-primary">
                      {agent.direction ?? "IDLE"}
                    </div>
                  </div>
                  <div className="rounded border border-border/60 bg-muted/30 p-2 text-center">
                    <span className="text-[10px] text-muted-foreground uppercase">Door</span>
                    <div className="font-mono text-base font-bold text-foreground">
                      {agent.door ?? "closed"}
                    </div>
                  </div>
                </div>

                {/* Planned Route */}
                <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-1.5">
                  <div className="flex items-center gap-1.5 font-semibold text-foreground">
                    <Route className="h-3.5 w-3.5 text-primary" />
                    <span>A* Planned Stop Sequence</span>
                  </div>
                  {agent.route && agent.route.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {agent.route.map((st: any, idx: number) => (
                        <span
                          key={idx}
                          className="rounded bg-secondary px-2 py-0.5 font-mono text-[11px] font-bold text-foreground"
                        >
                          F{st.floor} {st.kind === "dropoff" ? "●" : st.direction === "UP" ? "▲" : "▼"}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-muted-foreground italic">No pending stops planned.</p>
                  )}
                </div>

                {/* Replans & Search Costs */}
                <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-1.5 font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Search Nodes:</span>
                    <span className="font-bold">{agent.plan_nodes_expanded ?? agent.plan_nodes ?? 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Plan Cost (Wait+Energy):</span>
                    <span className="font-bold">
                      {typeof agent.plan_cost === "number" ? agent.plan_cost.toFixed(1) : (agent.plan_cost ?? "0.0")}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Energy (Floors):</span>
                    <span className="font-bold flex items-center gap-1">
                      <Battery className="h-3 w-3 text-emerald-500" />
                      {typeof agent.energy === "object" && agent.energy !== null
                        ? agent.energy.floors_travelled
                        : (typeof agent.energy === "number" ? agent.energy.toFixed(0) : (agent.energy ?? 0))}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Dispatcher-Specific Inspector */}
            {isDispatcher && (
              <div className="space-y-3 font-mono">
                <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-2">
                  <div className="font-semibold text-foreground text-xs uppercase tracking-wider">
                    Dispatcher Fleet State
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Total Auctions Held:</span>
                      <span className="font-bold">{agent.auction_history_count ?? 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Current Strategy:</span>
                      <span className="font-bold text-primary">{agent.strategy ?? "full"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Active Assignments:</span>
                      <span className="font-bold">{Object.keys(agent.assignments ?? {}).length}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Raw JSON State Inspector */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-semibold text-muted-foreground uppercase">
                Raw Agent Memory
              </span>
              <pre className="max-h-56 overflow-auto rounded bg-muted/50 p-2.5 font-mono text-[10px] text-foreground">
                {JSON.stringify(agent, null, 2)}
              </pre>
            </div>
          </>
        ) : (
          <div className="text-muted-foreground">No agent data found.</div>
        )}
      </div>
    </div>
  );
};
