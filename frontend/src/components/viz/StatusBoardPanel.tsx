import { useQuery } from "@tanstack/react-query";
import { Eye, Layers } from "lucide-react";
import React from "react";
import { api } from "../../api/client";
import { getDirectionIcon } from "../../lib/colors";
import { useLiveStore } from "../../store/liveStore";

export const StatusBoardPanel: React.FC = () => {
  const currentTick = useLiveStore((s) => s.snapshot?.tick ?? 0);

  const { data: board } = useQuery({
    queryKey: ["board"],
    queryFn: () => api.getBoard(),
    refetchInterval: 1000,
  });

  return (
    <div className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">
            Shared Status Board (Multi-Agent Blackboard)
          </h3>
        </div>
        <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <Eye className="h-3 w-3" /> Total Writes: {board?.writes ?? 0}
        </span>
      </div>

      <div className="mb-2 text-xs italic text-muted-foreground">
        "What every agent is allowed to see; nobody reads another agent's private state."
      </div>

      {/* Board Table */}
      <div className="overflow-x-auto rounded-lg border border-border/60">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-border bg-muted/40 font-semibold text-foreground">
            <tr>
              <th className="p-2">Car</th>
              <th className="p-2">Floor</th>
              <th className="p-2">Dir</th>
              <th className="p-2">Load / Space</th>
              <th className="p-2">Status</th>
              <th className="p-2">Assigned Calls</th>
              <th className="p-2">Car Calls</th>
              <th className="p-2">Plan End (ETA)</th>
              <th className="p-2">Target</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40 font-mono">
            {board?.cars.map((c) => {
              const wasUpdatedThisTick = c.tick === currentTick;

              return (
                <tr
                  key={c.car_id}
                  className={`transition-colors hover:bg-muted/30 ${
                    wasUpdatedThisTick ? "bg-primary/5" : ""
                  }`}
                >
                  <td className="p-2 font-bold text-foreground">Car {c.car_id}</td>
                  <td className="p-2">F{c.floor}</td>
                  <td className="p-2 font-bold text-primary">{getDirectionIcon(c.direction)}</td>
                  <td className="p-2">
                    {c.load} / {c.capacity} ({c.space} left)
                  </td>
                  <td className="p-2">
                    {c.out_of_service ? (
                      <span className="rounded bg-rose-500/20 px-1 py-0.5 text-[10px] font-bold text-rose-500">
                        FAULT
                      </span>
                    ) : c.fire_mode ? (
                      <span className="rounded bg-red-600/20 px-1 py-0.5 text-[10px] font-bold text-red-600">
                        FIRE
                      </span>
                    ) : c.available ? (
                      <span className="rounded bg-emerald-500/20 px-1 py-0.5 text-[10px] font-bold text-emerald-500">
                        AVAILABLE
                      </span>
                    ) : (
                      <span className="text-muted-foreground">BUSY</span>
                    )}
                  </td>
                  <td className="p-2">
                    {c.assigned_calls.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {c.assigned_calls.map((call, idx) => (
                          <span
                            key={idx}
                            className="rounded bg-secondary px-1 py-0.5 text-[10px] text-foreground"
                          >
                            {call}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="p-2">
                    {c.car_calls.length > 0 ? c.car_calls.join(", ") : "—"}
                  </td>
                  <td className="p-2">
                    {c.plan_end_floor !== null
                      ? `F${c.plan_end_floor} (${c.plan_end_eta}s)`
                      : "—"}
                  </td>
                  <td className="p-2">
                    {c.park_target !== null ? `Park F${c.park_target}` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
