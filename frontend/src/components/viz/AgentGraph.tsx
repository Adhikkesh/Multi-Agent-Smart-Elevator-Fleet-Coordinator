import {
  Background,
  Controls,
  type Edge,
  type Node,
  ReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Play } from "lucide-react";
import React, { useMemo, useState } from "react";
import { getPerformativeColor } from "../../lib/colors";
import { useLiveStore } from "../../store/liveStore";
import { useUiStore } from "../../store/uiStore";

export const AgentGraph: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const recentMessages = useLiveStore((s) => s.recentMessages);
  const openInspector = useUiStore((s) => s.openInspector);
  const selectConv = useUiStore((s) => s.selectConversation);

  const [expandedFloors, setExpandedFloors] = useState(false);

  // Generate nodes layout:
  // Dispatcher at center (250, 200)
  // Safety at (120, 50), Monitor at (380, 50)
  // Cars arrayed in arc or line: (50 + i * 110, 320)
  // Floors group at (250, 440)
  const nodes: Node[] = useMemo(() => {
    const cars = snapshot?.cars ?? [];
    const floors = snapshot?.floors ?? [];
    const list: Node[] = [
      {
        id: "dispatcher",
        position: { x: 260, y: 170 },
        data: {
          label: (
            <div
              onClick={() => openInspector("dispatcher")}
              className="cursor-pointer rounded-lg border-2 border-primary bg-card p-2 text-center shadow-md transition-all hover:scale-105"
            >
              <div className="font-mono text-xs font-bold text-foreground">DispatcherAgent</div>
              <div className="text-[10px] text-primary font-medium">Goal / Utility / Auctioneer</div>
            </div>
          ),
        },
      },
      {
        id: "safety",
        position: { x: 60, y: 40 },
        style: { width: 190 },
        data: {
          label: (
            <div
              onClick={() => openInspector("safety")}
              className="cursor-pointer rounded-lg border border-rose-500/70 bg-card p-2 text-center shadow-xs transition-all hover:scale-105"
            >
              <div className="font-mono text-xs font-bold text-rose-500">SafetyAgent</div>
              <div className="text-[10px] text-muted-foreground">Knowledge-Based</div>
            </div>
          ),
        },
      },
      {
        id: "monitor",
        position: { x: 420, y: 40 },
        style: { width: 190 },
        data: {
          label: (
            <div
              onClick={() => openInspector("monitor")}
              className="cursor-pointer rounded-lg border border-purple-500/70 bg-card p-2 text-center shadow-xs transition-all hover:scale-105"
            >
              <div className="font-mono text-xs font-bold text-purple-400">TrafficMonitorAgent</div>
              <div className="text-[10px] text-muted-foreground">Learning Agent</div>
            </div>
          ),
        },
      },
    ];

    // Cars nodes
    cars.forEach((c, idx) => {
      list.push({
        id: `car-${c.car_id}`,
        // Centred under the dispatcher (x 260, width 150), 170 px apart so nodes never overlap.
        position: { x: 260 + (idx - (cars.length - 1) / 2) * 170, y: 310 },
        data: {
          label: (
            <div
              onClick={() => openInspector(`car-${c.car_id}`)}
              className="cursor-pointer rounded-lg border border-border bg-card p-2 text-center shadow-xs transition-all hover:scale-105"
            >
              <div className="font-mono text-xs font-bold text-foreground">Car-{c.car_id}</div>
              <div className="text-[10px] text-muted-foreground">
                F{c.floor} · {c.state}
              </div>
            </div>
          ),
        },
      });
    });

    // Floors node
    list.push({
      id: "floors",
      position: { x: 260, y: 440 },
      data: {
        label: (
          <div
            onClick={() => setExpandedFloors(!expandedFloors)}
            className="cursor-pointer rounded-lg border border-border/80 bg-secondary/40 p-2 text-center shadow-xs"
          >
            <div className="font-mono text-xs font-semibold text-foreground">
              Floors Agent Group ({floors.length})
            </div>
            <div className="text-[10px] text-muted-foreground">Model-based Reflex</div>
          </div>
        ),
      },
    });

    return list;
  }, [snapshot?.cars, snapshot?.floors, expandedFloors, openInspector]);

  // Edges based on recent messages
  const edges: Edge[] = useMemo(() => {
    const edgeMap = new Map<string, { count: number; lastPerf: string }>();

    recentMessages.slice(-40).forEach((m) => {
      let src = m.sender;
      let dst = m.receiver;
      if (src.startsWith("floor-")) src = "floors";
      if (dst.startsWith("floor-")) dst = "floors";

      if (src && dst && src !== dst) {
        const key = `${src}->${dst}`;
        const existing = edgeMap.get(key) || { count: 0, lastPerf: m.performative };
        existing.count += 1;
        existing.lastPerf = m.performative;
        edgeMap.set(key, existing);
      }
    });

    const edgeList: Edge[] = [];
    edgeMap.forEach((val, key) => {
      const [source, target] = key.split("->");
      if (source && target) {
        edgeList.push({
          id: key,
          source,
          target,
          animated: true,
          style: {
            stroke: getPerformativeColor(val.lastPerf),
            strokeWidth: Math.min(5, 1.5 + val.count * 0.4),
          },
        });
      }
    });

    return edgeList;
  }, [recentMessages]);

  const handleReplayLastConv = () => {
    if (recentMessages.length === 0) return;
    const lastMsg = recentMessages[recentMessages.length - 1];
    if (lastMsg) {
      selectConv(lastMsg.conversation_id);
    }
  };

  return (
    <div className="flex flex-col h-[520px] rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between border-b border-border pb-2">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Multi-Agent Communication Network</h2>
          <span className="text-[11px] text-muted-foreground">
            Live FIPA-ACL messages pulse along directed agent edges.
          </span>
        </div>

        <button
          onClick={handleReplayLastConv}
          className="flex items-center gap-1.5 rounded-lg border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
        >
          <Play className="h-3.5 w-3.5" />
          <span>Replay Last Auction Conversation</span>
        </button>
      </div>

      <div className="relative h-[440px] w-full rounded-lg overflow-hidden border border-border/40">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.5}
          maxZoom={1.5}
        >
          <Background gap={16} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
};
