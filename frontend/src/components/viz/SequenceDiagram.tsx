import React from "react";
import { getPerformativeColor } from "../../lib/colors";
import { buildSequenceDiagram } from "../../lib/sequence";
import { useLiveStore } from "../../store/liveStore";
import { useUiStore } from "../../store/uiStore";

export const SequenceDiagram: React.FC = () => {
  const recentMessages = useLiveStore((s) => s.recentMessages);
  const selectedConvId = useUiStore((s) => s.selectedConversationId);
  const selectConv = useUiStore((s) => s.selectConversation);

  // If none explicitly selected, pick the latest conversation from messages
  const effectiveConvId =
    selectedConvId ||
    (recentMessages.length > 0 ? recentMessages[recentMessages.length - 1]?.conversation_id : null);

  const seq = effectiveConvId ? buildSequenceDiagram(recentMessages, effectiveConvId) : null;

  if (!seq || seq.steps.length === 0) {
    return (
      <div className="flex h-64 flex-col items-center justify-center rounded-xl border border-dashed border-border p-6 text-center text-xs text-muted-foreground">
        <span>No conversation selected or no messages in protocol trace.</span>
        <span className="mt-1 text-[11px] text-muted-foreground/70">
          Hover or click a conversation chip #{effectiveConvId || "c..."} to visualize its FIPA-ACL lifelines.
        </span>
      </div>
    );
  }

  const { participants, steps } = seq;
  const colWidth = Math.max(120, 500 / (participants.length || 1));
  const rowHeight = 44;
  const svgWidth = Math.max(500, participants.length * colWidth + 60);
  const svgHeight = (steps.length + 1) * rowHeight + 60;

  const getParticipantX = (p: string) => {
    const idx = participants.indexOf(p);
    return idx >= 0 ? 50 + idx * colWidth : 50;
  };

  return (
    <div className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-foreground">
            Protocol Sequence Diagram: #{seq.conversationId}
          </h3>
          <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-mono font-bold text-primary">
            {steps.length} exchanges
          </span>
        </div>
        {selectedConvId && (
          <button
            onClick={() => selectConv(null)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            Clear selection
          </button>
        )}
      </div>

      <div className="overflow-x-auto rounded-lg border border-border/40 bg-muted/10 p-2">
        <svg width={svgWidth} height={svgHeight} className="mx-auto block select-none">
          <defs>
            <marker
              id="arrowhead"
              markerWidth="8"
              markerHeight="6"
              refX="7"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 8 3, 0 6" fill="currentColor" className="text-foreground" />
            </marker>
          </defs>

          {/* Participant headers and vertical lifelines */}
          {participants.map((p) => {
            const x = getParticipantX(p);
            return (
              <g key={p}>
                {/* Header box */}
                <rect
                  x={x - 45}
                  y={10}
                  width={90}
                  height={26}
                  rx={4}
                  className="fill-secondary stroke-border stroke-1"
                />
                <text
                  x={x}
                  y={27}
                  textAnchor="middle"
                  className="fill-foreground font-mono text-[11px] font-bold"
                >
                  {p}
                </text>
                {/* Lifeline dashed line */}
                <line
                  x1={x}
                  y1={36}
                  x2={x}
                  y2={svgHeight - 20}
                  strokeDasharray="4 4"
                  className="stroke-border/70 stroke-1"
                />
              </g>
            );
          })}

          {/* Message arrows */}
          {steps.map((st, i) => {
            const y = 60 + i * rowHeight;
            const x1 = getParticipantX(st.sender);
            const x2 = getParticipantX(st.receiver);
            const color = getPerformativeColor(st.performative);

            return (
              <g key={st.seq}>
                {/* Arrow line */}
                <line
                  x1={x1}
                  y1={y}
                  x2={x2 > x1 ? x2 - 5 : x2 + 5}
                  y2={y}
                  stroke={color}
                  strokeWidth="2"
                  markerEnd="url(#arrowhead)"
                />
                {/* Message label */}
                <text
                  x={(x1 + x2) / 2}
                  y={y - 6}
                  textAnchor="middle"
                  fill={color}
                  className="font-mono text-[10px] font-bold"
                >
                  {st.performative}: {st.summary}
                </text>
                {/* Tick dot */}
                <circle cx={x1} cy={y} r="3" fill={color} />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
