import { MessageSquare, Pause, X } from "lucide-react";
import React, { useState } from "react";
import type { Message, Performative } from "../../api/types";
import { useLiveStore } from "../../store/liveStore";
import { MessageChip } from "./MessageChip";

const ALL_PERFORMATIVES: Performative[] = [
  "REQUEST",
  "CFP",
  "PROPOSE",
  "REFUSE",
  "ACCEPT_PROPOSAL",
  "REJECT_PROPOSAL",
  "INFORM",
  "CANCEL",
  "FAILURE",
];

export const MessageStream: React.FC = () => {
  const recentMessages = useLiveStore((s) => s.recentMessages);
  const [isPaused, setIsPaused] = useState(false);
  const [filterPerf, setFilterPerf] = useState<string>("ALL");
  const [selectedMsg, setSelectedMsg] = useState<Message | null>(null);

  // Messages are stored newest last; display descending (newest at top)
  const displayMessages = recentMessages.slice().reverse();

  const filtered = displayMessages.filter((m) => {
    if (filterPerf !== "ALL" && m.performative !== filterPerf) return false;
    return true;
  });

  return (
    <div
      data-testid="message-stream"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
      className="flex h-full flex-col rounded-xl border border-border bg-card p-4 shadow-sm"
    >
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">Live Message Bus (FIPA-ACL)</h2>
          {isPaused && (
            <span className="flex items-center gap-1 rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-500">
              <Pause className="h-2.5 w-2.5" /> Paused on hover
            </span>
          )}
        </div>

        {/* Filter select */}
        <div className="flex items-center gap-2">
          <select
            aria-label="Filter messages by performative"
            value={filterPerf}
            onChange={(e) => setFilterPerf(e.target.value)}
            className="rounded border border-border bg-background px-2 py-0.5 text-xs text-foreground"
          >
            <option value="ALL">All Performatives</option>
            {ALL_PERFORMATIVES.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Virtualized/Scrollable message list */}
      <div className="flex-1 space-y-1.5 overflow-y-auto pr-1">
        {filtered.length > 0 ? (
          filtered.slice(0, 50).map((msg) => (
            <MessageChip
              key={msg.seq}
              message={msg}
              onClick={() => setSelectedMsg(msg)}
            />
          ))
        ) : (
          <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
            No messages matching filter
          </div>
        )}
      </div>

      {/* Detail Modal if clicked */}
      {selectedMsg && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4"
        >
          <div className="w-full max-w-lg rounded-xl border border-border bg-card p-6 shadow-2xl">
            <div className="mb-4 flex items-center justify-between border-b border-border pb-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-bold text-primary">
                  Message #{selectedMsg.seq} · {selectedMsg.performative}
                </span>
                <span className="text-xs text-muted-foreground">t={selectedMsg.tick}s</span>
              </div>
              <button
                onClick={() => setSelectedMsg(null)}
                className="rounded p-1 hover:bg-muted"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <span className="text-muted-foreground">Sender:</span>{" "}
                  <span className="font-mono font-semibold">{selectedMsg.sender}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Receiver:</span>{" "}
                  <span className="font-mono font-semibold">{selectedMsg.receiver}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Conversation:</span>{" "}
                  <span className="font-mono font-semibold">#{selectedMsg.conversation_id}</span>
                </div>
              </div>

              <div>
                <span className="text-muted-foreground">Payload Content:</span>
                <pre className="mt-1 max-h-60 overflow-auto rounded bg-muted/50 p-2.5 font-mono text-[11px] text-foreground">
                  {JSON.stringify(selectedMsg.content, null, 2)}
                </pre>
              </div>
            </div>

            <div className="mt-5 flex justify-end">
              <button
                onClick={() => setSelectedMsg(null)}
                className="rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
