import { Search } from "lucide-react";
import React, { useState } from "react";
import { AgentGraph } from "../../components/viz/AgentGraph";
import { SequenceDiagram } from "../../components/viz/SequenceDiagram";
import { StatusBoardPanel } from "../../components/viz/StatusBoardPanel";
import { getPerformativeColor } from "../../lib/colors";
import { useLiveStore } from "../../store/liveStore";
import { useUiStore } from "../../store/uiStore";

export const AgentsPage: React.FC = () => {
  const recentMessages = useLiveStore((s) => s.recentMessages);
  const openInspector = useUiStore((s) => s.openInspector);
  const selectConv = useUiStore((s) => s.selectConversation);

  const [searchQuery, setSearchQuery] = useState("");
  const [filterPerf, setFilterPerf] = useState("ALL");

  const filteredMessages = recentMessages.filter((m) => {
    if (filterPerf !== "ALL" && m.performative !== filterPerf) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        m.sender.toLowerCase().includes(q) ||
        m.receiver.toLowerCase().includes(q) ||
        m.conversation_id.toLowerCase().includes(q) ||
        JSON.stringify(m.content).toLowerCase().includes(q)
      );
    }
    return true;
  });

  return (
    <div className="flex h-full flex-col overflow-y-auto p-4 space-y-4">
      {/* 1. Interactive Agent Graph */}
      <AgentGraph />

      {/* 2. Sequence Diagram of selected / latest conversation */}
      <SequenceDiagram />

      {/* 3. Shared Blackboard Status Board Panel */}
      <StatusBoardPanel />

      {/* 4. Full Searchable & Filterable Message Log */}
      <div className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2">
          <h3 className="text-sm font-semibold text-foreground">
            Complete FIPA-ACL Message Protocol Stream ({filteredMessages.length})
          </h3>

          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search messages..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="rounded-lg border border-border bg-background pl-8 pr-3 py-1 text-xs text-foreground placeholder:text-muted-foreground focus:ring-1 focus:ring-primary"
              />
            </div>

            <select
              aria-label="Filter messages by performative"
              value={filterPerf}
              onChange={(e) => setFilterPerf(e.target.value)}
              className="rounded-lg border border-border bg-background px-2 py-1 text-xs text-foreground font-medium"
            >
              <option value="ALL">All Performatives</option>
              <option value="REQUEST">REQUEST</option>
              <option value="CFP">CFP</option>
              <option value="PROPOSE">PROPOSE</option>
              <option value="REFUSE">REFUSE</option>
              <option value="ACCEPT_PROPOSAL">ACCEPT_PROPOSAL</option>
              <option value="REJECT_PROPOSAL">REJECT_PROPOSAL</option>
              <option value="INFORM">INFORM</option>
              <option value="CANCEL">CANCEL</option>
              <option value="FAILURE">FAILURE</option>
            </select>
          </div>
        </div>

        <div className="max-h-80 overflow-y-auto rounded-lg border border-border/60">
          <table className="w-full text-left text-xs font-mono">
            <thead className="sticky top-0 border-b border-border bg-muted/60 font-sans font-semibold text-foreground">
              <tr>
                <th className="p-2">Seq</th>
                <th className="p-2">Tick</th>
                <th className="p-2">Performative</th>
                <th className="p-2">Sender</th>
                <th className="p-2">Receiver</th>
                <th className="p-2">Conv ID</th>
                <th className="p-2">Payload Summary</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {filteredMessages.slice().reverse().map((msg) => (
                <tr
                  key={msg.seq}
                  onClick={() => selectConv(msg.conversation_id)}
                  className="cursor-pointer transition-colors hover:bg-muted/40"
                >
                  <td className="p-2 text-muted-foreground">#{msg.seq}</td>
                  <td className="p-2">t={msg.tick}s</td>
                  <td className="p-2">
                    <span
                      style={{ color: getPerformativeColor(msg.performative) }}
                      className="font-bold uppercase"
                    >
                      {msg.performative}
                    </span>
                  </td>
                  <td
                    onClick={(e) => {
                      e.stopPropagation();
                      openInspector(msg.sender);
                    }}
                    className="p-2 font-bold hover:underline"
                  >
                    {msg.sender}
                  </td>
                  <td
                    onClick={(e) => {
                      e.stopPropagation();
                      openInspector(msg.receiver);
                    }}
                    className="p-2 font-bold hover:underline"
                  >
                    {msg.receiver}
                  </td>
                  <td className="p-2 text-primary font-bold">#{msg.conversation_id}</td>
                  <td className="p-2 text-muted-foreground truncate max-w-xs font-sans">
                    {JSON.stringify(msg.content)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
