import { ArrowRight, ShieldAlert } from "lucide-react";
import React from "react";
import type { Message } from "../../api/types";
import { getPerformativeColor } from "../../lib/colors";
import { useUiStore } from "../../store/uiStore";

interface MessageChipProps {
  message: Message;
  onClick?: () => void;
}

export const MessageChip: React.FC<MessageChipProps> = ({ message, onClick }) => {
  const selectedConv = useUiStore((s) => s.selectedConversationId);
  const selectConv = useUiStore((s) => s.selectConversation);

  const isHighlighted = selectedConv === message.conversation_id;
  const isSafetyOrder =
    message.performative === "REQUEST" &&
    typeof message.content === "object" &&
    message.content?.order;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => selectConv(message.conversation_id)}
      onMouseLeave={() => selectConv(null)}
      className={`group flex cursor-pointer items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 text-xs transition-colors ${
        isHighlighted
          ? "border-primary bg-primary/10 shadow-xs"
          : "border-border/60 bg-card hover:bg-muted/40"
      }`}
    >
      <div className="flex items-center gap-2 overflow-hidden">
        {/* Performative pill */}
        <span
          style={{
            borderColor: getPerformativeColor(message.performative),
            color: getPerformativeColor(message.performative),
          }}
          className="shrink-0 rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider"
        >
          {message.performative}
        </span>

        {/* Sender -> Receiver */}
        <div className="flex items-center gap-1 font-mono text-[11px] text-muted-foreground truncate">
          <span className="font-semibold text-foreground">{message.sender}</span>
          <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground/60" />
          <span className="font-semibold text-foreground">{message.receiver}</span>
        </div>

        {/* Safety Order Badge */}
        {isSafetyOrder && (
          <span className="flex items-center gap-1 rounded bg-rose-500/15 px-1.5 py-0.5 font-mono text-[10px] font-bold text-rose-500">
            <ShieldAlert className="h-3 w-3" /> ORDER · {message.content.order}
          </span>
        )}
      </div>

      {/* Right side: Conversation ID and Tick */}
      <div className="flex items-center gap-2 shrink-0 font-mono text-[10px] text-muted-foreground">
        <span
          className={`rounded px-1.5 py-0.5 transition-colors ${
            isHighlighted ? "bg-primary text-primary-foreground font-bold" : "bg-muted"
          }`}
        >
          #{message.conversation_id}
        </span>
        <span>t={message.tick}</span>
      </div>
    </div>
  );
};
