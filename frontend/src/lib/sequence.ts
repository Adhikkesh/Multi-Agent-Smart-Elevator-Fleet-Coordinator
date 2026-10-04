import type { Message, Performative } from "../api/types";

export interface SequenceStep {
  seq: number;
  tick: number;
  sender: string;
  receiver: string;
  performative: Performative;
  summary: string;
  content: any;
}

export interface ConversationSequence {
  conversationId: string;
  participants: string[];
  steps: SequenceStep[];
}

export function buildSequenceDiagram(
  messages: Message[],
  conversationId: string,
): ConversationSequence {
  const matching = messages.filter((m) => m.conversation_id === conversationId);
  const participantsSet = new Set<string>();

  const steps: SequenceStep[] = matching.map((m) => {
    participantsSet.add(m.sender);
    participantsSet.add(m.receiver);

    let summary = "";
    if (m.performative === "REQUEST") {
      if (m.content?.order) {
        summary = `ORDER: ${m.content.order}`;
      } else if (m.content?.floor !== undefined) {
        summary = `HallCall: floor ${m.content.floor} ${m.content.direction ?? ""}`;
      }
    } else if (m.performative === "CFP") {
      summary = `CFP: floor ${m.content?.floor} ${m.content?.direction ?? ""}`;
    } else if (m.performative === "PROPOSE") {
      summary = `Bid: cost ${m.content?.total ?? "—"} (ETA ${m.content?.eta ?? "—"}s)`;
    } else if (m.performative === "REFUSE") {
      summary = `Refused: ${m.content?.reason ?? "busy"}`;
    } else if (m.performative === "ACCEPT_PROPOSAL") {
      summary = `Awarded call: ${m.content?.floor ?? ""} ${m.content?.direction ?? ""}`;
    } else if (m.performative === "REJECT_PROPOSAL") {
      summary = "Proposal rejected";
    } else if (m.performative === "INFORM") {
      if (m.content?.eta !== undefined) {
        summary = `ETA update: ${m.content.eta}s`;
      } else {
        summary = "Status update";
      }
    } else if (m.performative === "CANCEL") {
      summary = `Cancelled: ${m.content?.reason ?? "reassigned"}`;
    } else {
      summary = m.performative;
    }

    return {
      seq: m.seq,
      tick: m.tick,
      sender: m.sender,
      receiver: m.receiver,
      performative: m.performative,
      summary,
      content: m.content,
    };
  });

  return {
    conversationId,
    participants: Array.from(participantsSet).sort(),
    steps,
  };
}
