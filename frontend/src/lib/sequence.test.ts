import { describe, expect, it } from "vitest";
import type { Message } from "../api/types";
import { buildSequenceDiagram } from "./sequence";

describe("buildSequenceDiagram", () => {
  const sampleMessages: Message[] = [
    {
      seq: 1,
      tick: 10,
      performative: "REQUEST",
      sender: "floor-3",
      receiver: "dispatcher",
      conversation_id: "conv-1",
      content: { floor: 3, direction: "UP" },
    },
    {
      seq: 2,
      tick: 10,
      performative: "CFP",
      sender: "dispatcher",
      receiver: "car-0",
      conversation_id: "conv-1",
      content: { floor: 3, direction: "UP" },
    },
    {
      seq: 3,
      tick: 11,
      performative: "PROPOSE",
      sender: "car-0",
      receiver: "dispatcher",
      conversation_id: "conv-1",
      content: { total: 4.5, eta: 2.0 },
    },
    {
      seq: 4,
      tick: 11,
      performative: "ACCEPT_PROPOSAL",
      sender: "dispatcher",
      receiver: "car-0",
      conversation_id: "conv-1",
      content: { floor: 3, direction: "UP" },
    },
    {
      seq: 5,
      tick: 12,
      performative: "INFORM",
      sender: "car-0",
      receiver: "floor-3",
      conversation_id: "conv-1",
      content: { eta: 4.0 },
    },
    {
      seq: 6,
      tick: 15,
      performative: "CANCEL",
      sender: "dispatcher",
      receiver: "car-0",
      conversation_id: "conv-2",
      content: { reason: "global reassignment" },
    },
  ];

  it("filters and orders messages for the given conversation", () => {
    const seq = buildSequenceDiagram(sampleMessages, "conv-1");
    expect(seq.conversationId).toBe("conv-1");
    expect(seq.steps.length).toBe(5);
    expect(seq.participants).toEqual(["car-0", "dispatcher", "floor-3"]);
  });

  it("extracts meaningful summaries for performatives", () => {
    const seq = buildSequenceDiagram(sampleMessages, "conv-1");
    expect(seq.steps[0]?.summary).toContain("HallCall");
    expect(seq.steps[1]?.summary).toContain("CFP");
    expect(seq.steps[2]?.summary).toContain("Bid: cost 4.5");
    expect(seq.steps[3]?.summary).toContain("Awarded call");
    expect(seq.steps[4]?.summary).toContain("ETA update: 4s");
  });

  it("handles empty or nonexistent conversations", () => {
    const seq = buildSequenceDiagram(sampleMessages, "conv-none");
    expect(seq.steps.length).toBe(0);
    expect(seq.participants.length).toBe(0);
  });
});
