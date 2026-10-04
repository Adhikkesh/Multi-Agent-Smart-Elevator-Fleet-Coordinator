import { beforeEach, describe, expect, it } from "vitest";
import type { Auction, Message, Metrics, Snapshot } from "../api/types";
import { useLiveStore } from "./liveStore";

const dummyMetrics: Metrics = {
  tick: 1,
  arrived: 10,
  delivered: 5,
  waiting: 3,
  riding: 2,
  avg_wait: 12.0,
  p95_wait: 20.0,
  max_wait: 25.0,
  avg_ride: 15.0,
  avg_system: 27.0,
  long_wait_pct: 0.0,
  throughput: 150.0,
  energy: 50.0,
  floors_travelled: 40,
  stops: 10,
  reversals: 0,
  messages: 60,
  messages_per_call: 6.0,
  calls: 10,
  nodes_expanded: 50,
  replans: 10,
  compute_ms_per_tick: 0.2,
  rules_fired: 0,
};

const dummySnapshot: Snapshot = {
  tick: 1,
  scenario: "demo_story",
  strategy: "full",
  seed: 42,
  building: { floors: 15, cars: 4, capacity: 10, lobby: 0 },
  cars: [],
  floors: [],
  metrics: dummyMetrics,
  auction: null,
  messages: [],
  rules: [],
  traffic: {
    pattern: "interfloor",
    reason: "normal",
    observed: 10,
    true_pattern: "interfloor",
    weights: { wait: 1.0, ride: 0.5, crowding: 0.3, energy: 0.2 },
  },
  events: [],
  fire_alarm: false,
  reassignment: null,
  parking: {},
  session: { running: false, speed: 1.0, duration: 330 },
};

describe("liveStore", () => {
  beforeEach(() => {
    useLiveStore.setState({
      snapshot: null,
      connected: false,
      connecting: false,
      metricsHistory: [],
      recentMessages: [],
      auctionHistory: [],
      selectedAuctionIndex: null,
    });
  });

  it("sets connection states correctly", () => {
    useLiveStore.getState().setConnecting(true);
    expect(useLiveStore.getState().connecting).toBe(true);

    useLiveStore.getState().setConnected(true);
    expect(useLiveStore.getState().connected).toBe(true);
    expect(useLiveStore.getState().connecting).toBe(false);
  });

  it("updates snapshot and maintains metrics ring buffer <= 600", () => {
    for (let i = 1; i <= 650; i++) {
      useLiveStore.getState().updateSnapshot({
        ...dummySnapshot,
        tick: i,
        metrics: { ...dummyMetrics, tick: i },
      });
    }

    const state = useLiveStore.getState();
    expect(state.snapshot?.tick).toBe(650);
    expect(state.metricsHistory.length).toBe(600);
    expect(state.metricsHistory[0]?.tick).toBe(51);
    expect(state.metricsHistory[599]?.tick).toBe(650);
  });

  it("appends and deduplicates messages by seq", () => {
    const msg1: Message = {
      seq: 1,
      tick: 1,
      performative: "REQUEST",
      sender: "floor-0",
      receiver: "dispatcher",
      conversation_id: "c1",
      content: {},
    };
    const msg2: Message = {
      seq: 2,
      tick: 2,
      performative: "CFP",
      sender: "dispatcher",
      receiver: "car-0",
      conversation_id: "c1",
      content: {},
    };

    useLiveStore.getState().updateSnapshot({
      ...dummySnapshot,
      messages: [msg1],
    });

    useLiveStore.getState().updateSnapshot({
      ...dummySnapshot,
      messages: [msg1, msg2], // msg1 is duplicate
    });

    const state = useLiveStore.getState();
    expect(state.recentMessages.length).toBe(2);
    expect(state.recentMessages.map((m) => m.seq)).toEqual([1, 2]);
  });

  it("deduplicates auction history by conversation_id", () => {
    const auction1: Auction = {
      tick: 10,
      conversation_id: "c10",
      floor: 3,
      direction: "UP",
      bids: [],
      winner: 1,
      reason: "car 1 won",
    };
    const auction1Update: Auction = {
      ...auction1,
      reason: "car 1 won updated",
    };

    useLiveStore.getState().updateSnapshot({
      ...dummySnapshot,
      auction: auction1,
    });

    useLiveStore.getState().updateSnapshot({
      ...dummySnapshot,
      auction: auction1Update,
    });

    const state = useLiveStore.getState();
    expect(state.auctionHistory.length).toBe(1);
    expect(state.auctionHistory[0]?.reason).toBe("car 1 won updated");
  });

  it("allows selecting an auction by index", () => {
    useLiveStore.getState().selectAuction(3);
    expect(useLiveStore.getState().selectedAuctionIndex).toBe(3);
    useLiveStore.getState().selectAuction(null);
    expect(useLiveStore.getState().selectedAuctionIndex).toBe(null);
  });
});
