import { create } from "zustand";
import type { Auction, Message, Metrics, Snapshot } from "../api/types";

const MAX_METRICS_HISTORY = 600;
const MAX_MESSAGES_HISTORY = 250;
const MAX_AUCTION_HISTORY = 100;

interface LiveState {
  snapshot: Snapshot | null;
  connected: boolean;
  connecting: boolean;
  metricsHistory: Metrics[];
  recentMessages: Message[];
  auctionHistory: Auction[];
  selectedAuctionIndex: number | null;

  setConnected: (connected: boolean) => void;
  setConnecting: (connecting: boolean) => void;
  updateSnapshot: (snapshot: Snapshot) => void;
  selectAuction: (index: number | null) => void;
  setAuctionHistory: (auctions: Auction[]) => void;
  setMessages: (messages: Message[]) => void;
}

export const useLiveStore = create<LiveState>((set) => ({
  snapshot: null,
  connected: false,
  connecting: false,
  metricsHistory: [],
  recentMessages: [],
  auctionHistory: [],
  selectedAuctionIndex: null,

  setConnected: (connected) => set({ connected, connecting: false }),
  setConnecting: (connecting) => set({ connecting }),

  updateSnapshot: (nextSnapshot) =>
    set((state) => {
      // 1. Metrics ring buffer
      const newMetricsHistory = [...state.metricsHistory, nextSnapshot.metrics];
      if (newMetricsHistory.length > MAX_METRICS_HISTORY) {
        newMetricsHistory.splice(0, newMetricsHistory.length - MAX_METRICS_HISTORY);
      }

      // 2. Messages ring buffer (dedup by seq)
      const existingSeqs = new Set(state.recentMessages.map((m) => m.seq));
      const incomingNewMessages = nextSnapshot.messages.filter((m) => !existingSeqs.has(m.seq));
      let newMessages = [...state.recentMessages, ...incomingNewMessages];
      if (newMessages.length > MAX_MESSAGES_HISTORY) {
        newMessages = newMessages.slice(newMessages.length - MAX_MESSAGES_HISTORY);
      }

      // 3. Auction history ring buffer (dedup by conversation_id)
      const newAuctionHistory = [...state.auctionHistory];
      if (nextSnapshot.auction) {
        const existingIdx = newAuctionHistory.findIndex(
          (a) => a.conversation_id === nextSnapshot.auction!.conversation_id,
        );
        if (existingIdx >= 0) {
          newAuctionHistory[existingIdx] = nextSnapshot.auction;
        } else {
          newAuctionHistory.push(nextSnapshot.auction);
          if (newAuctionHistory.length > MAX_AUCTION_HISTORY) {
            newAuctionHistory.shift();
          }
        }
      }

      return {
        snapshot: nextSnapshot,
        metricsHistory: newMetricsHistory,
        recentMessages: newMessages,
        auctionHistory: newAuctionHistory,
      };
    }),

  selectAuction: (index) => set({ selectedAuctionIndex: index }),

  setAuctionHistory: (auctions) => set({ auctionHistory: auctions.slice(-MAX_AUCTION_HISTORY) }),

  setMessages: (messages) => set({ recentMessages: messages.slice(-MAX_MESSAGES_HISTORY) }),
}));
