import { create } from "zustand";

export interface StoryBeat {
  id: number;
  title: string;
  tagline: string;
  description: string;
  actionName?: string;
  conditionDescription?: string;
}

export const STORY_BEATS: StoryBeat[] = [
  {
    id: 1,
    title: "1. The Building & Architecture",
    tagline: "15 floors, 4 cars, autonomous multi-agent coordination",
    description:
      "Welcome to LiftZero. Each car is an autonomous agent with its own search planner. Floors act as sensor agents detecting waiting crowds, while a dispatcher coordinates auctions via Contract Net Protocol.",
  },
  {
    id: 2,
    title: "2. Calm Traffic & Strategic Parking",
    tagline: "Minimax adversarial positioning for zero-demand periods",
    description:
      "When cars become idle, the dispatcher uses minimax with alpha-beta pruning to park elevators near anticipated demand spots (e.g. lobby, mid-building, top cafeteria).",
    actionName: "Reset to calm scenario",
  },
  {
    id: 3,
    title: "3. Lobby Rush & Contract Net Auction",
    tagline: "High demand spawns FIPA-ACL CFP & PROPOSE message storm",
    description:
      "Injecting a passenger rush at the lobby triggers Call For Proposals (CFP). Every eligible car calculates its marginal cost using A* and submits bids simultaneously.",
    actionName: "Inject lobby rush",
    conditionDescription: "Wait for Contract Net auction round",
  },
  {
    id: 4,
    title: "4. The Decision Trace",
    tagline: "Explainable AI: multi-attribute utility function breakdown",
    description:
      "Why did the winning car get the call? The auction panel breaks down the 4 utility components: passenger wait time, onboard ride time, car crowding, and motor energy.",
  },
  {
    id: 5,
    title: "5. Traffic Monitor Learning",
    tagline: "Online pattern inference & adaptive cost re-weighting",
    description:
      "The TrafficMonitorAgent observes passenger arrival flows via EWMA. It infers peak patterns (e.g. up-peak, down-peak) and dynamically tunes the fleet's bidding weights.",
  },
  {
    id: 6,
    title: "6. Resilient Fault Handling",
    tagline: "Safety rule R4 fires; orphaned calls are re-auctioned",
    description:
      "A car breakdown is injected. The knowledge-based Safety Agent fires rule R4, immediately marks the car OUT_OF_SERVICE, and hands back all its commitments for instant re-auction.",
    actionName: "Break Car 1",
    conditionDescription: "Wait for R4 safety rule event",
  },
  {
    id: 7,
    title: "7. Fire Alarm Override",
    tagline: "Rule R1 preempts all goals with emergency ground recall",
    description:
      "In a fire drill, the Safety Agent issues emergency FIRE_RECALL orders. All normal traffic is cancelled, and all cars immediately express to the lobby and hold doors open.",
    actionName: "Inject fire alarm",
    conditionDescription: "Wait for fire alarm state",
  },
  {
    id: 8,
    title: "8. Normal Service Restored",
    tagline: "Rule R7 resets fleet status and resumes distributed dispatch",
    description:
      "Alarm cleared and repairs completed. The safety agent restores cars to normal service, rebuilding the active fleet without restarting the simulation.",
    actionName: "Repair cars & clear alarm",
  },
  {
    id: 9,
    title: "9. Benchmarking vs Baselines",
    tagline: "Full multi-agent coordination vs Nearest-Car reflex",
    description:
      "Comparing Contract Net + A* against conventional nearest-car reflex logic shows up to 30-40% reduction in average wait times and elimination of starved passengers.",
  },
  {
    id: 10,
    title: "10. What's Next: LiftZero Learned Bidder",
    tagline: "Phase 6 neural agent with MCTS tree search",
    description:
      "The classical heuristic bidder provides expert demonstrations to train a neural bidding policy, paving the way for full deep reinforcement learning fleet coordination.",
  },
];

interface StoryState {
  activeBeatIndex: number;
  isAutoPlaying: boolean;
  beatExecuting: boolean;

  setBeatIndex: (idx: number) => void;
  nextBeat: () => void;
  prevBeat: () => void;
  setAutoPlaying: (val: boolean) => void;
  setBeatExecuting: (val: boolean) => void;
  resetStory: () => void;
}

export const useStoryStore = create<StoryState>((set) => ({
  activeBeatIndex: 0,
  isAutoPlaying: false,
  beatExecuting: false,

  setBeatIndex: (idx) =>
    set({
      activeBeatIndex: Math.max(0, Math.min(STORY_BEATS.length - 1, idx)),
    }),

  nextBeat: () =>
    set((state) => ({
      activeBeatIndex: Math.min(STORY_BEATS.length - 1, state.activeBeatIndex + 1),
    })),

  prevBeat: () =>
    set((state) => ({
      activeBeatIndex: Math.max(0, state.activeBeatIndex - 1),
    })),

  setAutoPlaying: (isAutoPlaying) => set({ isAutoPlaying }),
  setBeatExecuting: (beatExecuting) => set({ beatExecuting }),
  resetStory: () => set({ activeBeatIndex: 0, isAutoPlaying: false, beatExecuting: false }),
}));
