import { create } from "zustand";

interface UiState {
  theme: "dark" | "light";
  selectedCarId: number | null;
  selectedFloor: number | null;
  selectedConversationId: string | null;
  inspectorAgentAddress: string | null;
  inspectorOpen: boolean;
  shortcutsDialogOpen: boolean;
  presentMode: boolean;
  messageFilterPerf: string | null;
  messageFilterAgent: string | null;

  setTheme: (theme: "dark" | "light") => void;
  toggleTheme: () => void;
  selectCar: (carId: number | null) => void;
  selectFloor: (floor: number | null) => void;
  selectConversation: (convId: string | null) => void;
  openInspector: (address: string) => void;
  closeInspector: () => void;
  setShortcutsDialogOpen: (open: boolean) => void;
  setPresentMode: (present: boolean) => void;
  setMessageFilterPerf: (perf: string | null) => void;
  setMessageFilterAgent: (agent: string | null) => void;
}

function getInitialTheme(): "dark" | "light" {
  try {
    const saved = localStorage.getItem("liftzero-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // localStorage not accessible
  }
  return "dark";
}

export const useUiStore = create<UiState>((set) => ({
  theme: getInitialTheme(),
  selectedCarId: null,
  selectedFloor: null,
  selectedConversationId: null,
  inspectorAgentAddress: null,
  inspectorOpen: false,
  shortcutsDialogOpen: false,
  presentMode: false,
  messageFilterPerf: null,
  messageFilterAgent: null,

  setTheme: (theme) => {
    try {
      localStorage.setItem("liftzero-theme", theme);
    } catch {
      // ignore
    }
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    set({ theme });
  },

  toggleTheme: () =>
    set((state) => {
      const nextTheme = state.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem("liftzero-theme", nextTheme);
      } catch {
        // ignore
      }
      if (nextTheme === "dark") {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
      return { theme: nextTheme };
    }),

  selectCar: (selectedCarId) => set({ selectedCarId }),
  selectFloor: (selectedFloor) => set({ selectedFloor }),
  selectConversation: (selectedConversationId) => set({ selectedConversationId }),

  openInspector: (inspectorAgentAddress) =>
    set({ inspectorAgentAddress, inspectorOpen: true }),

  closeInspector: () => set({ inspectorOpen: false }),
  setShortcutsDialogOpen: (shortcutsDialogOpen) => set({ shortcutsDialogOpen }),
  setPresentMode: (presentMode) => set({ presentMode }),
  setMessageFilterPerf: (messageFilterPerf) => set({ messageFilterPerf }),
  setMessageFilterAgent: (messageFilterAgent) => set({ messageFilterAgent }),
}));
