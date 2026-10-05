import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { useEffect } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api } from "./api/client";
import { ErrorBoundary } from "./components/layout/ErrorBoundary";
import { KeyboardShortcutsDialog } from "./components/layout/KeyboardShortcutsDialog";
import { LeftRail } from "./components/layout/LeftRail";
import { TopBar } from "./components/layout/TopBar";
import { AgentInspector } from "./components/viz/AgentInspector";
import { AgentsPage } from "./features/agents/AgentsPage";
import { ExperimentsPage } from "./features/experiments/ExperimentsPage";
import { LabPage } from "./features/lab/LabPage";
import { MissionControlPage } from "./features/mission-control/MissionControlPage";
import { StoryPage } from "./features/story/StoryPage";
import { TheoryPage } from "./features/theory/TheoryPage";
import { BrainPage } from "./features/brain/BrainPage";
import { connectWebSocket, disconnectWebSocket } from "./live/socket";
import { useUiStore } from "./store/uiStore";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const AppShell: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const theme = useUiStore((s) => s.theme);
  const toggleTheme = useUiStore((s) => s.toggleTheme);
  const setShortcutsOpen = useUiStore((s) => s.setShortcutsDialogOpen);

  // Apply theme class
  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  // Connect live WebSocket on mount
  useEffect(() => {
    connectWebSocket();
    return () => {
      disconnectWebSocket();
    };
  }, []);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts if user is typing in an input or textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      if (e.key === " " && location.pathname !== "/story") {
        e.preventDefault();
        api.getState().then((st) => {
          if (st.session.running) api.pause();
          else api.play();
        });
      } else if (e.key === "ArrowRight" && location.pathname !== "/story") {
        e.preventDefault();
        if (e.shiftKey) {
          api.step(10);
        } else {
          api.step(1);
        }
      } else if ((e.key === "r" || e.key === "R") && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        api.reset({});
      } else if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        api.inject({ kind: "fire_alarm" });
      } else if ((e.key === "b" || e.key === "B") && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        api.inject({ kind: "car_fault", car: 1 });
      } else if (e.key === "1") {
        navigate("/");
      } else if (e.key === "2") {
        navigate("/agents");
      } else if (e.key === "3") {
        navigate("/lab");
      } else if (e.key === "4") {
        navigate("/experiments");
      } else if (e.key === "5") {
        navigate("/theory");
      } else if (e.key === "6") {
        navigate("/brain");
      } else if ((e.key === "p" || e.key === "P") && !e.ctrlKey && !e.metaKey) {
        navigate("/story");
      } else if ((e.key === "t" || e.key === "T") && !e.ctrlKey && !e.metaKey) {
        toggleTheme();
      } else if (e.key === "?") {
        setShortcutsOpen(true);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [location.pathname, navigate, setShortcutsOpen, toggleTheme]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Persistent Left Rail */}
      <LeftRail />

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Persistent Top Bar */}
        <TopBar />

        {/* Dynamic Pages */}
        <main className="flex-1 overflow-hidden" role="main">
          <Routes>
            <Route path="/" element={<MissionControlPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/lab" element={<LabPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/experiments/compare" element={<ExperimentsPage />} />
            <Route path="/theory" element={<TheoryPage />} />
            <Route path="/brain" element={<BrainPage />} />
            <Route path="/story" element={<StoryPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>

      {/* Keyboard Shortcuts Dialog */}
      <KeyboardShortcutsDialog />

      {/* Global Agent Inspector Drawer */}
      <AgentInspector />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <AppShell />
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};
