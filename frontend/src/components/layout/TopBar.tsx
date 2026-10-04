import { useQuery } from "@tanstack/react-query";
import {
  HelpCircle,
  Moon,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Sun,
  Zap,
} from "lucide-react";
import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { formatTime } from "../../lib/format";
import { useLiveStore } from "../../store/liveStore";
import { useUiStore } from "../../store/uiStore";

export const TopBar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const snapshot = useLiveStore((s) => s.snapshot);
  const connected = useLiveStore((s) => s.connected);
  const connecting = useLiveStore((s) => s.connecting);

  const theme = useUiStore((s) => s.theme);
  const toggleTheme = useUiStore((s) => s.toggleTheme);
  const setShortcutsOpen = useUiStore((s) => s.setShortcutsDialogOpen);
  const presentMode = useUiStore((s) => s.presentMode);

  const { data: meta } = useQuery({
    queryKey: ["meta"],
    queryFn: () => api.getMeta(),
    staleTime: 60_000,
  });

  const [scenario, setScenario] = useState("demo_story");
  const [strategy, setStrategy] = useState("full");
  const [seed, setSeed] = useState(42);
  const [speed, setSpeed] = useState(1.0);
  const [isActing, setIsActing] = useState(false);

  if (presentMode || location.pathname === "/story") {
    return null;
  }

  const isRunning = snapshot?.session.running ?? false;
  const currentTick = snapshot?.tick ?? 0;
  const currentScenario = snapshot?.scenario ?? scenario;
  const currentStrategy = snapshot?.strategy ?? strategy;

  const handlePlayPause = async () => {
    setIsActing(true);
    try {
      if (isRunning) {
        await api.pause();
      } else {
        await api.play();
      }
    } finally {
      setIsActing(false);
    }
  };

  const handleStep = async (ticks: number) => {
    setIsActing(true);
    try {
      await api.step(ticks);
    } finally {
      setIsActing(false);
    }
  };

  const handleSpeedChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setSpeed(val);
    try {
      await api.setSpeed(val);
    } catch {
      // ignore
    }
  };

  const handleReset = async () => {
    setIsActing(true);
    try {
      await api.reset({ scenario, strategy, seed });
    } finally {
      setIsActing(false);
    }
  };

  return (
    <header
      aria-label="Simulation Controls Top Bar"
      className="flex h-14 w-full shrink-0 items-center justify-between border-b border-border bg-card px-4 text-sm"
    >
      {/* Left controls: Scenario, Strategy, Seed */}
      <div className="flex items-center gap-2 overflow-x-auto">
        {/* Scenario Select */}
        <div className="flex items-center gap-1.5">
          <label htmlFor="scenario-select" className="text-xs font-medium text-muted-foreground">
            Scenario:
          </label>
          <select
            id="scenario-select"
            value={currentScenario}
            onChange={(e) => {
              setScenario(e.target.value);
              api.reset({ scenario: e.target.value, strategy, seed });
            }}
            className="rounded-md border border-border bg-background px-2 py-1 text-xs font-semibold text-foreground focus:ring-1 focus:ring-primary"
          >
            {meta?.scenarios.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            )) || <option value="demo_story">demo_story</option>}
          </select>
        </div>

        {/* Strategy Select */}
        <div className="flex items-center gap-1.5">
          <label htmlFor="strategy-select" className="text-xs font-medium text-muted-foreground">
            Strategy:
          </label>
          <select
            id="strategy-select"
            value={currentStrategy}
            onChange={(e) => {
              setStrategy(e.target.value);
              api.reset({ scenario: currentScenario, strategy: e.target.value, seed });
            }}
            className="rounded-md border border-border bg-background px-2 py-1 text-xs font-semibold text-foreground focus:ring-1 focus:ring-primary"
          >
            {meta?.strategies.map((st) => (
              <option key={st.name} value={st.name}>
                {st.name}
              </option>
            )) || <option value="full">full</option>}
          </select>
        </div>

        {/* Seed Input */}
        <div className="flex items-center gap-1.5">
          <label htmlFor="seed-input" className="text-xs font-medium text-muted-foreground">
            Seed:
          </label>
          <input
            id="seed-input"
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || 1)}
            onBlur={() => api.reset({ scenario: currentScenario, strategy: currentStrategy, seed })}
            className="w-16 rounded-md border border-border bg-background px-2 py-1 font-mono text-xs text-foreground focus:ring-1 focus:ring-primary"
          />
        </div>
      </div>

      {/* Middle controls: Play, Pause, Step, Speed, Reset */}
      <div className="flex items-center gap-3">
        <button
          onClick={handlePlayPause}
          disabled={isActing}
          aria-label={isRunning ? "Pause simulation" : "Play simulation"}
          className={`flex h-8 items-center gap-1.5 rounded-lg px-3 text-xs font-bold transition-colors ${
            isRunning
              ? "bg-amber-500/20 text-amber-500 hover:bg-amber-500/30"
              : "bg-primary text-primary-foreground hover:bg-primary/90 shadow-xs"
          }`}
        >
          {isRunning ? (
            <>
              <Pause className="h-3.5 w-3.5 fill-current" /> Pause
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 fill-current" /> Play
            </>
          )}
        </button>

        {/* Step buttons */}
        <div className="flex items-center rounded-lg border border-border bg-background p-0.5">
          <button
            onClick={() => handleStep(1)}
            disabled={isActing || isRunning}
            title="Step 1 tick (→)"
            className="rounded px-2 py-1 text-xs font-medium hover:bg-muted disabled:opacity-40"
          >
            +1
          </button>
          <button
            onClick={() => handleStep(10)}
            disabled={isActing || isRunning}
            title="Step 10 ticks (Shift+→)"
            className="rounded px-2 py-1 text-xs font-medium hover:bg-muted disabled:opacity-40"
          >
            +10
          </button>
          <button
            onClick={() => handleStep(60)}
            disabled={isActing || isRunning}
            title="Step 60 ticks"
            className="rounded px-2 py-1 text-xs font-medium hover:bg-muted disabled:opacity-40"
          >
            +60
          </button>
        </div>

        {/* Speed slider */}
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Zap className="h-3.5 w-3.5 text-amber-500" />
          <input
            type="range"
            min="0.1"
            max="50"
            step="0.5"
            value={snapshot?.session.speed ?? speed}
            onChange={handleSpeedChange}
            aria-label="Playback speed"
            className="h-1.5 w-20 cursor-pointer accent-primary"
          />
          <span className="w-8 font-mono text-[11px] font-semibold text-foreground">
            {(snapshot?.session.speed ?? speed).toFixed(1)}x
          </span>
        </div>

        {/* Reset */}
        <button
          onClick={handleReset}
          disabled={isActing}
          title="Reset simulation (R)"
          className="flex h-8 items-center gap-1 rounded-lg border border-border bg-secondary px-2.5 text-xs font-medium text-secondary-foreground hover:bg-secondary/80 disabled:opacity-40"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Reset</span>
        </button>
      </div>

      {/* Right section: Ticks, Connection, Theme, Shortcuts, Present */}
      <div className="flex items-center gap-3">
        {/* Tick & Time Counter */}
        <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-2.5 py-1">
          <div className="flex items-center gap-1 font-mono text-xs">
            <span className="font-bold text-foreground">{formatTime(currentTick)}</span>
            <span className="text-muted-foreground">({currentTick}s)</span>
          </div>
        </div>

        {/* WS Connection Dot */}
        <div
          title={
            connected
              ? "WebSocket Connected (Streaming live snapshots)"
              : connecting
                ? "Connecting..."
                : "Disconnected"
          }
          className="flex items-center gap-1.5"
        >
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              connected
                ? "bg-emerald-500 shadow-xs shadow-emerald-500/50"
                : connecting
                  ? "bg-amber-500 animate-ping"
                  : "bg-rose-500"
            }`}
          />
        </div>

        {/* Present mode button */}
        <button
          onClick={() => navigate("/story")}
          title="Present Mode (P)"
          className="flex items-center gap-1.5 rounded-lg border border-primary/30 bg-primary/10 px-2.5 py-1 text-xs font-semibold text-primary hover:bg-primary/20 transition-colors"
        >
          <Sparkles className="h-3.5 w-3.5" />
          <span>Present</span>
        </button>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          title="Toggle Theme (T)"
          aria-label="Toggle theme"
          className="rounded-lg border border-border p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>

        {/* Keyboard Shortcuts */}
        <button
          onClick={() => setShortcutsOpen(true)}
          title="Keyboard shortcuts (?)"
          aria-label="Open keyboard shortcuts"
          className="rounded-lg border border-border p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
        >
          <HelpCircle className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
};
