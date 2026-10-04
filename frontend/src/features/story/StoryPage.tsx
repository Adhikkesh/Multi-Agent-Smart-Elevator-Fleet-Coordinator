import {
  ChevronLeft,
  ChevronRight,
  Play,
  Sparkles,
} from "lucide-react";
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { Building } from "../../components/viz/Building";
import { STORY_BEATS, useStoryStore } from "../../store/storyStore";
import { useUiStore } from "../../store/uiStore";

export const StoryPage: React.FC = () => {
  const navigate = useNavigate();
  const setPresentMode = useUiStore((s) => s.setPresentMode);

  const activeBeatIdx = useStoryStore((s) => s.activeBeatIndex);
  const nextBeat = useStoryStore((s) => s.nextBeat);
  const prevBeat = useStoryStore((s) => s.prevBeat);
  const setBeatIndex = useStoryStore((s) => s.setBeatIndex);

  const beat = STORY_BEATS[activeBeatIdx] || STORY_BEATS[0]!;

  const [isExecuting, setIsExecuting] = useState(false);

  // Esc key exits present mode
  useEffect(() => {
    setPresentMode(true);
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPresentMode(false);
        navigate("/");
      } else if (e.key === "ArrowRight") {
        nextBeat();
      } else if (e.key === "ArrowLeft") {
        prevBeat();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      setPresentMode(false);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [navigate, nextBeat, prevBeat, setPresentMode]);

  // Execute beat real actions
  const handleExecuteBeatAction = async () => {
    setIsExecuting(true);
    try {
      if (beat.id === 2) {
        // Reset to calm
        await api.reset({ scenario: "demo_story", strategy: "full", seed: 42 });
        await api.step(10);
      } else if (beat.id === 3) {
        // Inject lobby rush
        await api.inject({ kind: "rush", floor: 0, count: 12 });
        await api.step(5);
      } else if (beat.id === 6) {
        // Break car 1
        await api.inject({ kind: "car_fault", car: 1 });
        await api.step(5);
      } else if (beat.id === 7) {
        // Fire alarm
        await api.inject({ kind: "fire_alarm" });
        await api.step(5);
      } else if (beat.id === 8) {
        // Repair car & clear alarm
        await api.inject({ kind: "car_repair", car: 1 });
        await api.inject({ kind: "fire_clear" });
        await api.step(5);
      }
    } catch {
      // ignore
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex h-screen w-screen flex-col bg-background text-foreground select-none">
      {/* Top minimal header */}
      <header className="flex h-12 items-center justify-between border-b border-border/40 px-6">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          <span className="font-bold tracking-tight text-sm">
            LiftZero Presentation & Viva Story Walkthrough
          </span>
        </div>

        {/* Progress dots */}
        <div className="flex items-center gap-1.5">
          {STORY_BEATS.map((b, idx) => (
            <button
              key={b.id}
              onClick={() => setBeatIndex(idx)}
              className={`h-2.5 rounded-full transition-all ${
                activeBeatIdx === idx
                  ? "w-8 bg-primary"
                  : "w-2.5 bg-muted hover:bg-muted-foreground/40"
              }`}
              title={b.title}
            />
          ))}
        </div>

        {/* Exit Button */}
        <button
          onClick={() => {
            setPresentMode(false);
            navigate("/");
          }}
          className="flex items-center gap-1.5 rounded-lg border border-border bg-secondary px-3 py-1 text-xs font-semibold text-secondary-foreground hover:bg-secondary/80"
        >
          <span>Exit Presentation</span>
          <kbd className="rounded bg-muted px-1 font-mono text-[10px]">Esc</kbd>
        </button>
      </header>

      {/* Main split: Left Caption Card, Right Live Building */}
      <div className="flex flex-1 overflow-hidden p-6 gap-6">
        {/* Left Story Caption Card */}
        <div className="flex w-96 shrink-0 flex-col justify-between rounded-2xl border border-border bg-card p-6 shadow-xl">
          <div className="space-y-4">
            <div className="space-y-1">
              <span className="text-xs font-bold uppercase tracking-wider text-primary">
                Beat {activeBeatIdx + 1} of {STORY_BEATS.length}
              </span>
              <h2 className="text-2xl font-black text-foreground tracking-tight">{beat.title}</h2>
              <p className="text-sm font-semibold text-primary">{beat.tagline}</p>
            </div>

            <p className="text-sm text-muted-foreground leading-relaxed">{beat.description}</p>

            {/* Real action trigger */}
            {beat.actionName && (
              <div className="pt-2">
                <button
                  onClick={handleExecuteBeatAction}
                  disabled={isExecuting}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary py-2.5 text-xs font-bold text-primary-foreground shadow-sm hover:bg-primary/90 disabled:opacity-50"
                >
                  <Play className="h-3.5 w-3.5 fill-current" />
                  <span>{isExecuting ? "Executing..." : beat.actionName}</span>
                </button>
              </div>
            )}
          </div>

          {/* Navigation Controls */}
          <div className="flex items-center justify-between border-t border-border pt-4">
            <button
              onClick={prevBeat}
              disabled={activeBeatIdx === 0}
              className="flex items-center gap-1 rounded-lg border border-border bg-secondary px-3 py-1.5 text-xs font-semibold text-secondary-foreground hover:bg-secondary/80 disabled:opacity-40"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Back</span>
            </button>

            <button
              onClick={nextBeat}
              disabled={activeBeatIdx === STORY_BEATS.length - 1}
              className="flex items-center gap-1 rounded-lg bg-primary px-4 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-40 shadow-xs"
            >
              <span>Next Beat</span>
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Right Live Simulation Building */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <Building />
        </div>
      </div>
    </div>
  );
};
