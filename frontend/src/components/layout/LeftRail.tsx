import {
  BookOpen,
  Cpu,
  ExternalLink,
  Flame,
  LayoutDashboard,
  LineChart,
  Network,
  Sparkles,
  Wrench,
} from "lucide-react";
import React from "react";
import { Link, useLocation } from "react-router-dom";
import { useLiveStore } from "../../store/liveStore";
import { useUiStore } from "../../store/uiStore";

export const LeftRail: React.FC = () => {
  const location = useLocation();
  const presentMode = useUiStore((s) => s.presentMode);
  const snapshot = useLiveStore((s) => s.snapshot);

  if (presentMode || location.pathname === "/story") {
    return null;
  }

  const navItems = [
    { label: "Mission Control", path: "/", icon: LayoutDashboard },
    { label: "Agents", path: "/agents", icon: Network },
    { label: "Algorithm Lab", path: "/lab", icon: Cpu },
    { label: "Experiments", path: "/experiments", icon: LineChart },
    { label: "Theory", path: "/theory", icon: BookOpen },
    { label: "Story Mode", path: "/story", icon: Sparkles },
  ];

  const hasFault = snapshot?.cars.some((c) => c.out_of_service);
  const hasFire = snapshot?.fire_alarm;

  return (
    <aside
      aria-label="Navigation Rail"
      className="flex h-screen w-16 md:w-56 shrink-0 flex-col justify-between border-r border-border bg-card transition-all"
    >
      <div>
        {/* Brand */}
        <div className="flex h-14 items-center gap-3 border-b border-border px-4">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary font-mono text-base font-bold text-primary-foreground shadow-sm">
            LZ
          </div>
          <div className="hidden flex-col md:flex">
            <span className="text-sm font-bold tracking-tight text-foreground">LiftZero</span>
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Control Room
            </span>
          </div>
        </div>

        {/* Links */}
        <nav className="space-y-1 p-2" aria-label="Main Navigation">
          {navItems.map((item) => {
            const isActive =
              item.path === "/"
                ? location.pathname === "/"
                : location.pathname.startsWith(item.path);
            const Icon = item.icon;

            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-primary/15 text-primary font-semibold shadow-xs"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
                title={item.label}
              >
                <Icon className={`h-4 w-4 shrink-0 ${isActive ? "text-primary" : ""}`} />
                <span className="hidden md:inline truncate">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Fleet Alert Status & Classic fallback */}
      <div className="border-t border-border p-3 space-y-2">
        {(hasFire || hasFault) && (
          <div className="hidden md:flex flex-col gap-1 rounded-lg border border-destructive/30 bg-destructive/10 p-2 text-xs">
            {hasFire && (
              <span className="flex items-center gap-1.5 font-bold text-destructive animate-pulse">
                <Flame className="h-3.5 w-3.5" /> FIRE ALARM
              </span>
            )}
            {hasFault && (
              <span className="flex items-center gap-1.5 font-medium text-amber-500">
                <Wrench className="h-3.5 w-3.5" /> CAR FAULT
              </span>
            )}
          </div>
        )}

        <a
          href="/classic"
          className="flex items-center justify-between rounded-lg px-3 py-2 text-xs text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          title="Open Classic Dashboard"
        >
          <span className="hidden md:inline">Classic View</span>
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      </div>
    </aside>
  );
};
