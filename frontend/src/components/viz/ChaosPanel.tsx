import { AlertOctagon, Bell, BellOff, Flame, UserPlus, Wrench } from "lucide-react";
import React, { useState } from "react";
import { api } from "../../api/client";
import { useLiveStore } from "../../store/liveStore";

export const ChaosPanel: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const cars = snapshot?.cars ?? [];
  const floors = snapshot?.floors ?? [];
  const events = snapshot?.events ?? [];

  const [selectedCar, setSelectedCar] = useState(0);
  const [selectedFloor, setSelectedFloor] = useState(0);
  const [rushCount, setRushCount] = useState(12);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleInject = async (params: {
    kind: string;
    car?: number;
    floor?: number;
    count?: number;
  }) => {
    setIsSubmitting(true);
    try {
      await api.inject(params);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      data-testid="chaos-panel"
      className="flex flex-col rounded-xl border border-border bg-card p-4 shadow-sm"
    >
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <AlertOctagon className="h-4 w-4 text-amber-500" />
          <h2 className="text-sm font-semibold text-foreground">Disturbance & Chaos Injection</h2>
        </div>
      </div>

      <div className="space-y-3 text-xs">
        {/* Row 1: Car Fault & Repair */}
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/60 bg-muted/20 p-2">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground font-medium">Car:</span>
            <select
              aria-label="Select Car to fault or repair"
              value={selectedCar}
              onChange={(e) => setSelectedCar(parseInt(e.target.value, 10))}
              className="rounded border border-border bg-background px-2 py-1 font-mono text-xs font-semibold"
            >
              {cars.map((c) => (
                <option key={c.car_id} value={c.car_id}>
                  Car {c.car_id} {c.out_of_service ? "(Faulted)" : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => handleInject({ kind: "car_fault", car: selectedCar })}
              disabled={isSubmitting}
              className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-600 hover:bg-amber-500/20 disabled:opacity-50"
            >
              <Wrench className="h-3 w-3" /> Break Car
            </button>
            <button
              onClick={() => handleInject({ kind: "car_repair", car: selectedCar })}
              disabled={isSubmitting}
              className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-600 hover:bg-emerald-500/20 disabled:opacity-50"
            >
              Repair Car
            </button>
          </div>
        </div>

        {/* Row 2: Fire Drill Controls */}
        <div className="flex items-center justify-between gap-2 rounded-lg border border-border/60 bg-muted/20 p-2">
          <span className="text-muted-foreground font-medium flex items-center gap-1.5">
            <Flame className="h-3.5 w-3.5 text-destructive" /> Emergency Alarm:
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleInject({ kind: "fire_alarm" })}
              disabled={isSubmitting}
              className="flex items-center gap-1.5 rounded-lg border border-destructive/40 bg-destructive/10 px-2.5 py-1 text-xs font-medium text-destructive hover:bg-destructive/20 disabled:opacity-50"
            >
              <Bell className="h-3 w-3" /> Fire Alarm
            </button>
            <button
              onClick={() => handleInject({ kind: "fire_clear" })}
              disabled={isSubmitting}
              className="flex items-center gap-1.5 rounded-lg border border-border bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground hover:bg-secondary/80 disabled:opacity-50"
            >
              <BellOff className="h-3 w-3" /> Clear Alarm
            </button>
          </div>
        </div>

        {/* Row 3: Passenger Rush */}
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/60 bg-muted/20 p-2">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground font-medium">Rush:</span>
            <select
              aria-label="Select floor for rush disturbance"
              value={selectedFloor}
              onChange={(e) => setSelectedFloor(parseInt(e.target.value, 10))}
              className="rounded border border-border bg-background px-2 py-1 font-mono text-xs font-semibold"
            >
              {floors.map((f) => (
                <option key={f.floor} value={f.floor}>
                  F{f.floor} {f.floor === 0 ? "(Lobby)" : ""}
                </option>
              ))}
            </select>
            <input
              aria-label="Passenger count for rush disturbance"
              type="number"
              min={1}
              max={50}
              value={rushCount}
              onChange={(e) => setRushCount(parseInt(e.target.value, 10) || 1)}
              className="w-14 rounded border border-border bg-background px-2 py-1 font-mono text-xs"
            />
            <span className="text-muted-foreground">riders</span>
          </div>

          <button
            onClick={() =>
              handleInject({ kind: "rush", floor: selectedFloor, count: rushCount })
            }
            disabled={isSubmitting}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            <UserPlus className="h-3 w-3" /> Inject Rush
          </button>
        </div>

        {/* Events history */}
        {events.length > 0 && (
          <div className="pt-2 border-t border-border/60">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
              Injected Events History:
            </span>
            <div className="mt-1 flex flex-wrap gap-1.5 max-h-16 overflow-y-auto">
              {events.slice(-6).map((ev, idx) => (
                <span
                  key={idx}
                  className="rounded bg-muted px-2 py-0.5 font-mono text-[10px] text-foreground"
                >
                  t={ev.tick}s · {ev.kind} {ev.floor !== undefined ? `F${ev.floor}` : ""}{" "}
                  {ev.count ? `(${ev.count}p)` : ""}{" "}
                  {ev.car !== undefined ? `(C${ev.car})` : ""}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
