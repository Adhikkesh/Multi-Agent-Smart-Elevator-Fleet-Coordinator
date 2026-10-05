import { Flame, UserPlus, Wrench } from "lucide-react";
import React, { useMemo } from "react";
import { api } from "../../api/client";
import type { CarState, FloorState } from "../../api/types";
import { getCarStateBorder, getDirectionIcon } from "../../lib/colors";
import { getInterpolatedCarFloor } from "../../lib/interpolate";
import { useLiveStore } from "../../store/liveStore";
import { useUiStore } from "../../store/uiStore";

export const Building: React.FC = () => {
  const snapshot = useLiveStore((s) => s.snapshot);
  const selectCar = useUiStore((s) => s.selectCar);
  const openInspector = useUiStore((s) => s.openInspector);
  const selectedCarId = useUiStore((s) => s.selectedCarId);

  const totalFloors = snapshot?.building.floors ?? 15;
  const cars = snapshot?.cars ?? [];
  const floors = snapshot?.floors ?? [];
  const parking = snapshot?.parking ?? {};

  // Floor array sorted from top down (highest floor at top, 0 at bottom)
  const floorList = useMemo(() => {
    const list: number[] = [];
    for (let f = totalFloors - 1; f >= 0; f--) {
      list.push(f);
    }
    return list;
  }, [totalFloors]);

  const floorHeightPx = totalFloors > 25 ? 24 : totalFloors > 15 ? 32 : 44;
  const buildingHeightPx = totalFloors * floorHeightPx;

  const handleFloorClick = async (floor: number, e: React.MouseEvent) => {
    const isPriority = e.shiftKey;
    try {
      await api.addPassenger({ origin: floor, priority: isPriority });
    } catch {
      // ignore
    }
  };

  const handleCarClick = (carId: number) => {
    selectCar(carId);
    openInspector(`car-${carId}`);
  };

  return (
    <div
      data-testid="building-view"
      className="flex h-full flex-col rounded-xl border border-border bg-card p-4 shadow-sm"
    >
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-foreground">Building & Shafts</h2>
          <span className="rounded-full bg-secondary px-2 py-0.5 text-[11px] font-medium text-secondary-foreground">
            {totalFloors} floors · {cars.length} cars
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <UserPlus className="h-3 w-3" /> Click floor to spawn (Shift+Click: priority)
          </span>
        </div>
      </div>

      {/* Building Container with scroll if needed */}
      <div className="relative flex-1 overflow-auto rounded-lg border border-border/60 bg-muted/20 p-2">
        <div
          className="relative flex justify-start gap-2"
          style={{ height: `${buildingHeightPx}px` }}
        >
          {/* Left Floor Gutter */}
          <div
            className="flex w-32 shrink-0 flex-col justify-between border-r border-border/80 pr-2 select-none"
            style={{ height: `${buildingHeightPx}px` }}
          >
            {floorList.map((floorNum) => {
              const floorData = floors.find((f) => f.floor === floorNum);
              const isLobby = floorNum === (snapshot?.building.lobby ?? 0);
              const hasEscalation = (floorData?.escalations ?? 0) > 0;
              const hasHighWait = (floorData?.oldest_wait ?? 0) > 30;

              return (
                <div
                  key={floorNum}
                  onClick={(e) => handleFloorClick(floorNum, e)}
                  title={`Floor ${floorNum}${isLobby ? " (Lobby)" : ""} · Click to add passenger`}
                  style={{ height: `${floorHeightPx}px` }}
                  className={`group relative flex cursor-pointer items-center justify-between rounded px-1.5 transition-colors hover:bg-primary/20 ${
                    hasHighWait ? "bg-amber-500/10" : ""
                  }`}
                >
                  {/* Floor number & Lobby Badge */}
                  <div className="flex items-center gap-1">
                    <span
                      className={`font-mono text-xs font-bold ${
                        isLobby ? "text-primary" : "text-foreground"
                      }`}
                    >
                      F{floorNum.toString().padStart(2, "0")}
                    </span>
                    {isLobby && (
                      <span className="rounded bg-primary/20 px-1 text-[9px] font-bold text-primary">
                        LOBBY
                      </span>
                    )}
                    {hasEscalation && (
                      <span title="Escalated call">
                        <Flame className="h-3 w-3 text-rose-500 animate-bounce" />
                      </span>
                    )}
                  </div>

                  {/* Hall Lanterns & Waiting crowds */}
                  <div className="flex items-center gap-1.5">
                    {/* Up button */}
                    <span
                      className={`text-[10px] font-bold ${
                        floorData?.up ? "text-amber-400 drop-shadow-xs" : "text-muted-foreground/30"
                      }`}
                      title={floorData?.up ? `UP call active (${floorData?.waiting_up} waiting)` : "No UP call"}
                    >
                      ▲
                    </span>
                    {/* Down button */}
                    <span
                      className={`text-[10px] font-bold ${
                        floorData?.down ? "text-amber-400 drop-shadow-xs" : "text-muted-foreground/30"
                      }`}
                      title={floorData?.down ? `DOWN call active (${floorData?.waiting_down} waiting)` : "No DOWN call"}
                    >
                      ▼
                    </span>

                    {/* Waiting crowd pips */}
                    <WaitingCrowd floorData={floorData} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Elevator Shafts */}
          <div className="relative flex flex-1 justify-around gap-2">
            {cars.map((car) => {
              const parkFloor = parking[car.car_id.toString()];
              return (
                <div
                  key={car.car_id}
                  className="relative flex h-full min-w-[64px] max-w-28 flex-1 flex-col items-center border-x border-dashed border-border/40 bg-background/50"
                  style={{ height: `${buildingHeightPx}px` }}
                >
                  {/* Floor grid horizontal dividers */}
                  {floorList.map((f) => (
                    <div
                      key={f}
                      className="w-full border-b border-border/20"
                      style={{ height: `${floorHeightPx}px` }}
                    />
                  ))}

                  {/* Ghost parking marker */}
                  {parkFloor !== undefined && (
                    <div
                      className="absolute z-10 flex w-20 items-center justify-center rounded border border-dashed border-primary/40 bg-primary/5 py-0.5 text-[9px] font-medium text-primary/70"
                      style={{
                        bottom: `${(parkFloor / totalFloors) * buildingHeightPx}px`,
                      }}
                      title={`Car ${car.car_id} strategic park target: floor ${parkFloor}`}
                    >
                      PARK F{parkFloor}
                    </div>
                  )}

                  {/* Interpolated Car Cabin */}
                  <CarCabin
                    car={car}
                    totalFloors={totalFloors}
                    buildingHeightPx={buildingHeightPx}
                    floorHeightPx={floorHeightPx}
                    isSelected={selectedCarId === car.car_id}
                    onClick={() => handleCarClick(car.car_id)}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

interface CarCabinProps {
  car: CarState;
  totalFloors: number;
  buildingHeightPx: number;
  floorHeightPx: number;
  isSelected: boolean;
  onClick: () => void;
}

const CarCabin: React.FC<CarCabinProps> = ({
  car,
  totalFloors,
  buildingHeightPx,
  floorHeightPx,
  isSelected,
  onClick,
}) => {
  const interpolatedFloor = getInterpolatedCarFloor(car.floor, car.direction, car.progress);
  // Bottom position: (interpolatedFloor / totalFloors) * buildingHeightPx
  const bottomPx = (interpolatedFloor / totalFloors) * buildingHeightPx;

  const loadPct = (car.load / (car.capacity || 10)) * 100;
  const isFull = car.load >= car.capacity;
  const isNearFull = loadPct >= 80;

  const borderClass = getCarStateBorder(car.state, car.out_of_service, car.fire_mode);

  return (
    <div
      data-testid={`car-${car.car_id}`}
      onClick={onClick}
      style={{
        bottom: `${bottomPx}px`,
        height: `${floorHeightPx - 2}px`,
      }}
      className={`absolute z-20 flex w-full max-w-[104px] cursor-pointer flex-col justify-between rounded-md border-2 bg-card p-1 shadow-md transition-all duration-75 hover:scale-102 ${borderClass} ${
        isSelected ? "ring-2 ring-primary ring-offset-1" : ""
      }`}
      title={`Car ${car.car_id} · Floor ${car.floor} · State: ${car.state} · Load: ${car.load}/${car.capacity}`}
    >
      {/* Top row: id, direction, badge */}
      <div className="flex items-center justify-between text-[10px] leading-none">
        <span className="font-mono font-bold text-foreground">C{car.car_id}</span>
        <span className="font-bold text-primary">{getDirectionIcon(car.direction)}</span>
        {car.out_of_service ? (
          <Wrench className="h-3 w-3 text-amber-500" />
        ) : (
          <span className="text-[9px] text-muted-foreground uppercase">{car.door}</span>
        )}
      </div>

      {/* Middle row: animated doors */}
      <div className="relative my-0.5 flex h-1.5 w-full overflow-hidden rounded-xs bg-muted">
        {car.door === "open" ? (
          <div className="flex h-full w-full justify-between">
            <span className="h-full w-1 bg-emerald-500" />
            <span className="h-full w-1 bg-emerald-500" />
          </div>
        ) : car.door === "opening" || car.door === "closing" ? (
          <div className="flex h-full w-full justify-between">
            <span className="h-full w-2 bg-amber-500" />
            <span className="h-full w-2 bg-amber-500" />
          </div>
        ) : (
          <div className="h-full w-full bg-border" />
        )}
      </div>

      {/* Bottom row: Load Bar */}
      <div className="relative h-1 w-full overflow-hidden rounded-xs bg-muted">
        <div
          style={{ width: `${Math.min(100, loadPct)}%` }}
          className={`h-full transition-all duration-200 ${
            isFull ? "bg-rose-500" : isNearFull ? "bg-amber-500" : "bg-primary"
          }`}
        />
      </div>
    </div>
  );
};

const WaitingCrowd: React.FC<{ floorData?: FloorState }> = ({ floorData }) => {
  if (!floorData) return null;
  const waiting = (floorData.waiting_up || 0) + (floorData.waiting_down || 0);
  if (waiting === 0) return null;

  const displayDots = Math.min(waiting, 8);
  const remainder = waiting - displayDots;

  return (
    <div className="flex items-center gap-0.5" title={`${waiting} passengers waiting`}>
      {Array.from({ length: displayDots }).map((_, i) => (
        <span key={i} className="h-1.5 w-1.5 rounded-full bg-primary" />
      ))}
      {remainder > 0 && (
        <span className="font-mono text-[9px] font-bold text-primary">+{remainder}</span>
      )}
    </div>
  );
};
