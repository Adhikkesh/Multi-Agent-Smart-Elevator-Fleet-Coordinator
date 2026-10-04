import type { Direction } from "../api/types";

/**
 * Calculates the interpolated floor position of a car.
 * floor + sign(direction) * progress
 *
 * @param floor Current integer floor
 * @param direction Direction enum (UP, DOWN, IDLE)
 * @param progress Fraction of the way to next floor [0, 1]
 * @returns Real number representing vertical floor position
 */
export function getInterpolatedCarFloor(
  floor: number,
  direction: Direction,
  progress: number,
): number {
  const clampedProgress = Math.max(0, Math.min(1, isNaN(progress) ? 0 : progress));
  if (direction === "UP") {
    return floor + clampedProgress;
  }
  if (direction === "DOWN") {
    return floor - clampedProgress;
  }
  return floor;
}
