import { describe, expect, it } from "vitest";
import { getInterpolatedCarFloor } from "./interpolate";

describe("interpolate car floor", () => {
  it("interpolates UP direction smoothly", () => {
    expect(getInterpolatedCarFloor(3, "UP", 0.0)).toBe(3.0);
    expect(getInterpolatedCarFloor(3, "UP", 0.5)).toBe(3.5);
    expect(getInterpolatedCarFloor(3, "UP", 1.0)).toBe(4.0);
  });

  it("interpolates DOWN direction smoothly", () => {
    expect(getInterpolatedCarFloor(5, "DOWN", 0.0)).toBe(5.0);
    expect(getInterpolatedCarFloor(5, "DOWN", 0.5)).toBe(4.5);
    expect(getInterpolatedCarFloor(5, "DOWN", 1.0)).toBe(4.0);
  });

  it("remains at floor level for IDLE", () => {
    expect(getInterpolatedCarFloor(7, "IDLE", 0.5)).toBe(7.0);
  });

  it("clamps progress to [0, 1]", () => {
    expect(getInterpolatedCarFloor(2, "UP", -0.5)).toBe(2.0);
    expect(getInterpolatedCarFloor(2, "UP", 1.5)).toBe(3.0);
  });

  it("handles NaN progress safely", () => {
    expect(getInterpolatedCarFloor(4, "UP", NaN)).toBe(4.0);
  });
});
