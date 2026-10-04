import { describe, expect, it } from "vitest";
import {
  getCarStateBorder,
  getDirectionIcon,
  getPerformativeColor,
  PERFORMATIVE_COLORS,
} from "./colors";

describe("colors helpers", () => {
  it("provides valid CSS variables for all 9 performatives", () => {
    const performatives = [
      "REQUEST",
      "CFP",
      "PROPOSE",
      "REFUSE",
      "ACCEPT_PROPOSAL",
      "REJECT_PROPOSAL",
      "INFORM",
      "CANCEL",
      "FAILURE",
    ] as const;

    for (const p of performatives) {
      expect(PERFORMATIVE_COLORS[p]).toBeDefined();
      expect(getPerformativeColor(p)).toBe(PERFORMATIVE_COLORS[p]);
    }

    expect(getPerformativeColor("UNKNOWN")).toBe("#94a3b8");
  });

  it("returns appropriate border styles for car states", () => {
    expect(getCarStateBorder("moving_up", false, false)).toContain("border-primary");
    expect(getCarStateBorder("doors", false, false)).toContain("border-emerald-500");
    expect(getCarStateBorder("idle", true, false)).toContain("opacity-60");
    expect(getCarStateBorder("moving_up", false, true)).toContain("border-red-500");
    expect(getCarStateBorder("idle", false, false)).toBe("border-border");
  });

  it("returns direction icons", () => {
    expect(getDirectionIcon("UP")).toBe("▲");
    expect(getDirectionIcon("DOWN")).toBe("▼");
    expect(getDirectionIcon("IDLE")).toBe("—");
  });
});
