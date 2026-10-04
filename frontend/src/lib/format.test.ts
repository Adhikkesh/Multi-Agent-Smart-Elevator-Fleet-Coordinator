import { describe, expect, it } from "vitest";
import { formatDelta, formatInt, formatNumber, formatPercent, formatTime } from "./format";

describe("format helpers", () => {
  it("formats seconds to mm:ss", () => {
    expect(formatTime(0)).toBe("00:00");
    expect(formatTime(59)).toBe("00:59");
    expect(formatTime(60)).toBe("01:00");
    expect(formatTime(125)).toBe("02:05");
    expect(formatTime(-5)).toBe("00:00");
    expect(formatTime(NaN)).toBe("00:00");
  });

  it("formats floating point numbers", () => {
    expect(formatNumber(12.345, 1)).toBe("12.3");
    expect(formatNumber(12.345, 2)).toBe("12.35");
    expect(formatNumber(null)).toBe("—");
    expect(formatNumber(undefined)).toBe("—");
    expect(formatNumber(NaN)).toBe("—");
  });

  it("formats integers", () => {
    expect(formatInt(1234)).toBe("1,234");
    expect(formatInt(12.8)).toBe("13");
    expect(formatInt(null)).toBe("—");
  });

  it("formats percentages", () => {
    expect(formatPercent(45.2)).toBe("45.2%");
    expect(formatPercent(null)).toBe("—%");
  });

  it("formats deltas correctly with directionality", () => {
    const positive = formatDelta(110, 100);
    expect(positive.text).toBe("+10.0%");
    expect(positive.isGood).toBe(true);
    expect(positive.isNeutral).toBe(false);

    const negative = formatDelta(90, 100);
    expect(negative.text).toBe("-10.0%");
    expect(negative.isGood).toBe(false);

    const inverted = formatDelta(80, 100, true); // lower wait is good
    expect(inverted.text).toBe("-20.0%");
    expect(inverted.isGood).toBe(true);

    const neutral = formatDelta(100.02, 100);
    expect(neutral.isNeutral).toBe(true);

    const zeroBaseline = formatDelta(5, 0);
    expect(zeroBaseline.isNeutral).toBe(true);
  });
});
