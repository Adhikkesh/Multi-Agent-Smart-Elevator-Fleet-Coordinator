import { beforeEach, describe, expect, it } from "vitest";
import { STORY_BEATS, useStoryStore } from "./storyStore";

describe("storyStore", () => {
  beforeEach(() => {
    useStoryStore.getState().resetStory();
  });

  it("contains 10 beats covering the full live demo script", () => {
    expect(STORY_BEATS.length).toBe(10);
    expect(STORY_BEATS[0]?.title).toContain("Building");
    expect(STORY_BEATS[9]?.title).toContain("LiftZero");
  });

  it("advances and steps back through beats within valid bounds", () => {
    expect(useStoryStore.getState().activeBeatIndex).toBe(0);

    useStoryStore.getState().nextBeat();
    expect(useStoryStore.getState().activeBeatIndex).toBe(1);

    useStoryStore.getState().prevBeat();
    expect(useStoryStore.getState().activeBeatIndex).toBe(0);

    useStoryStore.getState().prevBeat();
    expect(useStoryStore.getState().activeBeatIndex).toBe(0); // Clamped at 0

    useStoryStore.getState().setBeatIndex(99);
    expect(useStoryStore.getState().activeBeatIndex).toBe(9); // Clamped at max

    useStoryStore.getState().setBeatIndex(-5);
    expect(useStoryStore.getState().activeBeatIndex).toBe(0); // Clamped at min
  });

  it("manages autoplay and beat execution states", () => {
    useStoryStore.getState().setAutoPlaying(true);
    expect(useStoryStore.getState().isAutoPlaying).toBe(true);

    useStoryStore.getState().setBeatExecuting(true);
    expect(useStoryStore.getState().beatExecuting).toBe(true);

    useStoryStore.getState().resetStory();
    expect(useStoryStore.getState().activeBeatIndex).toBe(0);
    expect(useStoryStore.getState().isAutoPlaying).toBe(false);
    expect(useStoryStore.getState().beatExecuting).toBe(false);
  });
});
