import { beforeEach, describe, expect, it } from "vitest";
import { useUiStore } from "./uiStore";

describe("uiStore", () => {
  beforeEach(() => {
    useUiStore.setState({
      theme: "dark",
      selectedCarId: null,
      selectedFloor: null,
      selectedConversationId: null,
      inspectorAgentAddress: null,
      inspectorOpen: false,
      shortcutsDialogOpen: false,
      presentMode: false,
      messageFilterPerf: null,
      messageFilterAgent: null,
    });
  });

  it("toggles theme and updates documentElement classes", () => {
    expect(useUiStore.getState().theme).toBe("dark");

    useUiStore.getState().toggleTheme();
    expect(useUiStore.getState().theme).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    useUiStore.getState().toggleTheme();
    expect(useUiStore.getState().theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("selects car, floor, and conversation", () => {
    useUiStore.getState().selectCar(2);
    expect(useUiStore.getState().selectedCarId).toBe(2);

    useUiStore.getState().selectFloor(5);
    expect(useUiStore.getState().selectedFloor).toBe(5);

    useUiStore.getState().selectConversation("c44");
    expect(useUiStore.getState().selectedConversationId).toBe("c44");
  });

  it("opens and closes inspector drawer", () => {
    useUiStore.getState().openInspector("car-0");
    expect(useUiStore.getState().inspectorAgentAddress).toBe("car-0");
    expect(useUiStore.getState().inspectorOpen).toBe(true);

    useUiStore.getState().closeInspector();
    expect(useUiStore.getState().inspectorOpen).toBe(false);
  });

  it("manages dialog and presentation states", () => {
    useUiStore.getState().setShortcutsDialogOpen(true);
    expect(useUiStore.getState().shortcutsDialogOpen).toBe(true);

    useUiStore.getState().setPresentMode(true);
    expect(useUiStore.getState().presentMode).toBe(true);
  });

  it("sets message filters", () => {
    useUiStore.getState().setMessageFilterPerf("CFP");
    expect(useUiStore.getState().messageFilterPerf).toBe("CFP");

    useUiStore.getState().setMessageFilterAgent("dispatcher");
    expect(useUiStore.getState().messageFilterAgent).toBe("dispatcher");
  });
});
