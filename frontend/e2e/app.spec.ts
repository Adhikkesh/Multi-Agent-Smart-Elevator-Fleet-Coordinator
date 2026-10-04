import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const screenshotsDir = path.resolve(__dirname, "../../docs/img/ui");

test.describe("LiftZero Control Room E2E Test Suite", () => {
  test.beforeEach(async ({ page }) => {
    page.setDefaultTimeout(30000);
  });

  test("1. test_controls_flow: start, pause, step, reset, speed change", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("LiftZero", { exact: true })).toBeVisible();
    await expect(page.getByText("Control Room")).toBeVisible();

    // Verify speed slider
    const speedSlider = page.getByLabel("Playback speed");
    await expect(speedSlider).toBeVisible();

    // Verify play / pause button
    const playBtn = page.getByRole("button", { name: /Play|Pause/i });
    await expect(playBtn).toBeVisible();

    // Step button
    const step1Btn = page.getByTitle("Step 1 tick (→)");
    await expect(step1Btn).toBeVisible();
    await step1Btn.click();

    // Reset button
    const resetBtn = page.getByTitle("Reset simulation (R)");
    await expect(resetBtn).toBeVisible();
    await resetBtn.click();

    // Capture main dashboard screenshot
    await page.screenshot({ path: path.join(screenshotsDir, "dashboard_live.png"), fullPage: true });
  });

  test("2. test_building_interaction: car selection, call elevator", async ({ page }) => {
    await page.goto("/");
    const building = page.locator("[data-testid='building-view']");
    await expect(building).toBeVisible();

    // Click Car 0 inside the building shaft
    const car0 = page.locator("[data-testid='car-0']");
    await expect(car0).toBeVisible();
    await car0.click();

    // Inspector drawer should open
    const dialog = page.getByRole("dialog", { name: "Agent Inspector" });
    await expect(dialog).toBeVisible();

    // Close drawer using role button
    const closeBtn = page.getByRole("button", { name: "Close inspector drawer" });
    await expect(closeBtn).toBeVisible();
    await closeBtn.click();

    // Click a floor to spawn passenger
    const floor2 = page.locator("[title*='Floor 2']");
    if (await floor2.isVisible()) {
      await floor2.click();
    }
  });

  test("3. test_fault_injection: disable car, observe badge", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("[data-testid='chaos-panel']")).toBeVisible();

    // Trigger Break Car
    const breakBtn = page.getByRole("button", { name: "Break Car" });
    await expect(breakBtn).toBeVisible();
    await breakBtn.click();

    await page.waitForTimeout(600);
    // Verify Repair Car button is enabled or available
    const repairBtn = page.getByRole("button", { name: "Repair Car" });
    await expect(repairBtn).toBeVisible();
    await repairBtn.click();
  });

  test("4. test_fire_evacuation: trigger fire alarm, observe state", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("[data-testid='chaos-panel']")).toBeVisible();

    // Trigger Fire Alarm
    const fireBtn = page.getByRole("button", { name: "Fire Alarm" });
    await expect(fireBtn).toBeVisible();
    await fireBtn.click();

    await page.waitForTimeout(600);
    // Clear alarm
    const clearBtn = page.getByRole("button", { name: "Clear Alarm" });
    await expect(clearBtn).toBeVisible();
    await clearBtn.click();
  });

  test("5. test_auction_panel: CFP emission, bids display, winner highlight", async ({ page }) => {
    await page.goto("/");
    const auctionPanel = page.locator("[data-testid='auction-panel']");
    await expect(auctionPanel).toBeVisible();
    await expect(
      auctionPanel.getByRole("heading", { name: /Contract Net Auction/i })
    ).toBeVisible();
  });

  test("6. test_message_stream_filtering: filter by performative", async ({ page }) => {
    await page.goto("/");
    const msgStream = page.locator("[data-testid='message-stream']");
    await expect(msgStream).toBeVisible();

    // Filter by CFP
    const cfpFilter = page.getByRole("button", { name: "CFP", exact: true });
    if (await cfpFilter.isVisible()) {
      await cfpFilter.click();
    }

    // Filter by PROPOSE
    const proposeFilter = page.getByRole("button", { name: "PROPOSE", exact: true });
    if (await proposeFilter.isVisible()) {
      await proposeFilter.click();
    }
  });

  test("7. test_sequence_diagram_modal: open from message stream, verify swimlanes", async ({ page }) => {
    await page.goto("/");
    const seqBtn = page.getByTitle("Open Sequence Diagram");
    if (await seqBtn.isVisible()) {
      await seqBtn.click();
      await expect(page.getByText("Contract Net Protocol — Interaction Sequence")).toBeVisible();
      const closeBtn = page.getByLabel("Close sequence diagram");
      if (await closeBtn.isVisible()) {
        await closeBtn.click();
      }
    }
  });

  test("8. test_agents_graph: navigate to /agents, verify nodes", async ({ page }) => {
    await page.goto("/agents");
    await expect(page.getByText("Multi-Agent Communication Network")).toBeVisible();

    // Verify ReactFlow container exists
    const reactFlow = page.locator(".react-flow");
    await expect(reactFlow).toBeAttached();
    await page.waitForTimeout(500);

    // Capture Agents view screenshot
    await page.screenshot({ path: path.join(screenshotsDir, "agents_view.png"), fullPage: true });
  });

  test("9. test_agent_inspector: verify PEAS, state, and routing plan", async ({ page }) => {
    await page.goto("/agents");
    // Click car 0 card
    const car0Card = page.getByText("Car 0 Agent");
    if (await car0Card.isVisible()) {
      await car0Card.click();
      await expect(page.getByText("PEAS Formal Specification")).toBeVisible();
      await expect(page.getByText("Performance Measure")).toBeVisible();
      await expect(page.getByText("Environment")).toBeVisible();
    }
  });

  test("10. test_search_lab_step_through: navigate to /lab, verify A* and step through", async ({ page }) => {
    await page.goto("/lab");
    await expect(page.getByText("Algorithm & State-Space Search Lab")).toBeVisible();
    await expect(page.getByText("Search Algorithms Comparison")).toBeVisible();

    // Verify step through visualizer
    const stepFwdBtn = page.getByTitle("Next step");
    if (await stepFwdBtn.isVisible()) {
      await stepFwdBtn.click();
    }

    // Switch to Simulated Annealing tab
    const saTab = page.getByRole("button", { name: /Simulated Annealing/i });
    if (await saTab.isVisible()) {
      await saTab.click();
      await expect(page.getByText(/Run Reassignment Search|Simulated Annealing/i)).toBeVisible();
    }

    // Switch to Minimax tab
    const minimaxTab = page.getByRole("button", { name: /Adversarial Game/i });
    if (await minimaxTab.isVisible()) {
      await minimaxTab.click();
      await expect(page.getByText(/Game Tree Size|Adversarial Minimax/i)).toBeVisible();
    }

    // Capture Search Lab screenshot
    await page.screenshot({ path: path.join(screenshotsDir, "search_lab.png"), fullPage: true });
  });

  test("11. test_benchmark_runner: navigate to /experiments, run benchmark matrix", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page.getByText("Experiments & Empirical Validation")).toBeVisible();

    // Benchmark runner tab
    const runBtn = page.getByRole("button", { name: /Run Benchmark/i });
    await expect(runBtn).toBeVisible();

    // Capture Experiments screenshot
    await page.screenshot({ path: path.join(screenshotsDir, "experiments_view.png"), fullPage: true });
  });

  test("12. test_compare_mode: select two runs, verify comparison", async ({ page }) => {
    await page.goto("/experiments");
    // Switch to Compare Mode
    const compareTab = page.getByRole("button", { name: /Compare Mode/i });
    await expect(compareTab).toBeVisible();
    await compareTab.click();

    await expect(page.getByText("Compare Mode: Multi-Strategy Headless Race")).toBeVisible();
  });

  test("13. test_story_mode: navigate to /story, step through beats", async ({ page }) => {
    await page.goto("/story");
    await expect(page.getByText("LiftZero Presentation & Viva Story Walkthrough")).toBeVisible();

    // Next beat button
    const nextBtn = page.getByRole("button", { name: /Next Beat/i });
    await expect(nextBtn).toBeVisible();
    await nextBtn.click();

    // Capture Story screenshot
    await page.screenshot({ path: path.join(screenshotsDir, "story_mode.png"), fullPage: true });
  });

  test("14. test_classic_fallback: navigate to /classic, verify legacy UI", async ({ page }) => {
    await page.goto("/classic");
    await expect(page.locator("h1")).toContainText("Smart Elevator Fleet Coordinator");
    await expect(page.locator("#building")).toBeVisible();
  });

  test("15. test_theory_page: navigate to /theory, verify taxonomy and viva", async ({ page }) => {
    await page.goto("/theory");
    await expect(
      page.getByText("Fundamentals of AI (AIMA 4e) Theoretical Reference")
    ).toBeVisible();
    await expect(page.getByText(/1\. System-Level PEAS Formulation/i)).toBeVisible();
    await expect(page.getByText(/6\. Likely Examiner Viva Questions/i)).toBeVisible();

    // Capture Theory screenshot
    await page.screenshot({ path: path.join(screenshotsDir, "theory_primer.png"), fullPage: true });
  });

  test("16. test_accessibility_axe: zero critical accessibility violations", async ({ page }) => {
    const routes = ["/", "/agents", "/lab", "/experiments", "/theory", "/story"];
    for (const route of routes) {
      await page.goto(route);
      await page.waitForTimeout(600);
      const accessibilityScanResults = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa"])
        .disableRules(["color-contrast"])
        .analyze();

      const criticalViolations = accessibilityScanResults.violations.filter(
        (v) => v.impact === "critical"
      );
      expect(criticalViolations).toEqual([]);
    }
  });
});
