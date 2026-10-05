// Capture screenshots of every dashboard page and fail on browser console errors.
//   uv run elevator serve --port 8765 --scenario morning_up_peak &   (then)
//   node scripts/capture_ui.mjs   (needs frontend/node_modules + `pnpm exec playwright install chromium`)
// Writes docs/img/ui/*.png. Exit code 1 if any page logs a console error.
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const require = createRequire(path.join(root, "frontend", "package.json"));
const { chromium } = require("@playwright/test");

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:8765";
const OUT = path.join(root, "docs", "img", "ui");

async function api(p, body) {
  const r = await fetch(`${BASE}${p}`, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${p} -> ${r.status}`);
  return r.json();
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 2 });
const THEME = process.env.UI_THEME ?? "light"; // the deck is white, so screenshots are light
await page.addInitScript((t) => localStorage.setItem("liftzero-theme", t), THEME);
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(`${page.url()}: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`${page.url()}: ${e.message}`));

async function shot(route, name, wait = 2500) {
  await page.goto(`${BASE}${route}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(wait);
  await page.screenshot({ path: path.join(OUT, name) });
  console.log(`  ${route} -> docs/img/ui/${name}`);
}

// A busy classical run for the main pages.
await api("/api/reset", { scenario: "morning_up_peak", strategy: "full" });
await api("/api/step", { ticks: 150 });
await shot("/", "mission_control.png");
await shot("/agents", "agents_view.png");
await page.screenshot({ path: path.join(OUT, "agents_graph.png"), clip: { x: 640, y: 170, width: 560, height: 400 } });
await shot("/lab", "search_lab.png", 3500);
await shot("/theory", "theory_primer.png");
await shot("/experiments", "experiments_view.png");

// The learned bidder with look-ahead for the Brain page.
await api("/api/reset", { scenario: "lunch_two_way", strategy: "liftzero_bc_mcts" });
await api("/api/step", { ticks: 200 });
await shot("/brain", "brain_panel.png", 3000);
const firstDecision = page.locator('ul[aria-label="Recent decisions"] button').first();
if (await firstDecision.count()) {
  await firstDecision.click();
  await page.getByRole("button", { name: /Why\?/ }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, "brain_why.png") });
  console.log("  /brain (Why?) -> docs/img/ui/brain_why.png");
}

await api("/api/reset", { scenario: "demo_story", strategy: "full" });
await browser.close();
if (errors.length) {
  console.error(`console errors (${errors.length}):\n` + errors.join("\n"));
  process.exit(1);
}
console.log("no console errors");
