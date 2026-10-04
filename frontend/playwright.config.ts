import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  timeout: 45000,
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "on-first-retry",
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "uv run elevator serve --port 8000 --scenario demo_story",
    url: "http://127.0.0.1:8000/api/version",
    reuseExistingServer: true,
    cwd: path.resolve(__dirname, ".."),
    timeout: 30000,
  },
});
