import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./src/test/e2e",
  snapshotDir: "./src/test/e2e/__snapshots__",
  // Update baselines with: npx playwright test --update-snapshots
  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.02, // 2% pixel tolerance for font-rendering diffs
    },
  },
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173",
    port: 4173,
    reuseExistingServer: !process.env.CI,
  },
});
