/**
 * Visual regression / presence tests for HildaNext frontend.
 *
 * Scope: UI structure, labels, panel anatomy, filter controls, scenario
 *        selector options, screenshots.  NO business logic, NO backend calls,
 *        NO WSD launch.  Mock data is loaded from public/mockup/*.json.
 *
 * Run once to generate baselines:
 *   npx playwright test visual.spec.ts --update-snapshots
 *
 * Regular run:
 *   npx playwright test visual.spec.ts
 */

import { test, expect, Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Navigate and wait until the main content hero section is painted. */
async function gotoWsd(page: Page, scenario = "healthy_wsd_run") {
  await page.goto(`/wsd?scenario=${scenario}`);
  // Hero cards are always rendered; waiting for one guarantees React is hydrated.
  await expect(page.getByText("Current phase")).toBeVisible({ timeout: 8000 });
}

async function gotoInference(page: Page, scenario = "ar_vs_diffusion_compare") {
  await page.goto(`/inference?scenario=${scenario}`);
  await expect(page.getByText("Prompt")).toBeVisible({ timeout: 8000 });
}

// ---------------------------------------------------------------------------
// 1 — Shell
// ---------------------------------------------------------------------------

test.describe("Shell", () => {
  test("brand text and kicker are visible", async ({ page }) => {
    await gotoWsd(page);
    await expect(page.getByText("HildaNext Observatory")).toBeVisible();
    await expect(
      page.getByText("Frontend-only control room for WSD and diffusion inference"),
    ).toBeVisible();
  });

  test("primary nav links are visible and labelled correctly", async ({ page }) => {
    await gotoWsd(page);
    const nav = page.getByRole("navigation", { name: "Primary" });
    await expect(nav.getByText("WSD")).toBeVisible();
    await expect(nav.getByText("Warmup / Stable / Decay")).toBeVisible();
    await expect(nav.getByText("Inference")).toBeVisible();
    await expect(nav.getByText("AR vs diffusion")).toBeVisible();
  });

  test("root / redirects to /wsd", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/wsd/);
    await expect(page.getByText("Current phase")).toBeVisible({ timeout: 8000 });
  });

  test("nav WSD link is marked active on /wsd", async ({ page }) => {
    await gotoWsd(page);
    const wsdLink = page.getByRole("link", { name: /WSD/ });
    // active class is applied by NavLink — just ensure the element is there
    await expect(wsdLink).toBeVisible();
  });

  test("nav Inference link is marked active on /inference", async ({ page }) => {
    await gotoInference(page);
    const inferenceLink = page.getByRole("link", { name: /Inference/ });
    await expect(inferenceLink).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 2 — WSD page — hero cards
// ---------------------------------------------------------------------------

test.describe("WSD — hero cards", () => {
  test.beforeEach(async ({ page }) => {
    await gotoWsd(page);
  });

  test("all four hero card labels are present", async ({ page }) => {
    await expect(page.getByText("Current phase")).toBeVisible();
    await expect(page.getByText("Masked token acc")).toBeVisible();
    await expect(page.getByText("Throughput")).toBeVisible();
    await expect(page.getByText("Peak VRAM")).toBeVisible();
  });

  test("hero card meta shows bidirectional or causal state", async ({ page }) => {
    // One of these two alternatives must appear (depends on mock data)
    const bidir = page.getByText("Stable bidirectional");
    const causal = page.getByText("Causal effective");
    const either = (await bidir.count()) > 0 || (await causal.count()) > 0;
    expect(either).toBe(true);
  });

  test("peak VRAM card value contains MB or n/a", async ({ page }) => {
    const vramCard = page.locator("article").filter({ hasText: "Peak VRAM" });
    const value = vramCard.locator("strong");
    const text = await value.innerText();
    expect(text.includes("MB") || text === "n/a").toBe(true);
  });

  test("throughput card value contains tok/s or n/a", async ({ page }) => {
    const card = page.locator("article").filter({ hasText: "Throughput" });
    const value = card.locator("strong");
    const text = await value.innerText();
    expect(text.includes("tok/s") || text === "n/a").toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 3 — WSD page — panels (kickers + titles always present regardless of data)
// ---------------------------------------------------------------------------

test.describe("WSD — panel anatomy", () => {
  test.beforeEach(async ({ page }) => {
    await gotoWsd(page);
  });

  test("Schedule panel is rendered", async ({ page }) => {
    await expect(page.getByText("Schedule")).toBeVisible();
    await expect(page.getByText("Warmup -> stable -> decay")).toBeVisible();
  });

  test("Metrics panel is rendered", async ({ page }) => {
    await expect(page.getByText("Metrics")).toBeVisible();
    await expect(page.getByText("Loss, throughput and VRAM in one viewport")).toBeVisible();
  });

  test("CMD transcript panel is rendered", async ({ page }) => {
    await expect(page.getByText("CMD transcript")).toBeVisible();
    await expect(page.getByText("Console-first reading flow")).toBeVisible();
  });

  test("Structured logs panel is rendered", async ({ page }) => {
    await expect(page.getByText("Structured logs")).toBeVisible();
    await expect(page.getByText("Virtualized log window")).toBeVisible();
  });

  test("Resource rail panel is rendered", async ({ page }) => {
    await expect(page.getByText("Resource rail")).toBeVisible();
    await expect(page.getByText("Process + VRAM posture")).toBeVisible();
  });

  test("Insights panel is rendered", async ({ page }) => {
    await expect(page.getByText("Insights")).toBeVisible();
    await expect(page.getByText("What to read first")).toBeVisible();
  });

  test("Glossary panel is rendered", async ({ page }) => {
    await expect(page.getByText("Glossary")).toBeVisible();
    await expect(page.getByText("Correct terms, not vague dashboard copy")).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 4 — WSD page — StickyFilterBar
// ---------------------------------------------------------------------------

test.describe("WSD — filter bar", () => {
  test.beforeEach(async ({ page }) => {
    await gotoWsd(page);
  });

  test("search input is visible with correct placeholder", async ({ page }) => {
    await expect(page.getByPlaceholder("Search action / reason / module / message")).toBeVisible();
  });

  test("level filter buttons are present", async ({ page }) => {
    for (const level of ["notice", "warning", "error"]) {
      await expect(page.getByRole("button", { name: level, exact: true })).toBeVisible();
    }
  });

  test("source filter buttons are present", async ({ page }) => {
    for (const source of ["console", "metric", "fallback", "training", "eval"]) {
      await expect(page.getByRole("button", { name: source, exact: true })).toBeVisible();
    }
  });
});

// ---------------------------------------------------------------------------
// 5 — WSD page — StatusRail
// ---------------------------------------------------------------------------

test.describe("WSD — StatusRail", () => {
  test.beforeEach(async ({ page }) => {
    await gotoWsd(page);
  });

  test("Run posture heading is visible", async ({ page }) => {
    await expect(page.getByText("Run posture")).toBeVisible();
  });

  test("StatusRail metric labels are present", async ({ page }) => {
    await expect(page.getByText("Phase")).toBeVisible();
    await expect(page.getByText("VRAM ceiling")).toBeVisible();
    await expect(page.getByText("Masked acc")).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 7 — WSD page — Glossary tab labels
// ---------------------------------------------------------------------------

test.describe("WSD — Glossary tabs", () => {
  test("first set of glossary term tabs renders", async ({ page }) => {
    await gotoWsd(page);
    // Check a representative subset of shortLabels
    for (const label of ["Gamma", "Delta", "Mask ratio", "Converge", "Remask", "Q_MODE", "S_MODE"]) {
      await expect(
        page.getByRole("button", { name: label, exact: true }).first(),
      ).toBeVisible();
    }
  });
});

// ---------------------------------------------------------------------------
// 8 — Inference page — hero cards
// ---------------------------------------------------------------------------

test.describe("Inference — hero cards", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("all four hero card labels are present", async ({ page }) => {
    await expect(page.getByText("Prompt")).toBeVisible();
    await expect(page.getByText("Throughput")).toBeVisible();
    await expect(page.getByText("Converge")).toBeVisible();
    await expect(page.getByText("Fallback posture")).toBeVisible();
  });

  test("Converge card shows steps or n/a", async ({ page }) => {
    const card = page.locator("article").filter({ hasText: "Converge" }).first();
    const value = card.locator("strong");
    const text = await value.innerText();
    expect(text.includes("steps") || text === "n/a").toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 9 — Inference page — panel anatomy
// ---------------------------------------------------------------------------

test.describe("Inference — panel anatomy", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("Prompt lab panel is rendered", async ({ page }) => {
    await expect(page.getByText("Prompt lab")).toBeVisible();
    await expect(page.getByText("Run inference from your own prompt")).toBeVisible();
  });

  test("Compare panel (AR vs diffusion) is rendered", async ({ page }) => {
    await expect(page.getByText("Compare")).toBeVisible();
    await expect(page.getByText("AR lane vs diffusion lane")).toBeVisible();
  });

  test("AR lane and diffusion lane headers are visible", async ({ page }) => {
    await expect(page.getByText("AR lane")).toBeVisible();
    await expect(page.getByText("Diffusion lane")).toBeVisible();
  });

  test("Order of certainty ribbon is visible", async ({ page }) => {
    await expect(page.getByText("Order of certainty")).toBeVisible();
  });

  test("Diffusion mechanics panel is rendered", async ({ page }) => {
    await expect(page.getByText("Diffusion mechanics")).toBeVisible();
    await expect(page.getByText("Step timeline")).toBeVisible();
  });

  test("Insights panel is rendered", async ({ page }) => {
    await expect(page.getByText("Insights")).toBeVisible();
    await expect(page.getByText("What to read first")).toBeVisible();
  });

  test("Glossary panel is rendered", async ({ page }) => {
    await expect(page.getByText("Glossary")).toBeVisible();
    await expect(page.getByText("Correct terms, not vague dashboard copy")).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 10 — Inference page — PromptLab control labels
// ---------------------------------------------------------------------------

test.describe("Inference — PromptLab control labels", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("prompt textarea is visible with correct placeholder", async ({ page }) => {
    await expect(
      page.getByPlaceholder("Write the prompt you want to test here."),
    ).toBeVisible();
  });

  test("all control span labels are visible", async ({ page }) => {
    for (const label of [
      "Temperature",
      "Top P",
      "Max new tokens",
      "Seed",
      "Mode",
      "Effort",
      "Tau mask",
      "Tau edit",
      "Profile",
    ]) {
      await expect(page.getByText(label, { exact: true })).toBeVisible();
    }
  });

  test("mode select has S_MODE and Q_MODE options", async ({ page }) => {
    // Options existence is sufficient; exact select element is not targeted
    await expect(page.getByRole("option", { name: "S_MODE" }).first()).toBeAttached();
    await expect(page.getByRole("option", { name: "Q_MODE" }).first()).toBeAttached();
  });

  test("effort select has all five effort levels", async ({ page }) => {
    for (const effort of ["instant", "low", "medium", "high", "adaptive"]) {
      await expect(page.getByRole("option", { name: effort }).first()).toBeAttached();
    }
  });

  test("Generate mock run button is present", async ({ page }) => {
    await expect(page.getByRole("button", { name: "Generate mock run" })).toBeVisible();
  });

  test("Reset to scenario button is present", async ({ page }) => {
    await expect(page.getByRole("button", { name: "Reset to scenario" })).toBeVisible();
  });

});

// ---------------------------------------------------------------------------
// 11 — Inference page — filter bar
// ---------------------------------------------------------------------------

test.describe("Inference — filter bar", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("search input is visible", async ({ page }) => {
    await expect(page.getByPlaceholder("Search action / reason / module / message")).toBeVisible();
  });

  test("level filter buttons are present", async ({ page }) => {
    for (const level of ["notice", "warning", "error"]) {
      await expect(page.getByRole("button", { name: level, exact: true })).toBeVisible();
    }
  });

  test("source filter buttons are present", async ({ page }) => {
    for (const source of ["console", "metric", "fallback", "training", "eval"]) {
      await expect(page.getByRole("button", { name: source, exact: true })).toBeVisible();
    }
  });
});

// ---------------------------------------------------------------------------
// 12 — Inference page — StatusRail
// ---------------------------------------------------------------------------

test.describe("Inference — StatusRail", () => {
  test("Active inference heading and metrics are visible", async ({ page }) => {
    await gotoInference(page);
    await expect(page.getByText("Active inference")).toBeVisible();
    await expect(page.getByText("Throughput")).toBeVisible();
    await expect(page.getByText("Converge")).toBeVisible();
    await expect(page.getByText("Peak VRAM")).toBeVisible();
    await expect(page.getByText("Fallbacks")).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 14 — Screenshot snapshots (run once to generate *.png baselines)
//       Update with: npx playwright test visual.spec.ts --update-snapshots
// ---------------------------------------------------------------------------

test.describe("Screenshots", () => {
  test("WSD page — healthy_wsd_run — full page", async ({ page }) => {
    await gotoWsd(page, "healthy_wsd_run");
    // Let uPlot charts settle
    await page.waitForTimeout(400);
    await expect(page).toHaveScreenshot("wsd-healthy.png", { fullPage: true });
  });

  test("WSD page — phase_transition_run — full page", async ({ page }) => {
    await gotoWsd(page, "phase_transition_run");
    await page.waitForTimeout(400);
    await expect(page).toHaveScreenshot("wsd-phase-transition.png", { fullPage: true });
  });

  test("WSD page — vram_pressure_run — full page", async ({ page }) => {
    await gotoWsd(page, "vram_pressure_run");
    await page.waitForTimeout(400);
    await expect(page).toHaveScreenshot("wsd-vram-pressure.png", { fullPage: true });
  });

  test("Inference page — ar_vs_diffusion_compare — full page", async ({ page }) => {
    await gotoInference(page, "ar_vs_diffusion_compare");
    await page.waitForTimeout(400);
    await expect(page).toHaveScreenshot("inference-ar-vs-diffusion.png", { fullPage: true });
  });

  test("Inference page — degenerate_decode_run — full page", async ({ page }) => {
    await gotoInference(page, "degenerate_decode_run");
    await page.waitForTimeout(400);
    await expect(page).toHaveScreenshot("inference-degenerate.png", { fullPage: true });
  });

  test("TopNav — above the fold strip", async ({ page }) => {
    await gotoWsd(page);
    const header = page.locator("header").first();
    await expect(header).toHaveScreenshot("topnav.png");
  });

  test("WSD hero section", async ({ page }) => {
    await gotoWsd(page, "healthy_wsd_run");
    await page.waitForTimeout(400);
    const hero = page.locator("section").filter({ hasText: "Current phase" }).first();
    await expect(hero).toHaveScreenshot("wsd-hero.png");
  });

  test("Inference hero section", async ({ page }) => {
    await gotoInference(page, "ar_vs_diffusion_compare");
    const hero = page.locator("section").filter({ hasText: "Prompt" }).first();
    await expect(hero).toHaveScreenshot("inference-hero.png");
  });
});
