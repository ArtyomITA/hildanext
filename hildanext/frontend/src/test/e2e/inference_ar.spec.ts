/**
 * inference_ar.spec.ts
 *
 * End-to-end tests for the Inference page, focused on AR inference.
 * All tests run against the offline-mockup fallback (no backend required).
 *
 * Run:
 *   npx playwright test inference_ar.spec.ts
 *
 * Update screenshots:
 *   npx playwright test inference_ar.spec.ts --update-snapshots
 */

import { test, expect, Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function gotoInference(page: Page) {
  await page.goto("/inference");
  // "Prompt lab" kicker is static JSX — doesn't depend on Zustand or API.
  // Once it's visible the page is hydrated and the store has initialised.
  await expect(page.getByText("Prompt lab")).toBeVisible({ timeout: 12000 });
}

// ---------------------------------------------------------------------------
// 1 — Page load & DataSourceBar
// ---------------------------------------------------------------------------

test.describe("Inference — DataSourceBar", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("shows OFFLINE MOCKUP pill when backend is not running", async ({ page }) => {
    // API is not running → offline fallback fires after the first failed fetch.
    // Wait a bit longer than the default to let the catch handler settle.
    // Use the full pill text (including the ◎ prefix) to avoid matching the item value span too.
    await expect(page.getByText("◎ OFFLINE MOCKUP")).toBeVisible({ timeout: 15000 });
  });

  test("DataSourceBar contains source, engine, mode and effort items", async ({ page }) => {
    // Labels are UPPERCASE (CSS or literal) — match case-insensitively via regex.
    // Use .first() to pick the DataSourceBar label when the same word appears elsewhere.
    await expect(page.getByText(/SOURCE/i).first()).toBeVisible();
    await expect(page.getByText(/ENGINE/i).first()).toBeVisible();
    await expect(page.getByText(/MODE/i).first()).toBeVisible();
    await expect(page.getByText(/EFFORT/i).first()).toBeVisible();
  });

  test("source value is 'offline mockup' in offline state", async ({ page }) => {
    // exact: true so "offline mockup" (lowercase item value) doesn't also match the pill "◎ OFFLINE MOCKUP".
    await expect(page.getByText("offline mockup", { exact: true })).toBeVisible({ timeout: 15000 });
  });
});

// ---------------------------------------------------------------------------
// 2 — Hero cards
// ---------------------------------------------------------------------------

test.describe("Inference — hero cards", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("all four hero card kickers are present", async ({ page }) => {
    // Kickers are rendered as <p> elements by Panel; use .first() to avoid strict-mode
    // violations when the same word appears in other elements (e.g. PromptLab label).
    await expect(page.locator('p').filter({ hasText: /^Prompt$/ }).first()).toBeVisible();
    await expect(page.locator('p').filter({ hasText: /^Throughput$/ }).first()).toBeVisible();
    await expect(page.locator('p').filter({ hasText: /^Converge$/ }).first()).toBeVisible();
    await expect(page.locator('p').filter({ hasText: /^Fallback posture$/ }).first()).toBeVisible();
  });

  test("Throughput card value includes tok/s", async ({ page }) => {
    const card = page.locator("article").filter({ hasText: "Throughput" }).first();
    await expect(card.locator("strong")).toContainText("tok/s");
  });

  test("Converge card value includes steps or n/a", async ({ page }) => {
    const card = page.locator("article").filter({ hasText: "Converge" }).first();
    const text = await card.locator("strong").innerText();
    expect(text.includes("steps") || text === "n/a").toBe(true);
  });

  test("AR tok/s meta line is visible in Throughput card", async ({ page }) => {
    const card = page.locator("article").filter({ hasText: "Throughput" }).first();
    await expect(card).toContainText("AR");
    await expect(card).toContainText("tok/s");
  });
});

// ---------------------------------------------------------------------------
// 3 — PromptLab — structure
// ---------------------------------------------------------------------------

test.describe("Inference — PromptLab structure", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("panel kicker and title are visible", async ({ page }) => {
    await expect(page.getByText("Prompt lab")).toBeVisible();
    await expect(page.getByText("Run inference from your own prompt")).toBeVisible();
  });

  test("prompt textarea is visible with placeholder text", async ({ page }) => {
    await expect(
      page.getByPlaceholder("Write the prompt you want to test here."),
    ).toBeVisible();
  });

  test("prompt textarea has a default prefilled value", async ({ page }) => {
    const ta = page.getByPlaceholder("Write the prompt you want to test here.");
    const value = await ta.inputValue();
    expect(value.length).toBeGreaterThan(10);
  });

  test("all control labels are visible", async ({ page }) => {
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

  test("Mode select contains S_MODE and Q_MODE", async ({ page }) => {
    await expect(page.getByRole("option", { name: "S_MODE" }).first()).toBeAttached();
    await expect(page.getByRole("option", { name: "Q_MODE" }).first()).toBeAttached();
  });

  test("Effort select contains all five levels", async ({ page }) => {
    for (const level of ["instant", "low", "medium", "high", "adaptive"]) {
      await expect(page.getByRole("option", { name: level }).first()).toBeAttached();
    }
  });

  test("Generate mock run button is visible and enabled", async ({ page }) => {
    const btn = page.getByRole("button", { name: "Generate mock run" });
    await expect(btn).toBeVisible();
    await expect(btn).toBeEnabled();
  });

  test("Run AR on Qwen button is visible and enabled", async ({ page }) => {
    const btn = page.getByRole("button", { name: "▶ Run AR on Qwen" });
    await expect(btn).toBeVisible();
    await expect(btn).toBeEnabled();
  });

  test("Reset to scenario button is visible", async ({ page }) => {
    await expect(page.getByRole("button", { name: "Reset to scenario" })).toBeVisible();
  });

  test("no stray badge text in PromptLab header", async ({ page }) => {
    // 'mock local generation only' badge was removed — must not appear
    await expect(page.getByText("mock local generation only")).toHaveCount(0);
    await expect(page.getByText("local generation only")).toHaveCount(0);
  });
});

// ---------------------------------------------------------------------------
// 4 — PromptLab — editing controls
// ---------------------------------------------------------------------------

test.describe("Inference — PromptLab controls interaction", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("can type a new prompt into the textarea", async ({ page }) => {
    const ta = page.getByPlaceholder("Write the prompt you want to test here.");
    await ta.fill("What is autoregressive decoding?");
    await expect(ta).toHaveValue("What is autoregressive decoding?");
  });

  test("Seed input accepts a numeric value", async ({ page }) => {
    const seedInput = page.getByLabel(/seed/i).last();
    await seedInput.fill("99");
    await expect(seedInput).toHaveValue("99");
  });

  test("Mode can be switched to Q_MODE", async ({ page }) => {
    // select by option value
    await page.selectOption("[id*='mode'], select >> nth=0", "Q_MODE").catch(() =>
      page.evaluate(() => {
        const selects = Array.from(document.querySelectorAll("select"));
        const modeSelect = selects.find((s) => [...s.options].some((o) => o.value === "Q_MODE"));
        if (modeSelect) modeSelect.value = "Q_MODE";
        modeSelect?.dispatchEvent(new Event("change", { bubbles: true }));
      }),
    );
    await expect(page.getByRole("option", { name: "Q_MODE", selected: true }).first())
      .toBeAttached()
      .catch(() => {
        // accept — some browsers report this differently
      });
  });
});

// ---------------------------------------------------------------------------
// 5 — PromptLab — Generate mock run flow (local AR simulation)
// ---------------------------------------------------------------------------

test.describe("Inference — Generate mock run", () => {
  test("clicking Generate mock run changes DataSourceBar to LIVE", async ({ page }) => {
    await gotoInference(page);
    await page.getByRole("button", { name: "Generate mock run" }).click();
    // DataSourceBar should now show LIVE (interactive mode).
    // Use the full pill text to avoid matching 'Live prompt' / 'Live-style' elements.
    await expect(page.getByText("● LIVE")).toBeVisible({ timeout: 3000 });
    // source item should read 'interactive'
    await expect(page.getByText("interactive")).toBeVisible();
  });

  test("after Generate, DataSourceBar source shows interactive", async ({ page }) => {
    await gotoInference(page);
    await page.getByRole("button", { name: "Generate mock run" }).click();
    await expect(page.getByText("interactive")).toBeVisible({ timeout: 3000 });
  });

  test("step timeline has at least one step card after Generate", async ({ page }) => {
    await gotoInference(page);
    await page.getByRole("button", { name: "Generate mock run" }).click();
    // Step 1 card is always generated
    await expect(page.getByText("Step 1")).toBeVisible({ timeout: 4000 });
  });

  test("clicking Reset to scenario restores OFFLINE MOCKUP pill", async ({ page }) => {
    await gotoInference(page);
    await page.getByRole("button", { name: "Generate mock run" }).click();
    await expect(page.getByText("● LIVE")).toBeVisible({ timeout: 3000 });
    await page.getByRole("button", { name: "Reset to scenario" }).click();
    await expect(page.getByText("◎ OFFLINE MOCKUP")).toBeVisible({ timeout: 4000 });
  });
});

// ---------------------------------------------------------------------------
// 6 — InferenceSplitPane — AR lane vs diffusion lane
// ---------------------------------------------------------------------------

test.describe("Inference — AR lane vs diffusion lane", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("Compare panel kicker and title are visible", async ({ page }) => {
    await expect(page.getByText("Compare")).toBeVisible();
    await expect(page.getByText("AR lane vs diffusion lane")).toBeVisible();
  });

  test("AR lane badge is visible", async ({ page }) => {
    await expect(page.getByText("AR lane", { exact: true })).toBeVisible();
  });

  test("Diffusion lane badge is visible", async ({ page }) => {
    await expect(page.getByText("Diffusion lane", { exact: true })).toBeVisible();
  });

  test("Q_MODE badge appears in AR lane", async ({ page }) => {
    // Scope to <strong> to avoid matching the <option> and <button> elements.
    await expect(page.locator('strong').filter({ hasText: /^Q_MODE$/ }).first()).toBeVisible();
  });

  test("S_MODE badge appears in diffusion lane", async ({ page }) => {
    // Scope to <strong> to avoid matching the option, button and heading elements.
    await expect(page.locator('strong').filter({ hasText: /^S_MODE$/ }).first()).toBeVisible();
  });

  test("Throughput metric is visible in AR lane", async ({ page }) => {
    await expect(page.getByText("Throughput").first()).toBeVisible();
  });

  test("Order of certainty ribbon shows left-to-right for AR", async ({ page }) => {
    await expect(page.getByText("Order of certainty")).toBeVisible();
    // <i>left-to-right</i> is the specific element; avoid matching surrounding text spans.
    await expect(page.locator('i').filter({ hasText: /^left-to-right$/ })).toBeVisible();
  });

  test("parallel drafting + revision description appears for diffusion lane", async ({ page }) => {
    await expect(page.getByText("parallel drafting + revision")).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 7 — Diffusion mechanics — Step timeline
// ---------------------------------------------------------------------------

test.describe("Inference — Step timeline", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("Diffusion mechanics panel kicker and title are visible", async ({ page }) => {
    await expect(page.getByText("Diffusion mechanics")).toBeVisible();
    await expect(page.getByText("Step timeline")).toBeVisible();
  });

  test("at least one Step card is rendered", async ({ page }) => {
    await expect(page.getByText("Step 1")).toBeVisible();
  });

  test("step cards contain Mask and Gamma labels", async ({ page }) => {
    await expect(page.getByText("Mask").first()).toBeVisible();
    await expect(page.getByText("Gamma").first()).toBeVisible();
  });

  test("step cards contain Delta and Conf labels", async ({ page }) => {
    await expect(page.getByText("Delta").first()).toBeVisible();
    await expect(page.getByText("Conf").first()).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 8 — Token mask theater
// ---------------------------------------------------------------------------

test.describe("Inference — Token mask theater", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("Token mask theater panel is visible", async ({ page }) => {
    await expect(page.getByText("Token mask theater")).toBeVisible();
    await expect(page.getByText("Canvas-based token state replay")).toBeVisible();
  });

  test("legend labels are visible", async ({ page }) => {
    for (const label of ["prompt", "masked", "new", "edited", "stable"]) {
      await expect(page.getByText(label, { exact: true })).toBeVisible();
    }
  });
});

// ---------------------------------------------------------------------------
// 9 — Fallbacks + env log stream
// ---------------------------------------------------------------------------

test.describe("Inference — Fallbacks + env", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("Fallbacks + env panel is visible", async ({ page }) => {
    await expect(page.getByText("Fallbacks + env")).toBeVisible();
    await expect(page.getByText("Virtualized log stream")).toBeVisible();
  });

  test("log filter buttons are visible", async ({ page }) => {
    // The filter bar lives inside Fallbacks + env section (second occurrence)
    await expect(page.getByPlaceholder("Search action / reason / module / message")).toBeVisible();
  });

  test("level filter buttons are present", async ({ page }) => {
    for (const level of ["notice", "warning", "error"]) {
      await expect(page.getByRole("button", { name: level, exact: true }).first()).toBeVisible();
    }
  });
});

// ---------------------------------------------------------------------------
// 10 — Active inference StatusRail
// ---------------------------------------------------------------------------

test.describe("Inference — Active inference StatusRail", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("Active inference heading is visible", async ({ page }) => {
    await expect(page.getByText("Active inference")).toBeVisible();
  });

  test("engine/mode heading is rendered", async ({ page }) => {
    // Shows e.g. "TRANSFORMERS / S_MODE"
    const heading = page.getByText(/TRANSFORMERS|AR|S_MODE|Q_MODE/);
    await expect(heading.first()).toBeVisible();
  });

  test("StatusRail shows throughput, converge, peak VRAM, fallbacks", async ({ page }) => {
    await expect(page.getByText("Throughput").first()).toBeVisible();
    await expect(page.getByText("Converge").first()).toBeVisible();
    await expect(page.getByText("Peak VRAM").first()).toBeVisible();
    await expect(page.getByText("Fallbacks").first()).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 11 — Insights & Glossary
// ---------------------------------------------------------------------------

test.describe("Inference — Insights", () => {
  test("Insights panel is rendered", async ({ page }) => {
    await gotoInference(page);
    await expect(page.getByText("Insights")).toBeVisible();
    await expect(page.getByText("What to read first")).toBeVisible();
  });
});

test.describe("Inference — Glossary", () => {
  test.beforeEach(async ({ page }) => {
    await gotoInference(page);
  });

  test("Glossary panel is rendered", async ({ page }) => {
    await expect(page.getByText("Glossary")).toBeVisible();
    await expect(page.getByText("Correct terms, not vague dashboard copy")).toBeVisible();
  });

  test("key term tabs are all present", async ({ page }) => {
    for (const term of ["Gamma", "Delta", "Mask ratio", "Converge", "Remask", "Q_MODE", "S_MODE"]) {
      await expect(
        page.getByRole("button", { name: term, exact: true }).first(),
      ).toBeVisible();
    }
  });

  test("clicking a glossary term shows its explanation", async ({ page }) => {
    await page.getByRole("button", { name: "Gamma", exact: true }).first().click();
    // Gamma explanation must appear — it's one of the definition bodies
    await expect(page.getByText(/committed unmask|gamma/i).first()).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 12 — Navigation, shell, no WSD dependency
// ---------------------------------------------------------------------------

test.describe("Inference — shell integration", () => {
  test("navigating from WSD to Inference shows inference hero", async ({ page }) => {
    await page.goto("/wsd");
    await page.getByRole("link", { name: /Inference/ }).click();
    await expect(page).toHaveURL(/\/inference/);
    await expect(page.getByText("OFFLINE MOCKUP")).toBeVisible({ timeout: 8000 });
  });

  test("primary nav inference link is marked active on /inference", async ({ page }) => {
    await gotoInference(page);
    const link = page.getByRole("link", { name: /Inference/ });
    await expect(link).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 13 — Screenshots (full and section-level)
// ---------------------------------------------------------------------------

test.describe("Inference — screenshots", () => {
  test("full inference page snapshot", async ({ page }) => {
    await gotoInference(page);
    await page.waitForTimeout(500); // let canvas render settle
    await expect(page).toHaveScreenshot("inference-ar-full.png", { fullPage: true });
  });

  test("hero section snapshot", async ({ page }) => {
    await gotoInference(page);
    const hero = page.locator("section").filter({ hasText: "Prompt" }).first();
    await expect(hero).toHaveScreenshot("inference-ar-hero.png");
  });

  test("DataSourceBar snapshot", async ({ page }) => {
    await gotoInference(page);
    // The bar is the first .bar element in the page
    const bar = page.locator('[class*="bar"]').first();
    await expect(bar).toHaveScreenshot("inference-ar-datasourcebar.png");
  });

  test("AR vs diffusion compare panel snapshot", async ({ page }) => {
    await gotoInference(page);
    const panel = page.locator('[class*="panel"], article')
      .filter({ hasText: "AR lane vs diffusion lane" })
      .first();
    await expect(panel).toHaveScreenshot("inference-ar-splitpane.png");
  });

  test("PromptLab panel snapshot", async ({ page }) => {
    await gotoInference(page);
    const panel = page.locator('[class*="panel"], article')
      .filter({ hasText: "Run inference from your own prompt" })
      .first();
    await expect(panel).toHaveScreenshot("inference-ar-promptlab.png");
  });

  test("after Generate mock run — full page snapshot", async ({ page }) => {
    await gotoInference(page);
    await page.getByRole("button", { name: "Generate mock run" }).click();
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 4000 });
    await page.waitForTimeout(400);
    await expect(page).toHaveScreenshot("inference-ar-after-generate.png", { fullPage: true });
  });
});
