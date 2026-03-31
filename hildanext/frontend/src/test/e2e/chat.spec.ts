import { expect, test } from "@playwright/test";

test.describe("chat-first inference studio", () => {
  test("runs AR and BOTH flows and persists after refresh", async ({ page }) => {
    await page.route("**/api/health", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "ok", model_loaded: false }),
      });
    });

    await page.route("**/api/generate/ar", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          text: "AR risposta",
          engine: "ar-greedy",
          stats: { tokens_per_sec: 50, tokens_generated: 12, dummy_model: false },
        }),
      });
    });

    await page.route("**/api/generate", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          text: "dLLM risposta",
          engine: "transformers",
          stats: { tokens_per_sec: 35, steps_to_converge: 4, dummy_model: false },
        }),
      });
    });

    await page.goto("/chat");
    await page.getByRole("button", { name: "Carica pesi modello" }).click();
    await page.getByPlaceholder("Scrivi prompt... Ctrl+Invio per inviare").fill("ciao mondo");
    await page.getByRole("button", { name: "Invia" }).click();
    await expect(page.getByText("AR risposta")).toBeVisible();
    await expect(page.getByText("dLLM risposta")).toBeVisible();

    await page.reload();
    await expect(page.getByText("ciao mondo").first()).toBeVisible();

    await page.getByRole("button", { name: "Carica pesi modello" }).click();
    await page.getByLabel("Engine mode").selectOption("AR");
    await page.getByPlaceholder("Scrivi prompt... Ctrl+Invio per inviare").fill("solo ar");
    await page.getByRole("button", { name: "Invia" }).click();
    await expect(page.getByText("solo ar").first()).toBeVisible();
  });

  test("legacy wsd route remains reachable", async ({ page }) => {
    await page.goto("/legacy/wsd");
    await expect(page.getByText("Warmup -> stable -> decay")).toBeVisible();
  });

  test("shows lane-specific offline fallback in BOTH mode", async ({ page }) => {
    let arCalls = 0;
    await page.route("**/api/health", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "ok", model_loaded: false }),
      });
    });

    await page.route("**/api/generate/ar", async (route) => {
      arCalls += 1;
      if (arCalls === 1) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            text: "warmup-ar",
            engine: "ar-greedy",
            stats: { tokens_per_sec: 50, tokens_generated: 1, dummy_model: false },
          }),
        });
        return;
      }
      await route.fulfill({
        status: 503,
        contentType: "text/plain",
        body: "backend offline",
      });
    });

    await page.route("**/api/generate", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          text: "dLLM comunque disponibile",
          engine: "transformers",
          stats: { tokens_per_sec: 35, steps_to_converge: 4, dummy_model: false },
        }),
      });
    });

    await page.goto("/chat");
    await page.getByRole("button", { name: "Carica pesi modello" }).click();
    await page.getByPlaceholder("Scrivi prompt... Ctrl+Invio per inviare").fill("test offline");
    await page.getByRole("button", { name: "Invia" }).click();
    await expect(page.getByText("Parziale: backend offline")).toBeVisible();
    await expect(page.getByText("dLLM comunque disponibile")).toBeVisible();
    await expect(page.getByText("Backend offline o non raggiungibile per questa lane.")).toBeVisible();
  });
});
