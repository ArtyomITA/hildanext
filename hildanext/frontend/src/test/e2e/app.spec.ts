import { test, expect } from "@playwright/test";

test("wsd page loads", async ({ page }) => {
  await page.goto("/wsd");
  await expect(page.getByText("Warmup -> stable -> decay")).toBeVisible();
  await expect(page.getByText("Virtualized log window")).toBeVisible();
});

test("inference page loads", async ({ page }) => {
  await page.goto("/inference");
  await expect(page.getByText("AR lane vs diffusion lane")).toBeVisible();
  await expect(page.getByText("Canvas-based token state replay")).toBeVisible();
});
