import { test, expect } from "@playwright/test";

test("root redirects to chat studio", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/chat$/);
  await expect(page.getByText("Chat-First Inference Studio")).toBeVisible();
  await expect(page.getByRole("button", { name: /nuova chat/i })).toBeVisible();
});

test("/inference alias redirects to chat", async ({ page }) => {
  await page.goto("/inference");
  await expect(page).toHaveURL(/\/chat$/);
  await expect(page.getByText("Chat-First Inference Studio")).toBeVisible();
});

test("legacy wsd route remains directly reachable", async ({ page }) => {
  await page.goto("/legacy/wsd");
  await expect(page.getByText("Warmup -> stable -> decay")).toBeVisible();
});
