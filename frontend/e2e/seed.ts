import { expect, type Page } from "@playwright/test";
import path from "node:path";

/** Seeds one real account with a project, a document, a completed research run, extracted claims and a report, all through the UI. */
export async function seed(page: Page): Promise<{ runId: string; reportUrl: string }> {
  const nav = page.getByRole("navigation", { name: "Main" });
  await page.goto("/app/projects");
  await page.getByRole("button", { name: "Create an account" }).click();
  await page.getByLabel("Email").fill(`seed-${Date.now()}@example.test`);
  await page.getByLabel("Password").fill("Orchid-Lantern-Quartz-2026");
  await page.getByRole("button", { name: "Create account" }).click();
  await page.getByRole("button", { name: "New project" }).click();
  await page.getByLabel("Title").fill("Polymer electrolytes");
  await page.getByLabel("Research question").fill("How does temperature affect polymer electrolyte conductivity?");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: "Polymer electrolytes" })).toBeVisible();
  await nav.getByRole("link", { name: "Literature Library", exact: true }).click();
  await page.getByRole("button", { name: "Upload document" }).first().click();
  await page.getByLabel(/File \(PDF/).setInputFiles(path.join(process.cwd(), "e2e", "fixtures", "polymer-note.txt"));
  await page.getByLabel("Source (required)").fill("fixture");
  await page.getByLabel("License (required)").fill("CC0-1.0");
  await page.getByLabel("Title").fill("Polymer electrolytes note");
  await page.getByLabel("Authors (separate with ;)").fill("A. Author");
  await page.getByLabel("Publication date (YYYY-MM-DD)").fill("2020-05-01");
  await page.getByRole("dialog").getByRole("button", { name: "Upload" }).click();
  await expect(page.getByText("Uploaded and indexed.")).toBeVisible();
  await page.getByRole("dialog").getByRole("button", { name: "Close", exact: true }).click();
  await nav.getByRole("link", { name: "Research Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Start research" }).click();
  await expect(page.getByText("Completed").first()).toBeVisible({ timeout: 90_000 });
  const runId = page.url().split("/").pop()!;
  await page.getByRole("button", { name: "Generate report" }).first().click();
  await expect(page.getByText("Report generated.").first()).toBeVisible();
  await nav.getByRole("link", { name: "Evidence Verification", exact: true }).click();
  await page.getByText("Add or extract claims").click();
  await page.getByLabel("Extract claims from a run").selectOption({ index: 1 });
  await page.getByRole("button", { name: "Extract and verify" }).click();
  await expect(page.getByRole("list", { name: "Claims" }).locator("button").first()).toBeVisible();
  await page.goto("/app/reports");
  await page.getByRole("link", { name: /Research report/ }).first().click();
  await expect(page.getByRole("heading", { name: /^1\. / })).toBeVisible();
  return { runId, reportUrl: page.url().replace(/.*\/app/, "") };
}

export const ROUTES = (runId: string, reportUrl: string): Record<string, string> => ({
  overview: "/", projects: "/projects", workspace: `/workspace/${runId}`, "workspace-new": "/workspace", agents: `/agents/${runId}`, library: "/library",
  gaps: "/gaps", hypotheses: "/hypotheses", experiments: "/experiments", evidence: "/evidence", novelty: "/novelty", reports: "/reports", report: reportUrl,
  "model-lab": "/model-lab", activity: "/activity", settings: "/settings",
});
