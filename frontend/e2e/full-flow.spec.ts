import { expect, test } from "@playwright/test";
import path from "node:path";

const EMAIL = `e2e-${Date.now()}@example.test`;
const PASSWORD = "Orchid-Lantern-Quartz-2026";

test.use({ acceptDownloads: false });
test("sign up → project → library → research run → agents → evidence → report → sign out", async ({ page }) => {
  // protected route redirects anonymous users to sign-in
  await page.goto("/app/projects");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();

  // register
  await page.getByRole("button", { name: "Create an account" }).click();
  await page.getByLabel("Email").fill(EMAIL);
  await page.getByLabel("Display name (optional)").fill("E2E User");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  // the originally requested protected page is restored after signing up
  await expect(page.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();

  // create a project
  await page.getByRole("button", { name: "New project" }).click();
  await page.getByLabel("Title").fill("Polymer electrolytes");
  await page.getByLabel("Research question").fill("How does temperature affect polymer electrolyte conductivity?");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: "Polymer electrolytes" })).toBeVisible();

  // upload a document
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Literature Library", exact: true }).click();
  await page.getByRole("button", { name: "Upload document" }).first().click();
  await page.getByLabel(/File \(PDF/).setInputFiles(path.join(process.cwd(), "e2e", "fixtures", "polymer-note.txt"));
  await page.getByLabel("Source (required)").fill("e2e fixture");
  await page.getByLabel("License (required)").fill("CC0-1.0");
  await page.getByLabel("Title").fill("Polymer electrolytes note");
  await page.getByRole("dialog").getByRole("button", { name: "Upload" }).click();
  await expect(page.getByText("Uploaded and indexed.")).toBeVisible();
  await page.getByRole("dialog").getByRole("button", { name: "Close", exact: true }).click();
  await expect(page.getByRole("button", { name: "Polymer electrolytes note", exact: true })).toBeVisible();

  // lexical search shows the label and a citation
  await page.getByLabel("Search query").fill("conductivity");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(page.getByText(/BM25 \(lexical\)/)).toBeVisible();
  await expect(page.getByText(/Citation: document doc-/).first()).toBeVisible();

  // start a research run and wait for it to complete
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Research Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Start research" }).click();
  await expect(page.getByRole("heading", { name: "Research run" })).toBeVisible();
  await expect(page.getByText("Completed").first()).toBeVisible({ timeout: 90_000 });

  // agent command center: graph + task table from real data
  await page.getByRole("link", { name: "Agent command center" }).first().click();
  await expect(page.getByRole("region", { name: /Task dependency graph/ })).toBeVisible();
  await expect(page.getByRole("table", { name: "Tasks by agent" })).toBeVisible();

  // gaps and hypotheses derived read-only from agent output
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Research Gaps", exact: true }).click();
  await expect(page.getByText(/accepted document/).first()).toBeVisible();
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Hypotheses", exact: true }).click();
  await expect(page.getByText(/not a finding|untested proposal/i).first()).toBeVisible();

  // evidence verification: extract claims and open one
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Evidence Verification", exact: true }).click();
  await page.getByText("Add or extract claims").click();
  await page.getByLabel("Extract claims from a run").selectOption({ index: 1 });
  await page.getByRole("button", { name: "Extract and verify" }).click();
  const claim = page.getByRole("list", { name: "Claims" }).locator("button").first();
  await expect(claim).toBeVisible();
  await claim.click();
  await expect(page.getByText(/lexical check against the uploaded library/i).first()).toBeVisible();

  // generate and open the report
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Research Workspace", exact: true }).click();
  await page.getByRole("link", { name: /How does temperature/ }).first().click();
  await page.getByRole("button", { name: "Generate report" }).click();
  await expect(page.getByText("Report generated.")).toBeVisible();
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Research Reports", exact: true }).click();
  await page.getByRole("link", { name: /Research report/ }).first().click();
  await expect(page.getByRole("heading", { name: /^1\. / })).toBeVisible();
  // the Markdown export is fetched through the API client (with the session cookie) and saved by the page
  const [exported] = await Promise.all([page.waitForResponse((r) => /\/export\.md$/.test(r.url())), page.getByRole("button", { name: "Markdown", exact: true }).click()]);
  expect(exported.status()).toBe(200);
  expect(await exported.text()).toContain("#");

  // session cookie is HttpOnly: not readable from script; no token in storage
  expect(await page.evaluate(() => document.cookie)).not.toMatch(/session|token/i);
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).not.toMatch(/csrf|token|password/i);

  // sign out returns to sign-in and protected data is gone
  await page.getByRole("button", { name: /account menu|user menu/i }).click();
  await page.getByRole("menuitem", { name: "Sign out" }).click();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  await page.goto("/app/reports");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});
