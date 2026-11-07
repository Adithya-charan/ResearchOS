import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { ROUTES, seed } from "./seed";

// Automated WCAG 2 A/AA scan of every page, in light and dark themes, on seeded real data.
// Automated checks find only part of the problems; keyboard and screen-reader behaviour are checked by hand (docs/FRONTEND_TESTING.md).
test.setTimeout(300_000);

test("no serious or critical accessibility violations on any page", async ({ page }) => {
  const { runId, reportUrl } = await seed(page);
  const found: string[] = [];
  for (const theme of ["light", "dark"]) {
    await page.evaluate((t) => { localStorage.setItem("researchos.theme", t); }, theme);
    for (const [name, url] of Object.entries(ROUTES(runId, reportUrl))) {
      await page.goto(`/app${url}`);
      await page.waitForLoadState("networkidle");
      await page.waitForTimeout(300);
      const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
      for (const v of r.violations.filter((x) => x.impact === "serious" || x.impact === "critical")) found.push(`${theme} ${name}: ${v.id} (${v.impact}) x${v.nodes.length} — ${v.nodes[0].target.join(" ")}`);
    }
  }
  expect(found, found.join("\n")).toEqual([]);
});
