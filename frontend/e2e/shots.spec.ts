import { test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { ROUTES, seed } from "./seed";

// Visual QA helper (not part of the regression run): seeds real data, then screenshots every page at desktop, tablet and mobile
// widths and records horizontal overflow. Run: SHOTS_DIR=F:/temp/shots/x npx playwright test shots
const OUT = process.env.SHOTS_DIR ?? "";
const SIZES = { desktop: { width: 1440, height: 900 }, tablet: { width: 820, height: 1100 }, mobile: { width: 390, height: 844 } };
test.skip(!process.env.SHOTS_DIR, "set SHOTS_DIR to capture screenshots");
test.setTimeout(300_000);

test("capture pages", async ({ page }) => {
  fs.mkdirSync(OUT, { recursive: true });
  const { runId, reportUrl } = await seed(page);
  const pages = ROUTES(runId, reportUrl);
  for (const [size, vp] of Object.entries(SIZES)) {
    await page.setViewportSize(vp);
    for (const [name, url] of Object.entries(pages)) {
      await page.goto(`/app${url}`);
      await page.waitForLoadState("networkidle");
      await page.waitForTimeout(400);
      await page.screenshot({ path: path.join(OUT, `${size}-${name}.png`), fullPage: size !== "desktop" });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      fs.appendFileSync(path.join(OUT, "overflow.txt"), `${size} ${name} overflow=${overflow}\n`);
    }
  }
});
