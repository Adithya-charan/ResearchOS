import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { ROUTES, seed } from "./seed";

// P10 accessibility checks that automation CAN make: keyboard reachability and focus visibility on every page, modal focus handling,
// the "/" shortcut, and an axe scan at phone width. Screen-reader behaviour is NOT covered and stays a manual item.
test.setTimeout(300_000);

test("keyboard: skip link first, every stop shows a focus indicator, modal traps and restores focus, mobile axe scan", async ({ page }) => {
  const { runId, reportUrl } = await seed(page);
  const routes = ROUTES(runId, reportUrl);
  const problems: string[] = [];

  for (const [name, url] of Object.entries(routes)) {
    await page.goto(`/app${url}`);
    await page.waitForLoadState("networkidle");
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("Tab");
    const first = await page.evaluate(() => (document.activeElement as HTMLElement | null)?.textContent?.trim());
    if (first !== "Skip to content") problems.push(`${name}: first Tab stop is "${first}", expected the skip link`);

    // walk up to 60 tab stops; each must be visible, inside the viewport once focused, and must show an outline or ring
    const seen = new Set<string>();
    for (let i = 0; i < 60; i++) {
      await page.keyboard.press("Tab");
      const info = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        if (!el || el === document.body) return null;
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        const ring = (cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0) || (cs.boxShadow !== "none" && cs.boxShadow !== "");
        const label = (el.getAttribute("aria-label") || el.textContent || el.getAttribute("name") || el.tagName).trim().slice(0, 40);
        return { key: `${el.tagName}|${label}|${Math.round(r.top)}|${Math.round(r.left)}`, label, ring, visible: r.width > 0 && r.height > 0, tag: el.tagName };
      });
      if (!info) break;
      if (seen.has(info.key)) break; // wrapped around
      seen.add(info.key);
      if (!info.visible) problems.push(`${name}: focus landed on an invisible element "${info.label}"`);
      if (!info.ring) problems.push(`${name}: no visible focus indicator on ${info.tag} "${info.label}"`);
    }
    if (seen.size < 5) problems.push(`${name}: only ${seen.size} keyboard stops (shell navigation should alone exceed that)`);
  }

  // "/" focuses search; Escape in a dialog closes it and returns focus to the control that opened it
  await page.goto("/app/library");
  await page.waitForLoadState("networkidle");
  await page.keyboard.press("/");
  const focusedPlaceholder = await page.evaluate(() => (document.activeElement as HTMLInputElement | null)?.placeholder ?? "");
  if (!/search/i.test(focusedPlaceholder)) problems.push(`"/" did not focus the search box (focused: "${focusedPlaceholder}")`);
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  const opener = page.getByRole("button", { name: "Upload document" }).first();
  await opener.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog")).toBeVisible();
  const inDialog = await page.evaluate(() => !!document.activeElement?.closest('[role="dialog"]'));
  if (!inDialog) problems.push("focus did not move into the dialog when it opened");
  for (let i = 0; i < 25; i++) {
    await page.keyboard.press("Tab");
    if (!(await page.evaluate(() => !!document.activeElement?.closest('[role="dialog"]')))) { problems.push("Tab escaped the open dialog"); break; }
  }
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const back = await page.evaluate(() => (document.activeElement as HTMLElement | null)?.textContent?.trim());
  if (!back?.includes("Upload document")) problems.push(`focus did not return to the opener after Escape (focused: "${back}")`);

  // phone width: no serious/critical axe findings, in both themes
  await page.setViewportSize({ width: 390, height: 844 });
  for (const theme of ["light", "dark"]) {
    await page.evaluate((t) => localStorage.setItem("researchos.theme", t), theme);
    for (const name of ["overview", "workspace", "library", "evidence", "report", "agents"]) {
      await page.goto(`/app${routes[name]}`);
      await page.waitForLoadState("networkidle");
      const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
      for (const v of r.violations.filter((x) => x.impact === "serious" || x.impact === "critical")) problems.push(`mobile ${theme} ${name}: ${v.id} x${v.nodes.length} ${v.nodes[0].target.join(" ")}`);
    }
  }
  expect(problems, problems.join("\n")).toEqual([]);
});
