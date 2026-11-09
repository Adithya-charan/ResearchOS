import fs from "node:fs";
import { test } from "@playwright/test";
import { seed } from "./seed";

// P10 measurement helper (not part of the regression run): cold-cache initial load of the sign-in page and of the Overview against the real backend,
// on loopback. Run: PERF_OUT=path.json npx playwright test perf
test.skip(!process.env.PERF_OUT, "set PERF_OUT to record page-load measurements");
test.setTimeout(300_000);

async function measure(page: import("@playwright/test").Page, url: string, cdp: import("@playwright/test").CDPSession) {
  await cdp.send("Network.clearBrowserCache");
  await page.goto("about:blank");
  await page.goto(url, { waitUntil: "load" });
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(500);
  return page.evaluate(async () => {
    const nav = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming;
    const res = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
    const lcp = await new Promise<number>((resolve) => {
      let v = 0;
      new PerformanceObserver((l) => { for (const e of l.getEntries()) v = e.startTime; }).observe({ type: "largest-contentful-paint", buffered: true });
      setTimeout(() => resolve(v), 300);
    });
    const kinds = (rx: RegExp) => res.filter((r) => rx.test(r.name));
    const sum = (xs: PerformanceResourceTiming[], k: "encodedBodySize" | "decodedBodySize") => xs.reduce((a, r) => a + r[k], 0);
    const js = kinds(/\.js(\?|$)/), css = kinds(/\.css(\?|$)/), fonts = kinds(/\.woff2(\?|$)/);
    return {
      requests: res.length + 1, dom_content_loaded_ms: Math.round(nav.domContentLoadedEventEnd), load_ms: Math.round(nav.loadEventEnd), lcp_ms: Math.round(lcp),
      js: { files: js.length, wire_kb: Math.round(sum(js, "encodedBodySize") / 1024), decoded_kb: Math.round(sum(js, "decodedBodySize") / 1024) },
      css: { files: css.length, wire_kb: Math.round(sum(css, "encodedBodySize") / 1024) },
      fonts: { files: fonts.length, wire_kb: Math.round(sum(fonts, "encodedBodySize") / 1024) },
      total_wire_kb: Math.round((res.reduce((a, r) => a + r.encodedBodySize, 0)) / 1024),
    };
  });
}

test("initial page load, cold cache", async ({ page }) => {
  const out: Record<string, unknown> = { environment: "Edge (msedge channel), loopback, cache cleared before each load, API serves the built SPA (gzip when accepted)" };
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Network.enable");
  const { runId } = await seed(page);
  out.overview = await measure(page, "/app/", cdp);
  out.library = await measure(page, "/app/library", cdp);
  out.agents_with_graph = await measure(page, `/app/agents/${runId}`, cdp);
  fs.writeFileSync(process.env.PERF_OUT!, JSON.stringify(out, null, 2));
});
