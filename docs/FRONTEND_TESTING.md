# Frontend testing

Run from `frontend/` (`npm ci` first).

| Layer | Command | What it covers |
|---|---|---|
| Types | `npx tsc --noEmit` | strict TypeScript |
| Lint | `npx eslint .` | typescript-eslint, react-hooks (0 warnings) |
| Build | `npm run build` | production bundle in `dist/` |
| Unit / component | `npx vitest run` (31 tests) | API client (CSRF on mutations only, in-memory token, GET retry once, POST never, 401 listener), SSE parser (dedupe, resume, end, 4xx), UI primitives (badges, highlight, tabs, confirm dialog, toasts, markdown sanitising), page logic (zod schemas, upload validation, stage derivation, graph layout incl. cycles, citation labels) |
| End to end | `npx playwright test full-flow` | real FastAPI backend (`e2e/serve.py`, isolated data dir, port 8793): register, project, upload, search, research run, agents, gaps, hypotheses, evidence, report + export response, no tokens in storage, sign out |
| Accessibility | `npx playwright test a11y` | axe WCAG 2 A/AA/21AA on all 16 routes, light and dark, seeded real data; fails on serious/critical |
| Keyboard / focus (P10) | `npx playwright test keyboard` | skip link is the first tab stop; every tab stop on all 16 routes is visible and shows a focus ring; `/` focuses search; dialogs trap focus and return it on Escape; axe scan at 390 px width in light and dark on six pages |
| Page load (P10, measurement only) | `PERF_OUT=<file.json> npx playwright test perf` | cold-cache load of overview, library and agents graph: request count, DOMContentLoaded, load, LCP, JS/CSS/font wire size (not a pass/fail test; skipped without `PERF_OUT`) |
| Screenshots | `SHOTS_DIR=<dir> npx playwright test shots` | desktop 1440, tablet 820, mobile 390 of every page, plus a horizontal-overflow check (not part of the regression run) |
| Backend | `python -m pytest` and `tests/test_p9_web.py` | `/overview`, `/audit`, `/app` (shell, CSP, fallback, 404 assets, traversal) |

Playwright uses the installed Microsoft Edge (`channel: "msedge"`), so no browser download is needed locally. In CI (`PW_CHANNEL=""`, `PW_PYTHON=python`) it uses
the Chromium that `npx playwright install --with-deps chromium` downloads (a browser, not a dataset). Normal CI downloads no datasets and trains nothing.

## Manual checks still required

Automated axe finds only part of the problems. P10 added an automated keyboard/focus pass (above); a *human* keyboard walkthrough has not been done.
Not covered and **not performed**: screen-reader announcements (toasts, live run status; no screen reader was available), zoom/reflow at 200-400 %,
Safari/Firefox rendering, and real touch devices.

P10 accessibility fix: React Flow (agent graph) made every edge a tab stop (100+ invisible stops) and gave nodes no visible focus ring. Edges are no longer
focusable (`edgesFocusable={false}`) and focused nodes show the accent outline (`src/index.css`).

## Known test gaps

No component tests for Workspace, Evidence, Library and Reports beyond the e2e path; no tests that stub failing endpoints per page; `npm audit`
warnings were reviewed in P10 (`SECURITY.md`); no stubbed-failure tests per page.
