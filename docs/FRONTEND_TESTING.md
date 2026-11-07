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
| Screenshots | `SHOTS_DIR=<dir> npx playwright test shots` | desktop 1440, tablet 820, mobile 390 of every page, plus a horizontal-overflow check (not part of the regression run) |
| Backend | `python -m pytest` and `tests/test_p9_web.py` | `/overview`, `/audit`, `/app` (shell, CSP, fallback, 404 assets, traversal) |

Playwright uses the installed Microsoft Edge (`channel: "msedge"`), so no browser download is needed. Normal CI downloads no datasets and trains nothing.

## Manual checks still required

Automated axe finds only part of the problems. Not covered by automation: full keyboard walkthrough, screen-reader announcements (toasts, live
run status), zoom/reflow at 200-400 %, Safari/Firefox rendering, and real touch devices. These have not been performed.

## Known test gaps

No component tests for Workspace, Evidence, Library and Reports beyond the e2e path; no tests that stub failing endpoints per page; `npm audit`
warnings have not been reviewed.
