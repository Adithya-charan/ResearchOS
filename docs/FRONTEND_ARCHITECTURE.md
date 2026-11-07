# Frontend architecture

Single-page app in `frontend/`: Vite 5, React 18, TypeScript (strict), react-router (basename `/app`), TanStack Query v5, react-hook-form + zod,
Tailwind 3 on CSS-variable tokens, Recharts (charts), `@xyflow/react` (agent graph), Radix Dialog, lucide-react, react-markdown + rehype-sanitize.
No hosted AI services, no analytics, no external fonts or scripts.

## Serving and security

`npm run build` writes `frontend/dist`; FastAPI serves it at `/app` (same origin, so the HttpOnly session cookie and CSRF flow are unchanged) with
`default-src 'self'`, `frame-ancestors 'none'`, no inline script or style. Development: `npm run dev` proxies the API. The CSRF token is held in
memory; no token is ever written to browser storage (only theme and reading size are).

## Layout

```
frontend/
  src/main.tsx, App.tsx      providers (Query, Auth, Toast), routes
  src/lib/                   api.ts (typed client), sse.ts, useRunStream.ts, useAgentOutput.ts, project.tsx (selected project), types.ts, format.ts
  src/components/            ui.tsx (design system), Shell.tsx, common.tsx, AgentOutputView.tsx, Markdown.tsx
  src/pages/                 one file per route (16 views)
  src/index.css              tokens, reading styles, print and reduced-motion rules
  e2e/                       Playwright (real backend), seed.ts, a11y.spec.ts, shots.spec.ts, serve.py
```

## Shell

Collapsible sidebar (tooltips when collapsed, mobile drawer), 48 px top bar with breadcrumbs, `/` to focus search, project selector,
pending-approvals bell, theme toggle and user menu, skip link. Routes: Overview, Projects, Research Workspace, Agent Command Center, Library, Gaps,
Hypotheses, Experiments, Evidence, Novelty, Reports, Model Lab, Activity, Settings, Login.

## State

* Server state: TanStack Query, keys scoped by project / run. Mutations invalidate the affected keys (`overview`, `approvals`, ...).
* The selected project lives in `lib/project.tsx` (URL-independent, remembered). Pages needing one use `RequireProject`.
* Live data: `useRunStream` consumes `/runs/{id}/stream` and merges events into the cached run; with no active run the UI says "not streaming".
* Derived data: gaps, hypotheses and experiment plans are read from agent task outputs via `useAgentOutput(projectId, agentType, fixedRunId?)`.
  They are not stored entities, and the pages say so.

## Page structure

| Page | Structure |
|---|---|
| Overview | metric strip, runs, evidence verification bar, charts (with sr-only tables), activity, recent reports/projects |
| Workspace | main column (progress, collapsible sections, tasks, events) + sticky aside (status, tasks, approvals, warnings) |
| Agents | dependency graph (fixed-size nodes, vertical layout) + task details panel (role, status, dependencies, duration, input, output, errors, evidence refs) |
| Library | search panel (level, phrase, filters, chips), result rows with citation line, sortable document table, preview sheet |
| Evidence | claims / selected claim / evidence three-pane layout |
| Reports | section nav with scroll-spy, serif reading column, citation panel |

## Limits

Not implemented because the API has no matching capability: password change, persisted gap/hypothesis entities, experiment creation, starting
model training. See `FRONTEND_API_INTEGRATION.md`.
