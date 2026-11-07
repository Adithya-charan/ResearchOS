# UI design system

A quiet, dense, evidence-first interface for reading and checking research. Tokens live as CSS variables in `frontend/src/index.css` and are
exposed to Tailwind in `frontend/tailwind.config.js`; light and dark share one set of utilities (`data-theme="light|dark"`, default follows the
OS, the choice is stored in `localStorage` under `researchos.theme`).

## Tokens

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | #F7F8FA | #101418 | page |
| `--surface` | #FFFFFF | #161B21 | cards, inputs |
| `--surface-2` | #F1F3F5 | #1C2229 | wells, code, table head |
| `--side` | #F1F3F5 | #12171C | sidebar |
| `--ink` | #17202C | #E6EAF0 | text |
| `--muted` | #5D6A7C | #93A0B2 | secondary text |
| `--line` / `--line-strong` | #E4E8ED / #CFD6DE | #28303A / #3A4552 | hairlines / control borders |
| `--accent` / `--accent-soft` | #2457A6 / #E8EEF8 | #7AA5E6 / #1D2A3D | actions, selection |
| `--on-accent` | #FFFFFF | #0B1220 | text on accent |
| `--ok` `--warn` `--bad` | #26734D #A66A19 #B54747 | #5FBF8A #D9A25B #E08585 | status |
| `--violet` | #5B5BA6 | #A5A5E6 | agent/plan identifiers |
| `--hl` | #FDF1C7 | #4A4220 | search/term highlight |

Deviation from the suggested palette: `--muted` is #5D6A7C rather than #647184 because the lighter value failed WCAG AA on `--surface-2`. The
dark accent is lighter and `--on-accent` is dark so primary buttons keep contrast (an axe finding fixed during QA).

## Type

* UI: Inter (variable). Long-form reading and quoted evidence: Source Serif 4. Identifiers, hashes, ids: JetBrains Mono. All self-hosted through
  `@fontsource-variable/*` because the CSP forbids external fonts.
* Scale is compact: 12 / 13 / 14 px (`xs`/`sm`/`base`). Page title 18-20 px semibold. Long-form uses `.reading` (serif, max 68ch, size from
  `--read-size`, adjustable A-/A+ in Reports and remembered per browser).

## Shape, spacing, elevation

* Radii 6 px (controls, badges) and 8 px (cards). No shadows; hierarchy comes from 1 px borders and surface steps. No gradients, glass or glow.
* Controls are 32 px (`h-8`) or 28 px (`h-7` small). Cards use 16 px padding; page gutters 24 px desktop, 16 px mobile.
* Motion: only opacity/colour transitions of 100-150 ms; everything is disabled under `prefers-reduced-motion`.

## Components (`frontend/src/components/ui.tsx`)

`Card`, `CardTitle`, `PageHeader`, `Button` (primary / default / ghost / outlined danger; `busy` state), `Field`/`Input`/`Select`/`Textarea`,
`Badge`, `StatusBadge`, `Notice`, `EmptyState`, `Modal` and `ConfirmDialog` (Radix Dialog, focus trapped), `Sheet` (right-hand panel),
`Tabs` and `Segmented` (roving keyboard), `Table` (sticky head, `aria-sort` on sortable columns), `KV`, `Section` (collapsible), `Highlight`
(regex-escaped term marking), `Mono`, `ToastProvider`/`useToast` (polite live region). Shell: `components/Shell.tsx`. Data wrappers: `common.tsx`
(`Q` renders loading / error / empty states for a query; `RequireProject`).

## Patterns

* **Status is never colour alone**: every badge carries an icon and a word.
* **Evidence-first**: a claim always shows its status, its supporting / contradicting / contextual passages with serif quotes, document metadata and
  a resolvable citation; a status explains that it is lexical agreement, not proof.
* **Honest empty states**: "nothing has been filled in", "no results exist for them"; no placeholder numbers.
* **Density**: lists are rows, not cards; secondary detail moves into a right panel or collapsible section.

## Reference to implementation

Only pages that were actually read are listed as sources; principles were adopted, no branding, wording or layouts were copied.

| Reference (what was read) | Principle taken | Where it is implemented |
|---|---|---|
| Linear, "A design refresh" article (full text) | quiet chrome, content over navigation, fewer saturated colours, consistent headers, subtle borders, density | dimmed sidebar with hairline, 48 px top bar, `PageHeader`, no shadows, muted accent, compact controls |
| linear.app home (opened; not analysed in depth) | none claimed | none |
| Elicit home (text) | evidence-oriented presentation, citations adjacent to claims, structured report | Evidence three-pane layout, citation chips and panel in Reports, "citation resolves" badges, 18 fixed report sections |
| Readwise Reader landing (text) | comfortable long-form reading, adjustable type, highlights | `.reading` serif column, A-/A+ control, `mark.hl`, passage preview in Library |
| Semantic Scholar home and a results URL | **results page did not render when fetched (stayed on "Loading results"), so its result layout was NOT studied.** Only the home page's search-first framing informed the large search box | Library search panel: filters, removable chips, title / authors / year / DOI / snippet / citation line were designed from general academic-listing conventions, not copied from that page |

## Verification

`npm run build`, `tsc --noEmit`, `eslint .` (0 warnings), `vitest` (31 tests), Playwright full flow and a WCAG 2 A/AA axe scan of all 16 routes in light
and dark. Automated axe covers only part of accessibility; keyboard and screen-reader passes remain manual (see `FRONTEND_TESTING.md`).
