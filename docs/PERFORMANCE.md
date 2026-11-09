# Performance (P10)

Every number here was measured on the machine below by the scripts in this repository, and the raw output is committed-ready in `artifacts/evaluation/`
(`api_bench.json`, `api_bench_postgres.json`, `page_load.json`, `database_bench.json` from P7). Nothing is estimated or taken from another system.

## Environment and limits (read first)

| | |
|---|---|
| Machine | Windows 11, Intel 20 logical CPUs, Python 3.11.0, local SSD |
| Topology | client, API and database on **one machine over loopback**; no network latency, no TLS, no reverse proxy |
| Database | SQLite files (default) and a local PostgreSQL 16 (pgserver, `127.0.0.1`); no pooler |
| Data | **synthetic generated text** (about 3 KB per document), not real literature; 25, 100 and 250 documents per library |
| Rate limits | switched off for the run (the benchmark would otherwise measure the limiter) |
| Repetition | one run per configuration, run on an otherwise idle machine after an earlier overlapping run was discarded (it competed with builds and tests for CPU and is not reported). Treat differences under about 30 % as noise |
| Not measured | Docker/container overhead (Docker unavailable), multi-host, real network, GPU/model inference throughput (see P8 reports), large real corpora, more than 8 concurrent users, long-duration soak tests |

Reproduce: `python scripts/bench_api.py --sizes 25,100,250` (add `--postgres <url>` for PostgreSQL), `PERF_OUT=file.json npx playwright test perf` in `frontend/`.

## API latency (200 requests each, one authenticated client)

| Endpoint | SQLite p50 / p95 (ms) | PostgreSQL p50 / p95 (ms) |
|---|---|---|
| `GET /health` | 3.6 / 4.9 | 3.9 / 5.9 |
| `GET /auth/me` | 6.7 / 9.3 | 9.0 / 12.6 |
| `GET /projects` | 8.6 / 13.8 | 11.7 / 15.1 |
| `GET /overview` | 14.5 / 25.3 | 23.6 / 46.4 |
| `GET /audit` | 9.2 / 14.6 | 15.6 / 25.2 |

PostgreSQL is slightly slower here because every request is a loopback round trip to a separate server, where SQLite is an in-process file read. This says
nothing about PostgreSQL under concurrent write load, which is why it is the production choice.

## Upload and search against library size

| Documents | Upload p50 / p95 (ms), SQLite | Upload p50 / p95 (ms), PostgreSQL | Chunk search p50 / p95 (ms) SQLite | Chunk search PostgreSQL | Document search p50 / p95, SQLite |
|---|---|---|---|---|---|
| 25 | 351 / 447 | 467 / 1072 | 12 / 28 | 20 / 61 | 13 / 24 |
| 100 | 354 / 512 | 371 / 495 | 19 / 81 | 21 / 80 | 25 / 78 |
| 250 | 369 / 485 | 347 / 485 | 24 / 171 | 30 / 179 | 41 / 179 |

* Upload is dominated by **isolated parsing**: each file is parsed in a separate, time-limited, memory-limited subprocess (a deliberate security control,
  `SECURITY.md`), so a fixed process-start cost is paid per file. It does not grow with library size. The PostgreSQL 25-document p95 of 1.07 s is a single slow
  outlier in a 25-sample set (first requests after start-up).
* Search p50 grows with the library (about 2x from 25 to 250 documents on this synthetic corpus). Tail latency (p95) grows faster; the first search after an
  upload may rebuild the in-memory index, which shows in p95/max. Phrase queries are the slowest search type.
* Library sizes beyond 250 documents of 3 KB were not measured; real papers are larger, so real indexing and search cost per document will be higher.

## Concurrent users (search, SQLite / PostgreSQL)

| Users | Requests | p50 / p95 (ms), SQLite | p50 / p95 (ms), PostgreSQL | Errors |
|---|---|---|---|---|
| 1 | 40 | 14 / 27 | 18 / 29 | 0 |
| 4 | 160 | 50 / 71 | 49 / 75 | 0 |
| 8 | 320 | 66 / 145 | 120 / 185 | 0 |

Latency rises roughly linearly with users because one API process serves them (CPU-bound searching in Python threads). Scale-out is by running more API
processes behind the proxy; note the in-process rate limiter then applies per process (`SECURITY.md`).

## Worker, queue and SSE

One research run (11 tasks, synthetic corpus, `completed`):

| | SQLite | PostgreSQL |
|---|---|---|
| Queue latency (create to first task started) | 0.92 s | 1.20 s |
| Completion | 1.59 s | 1.75 s |
| SSE events delivered / ended cleanly | 97 / yes | 98 / yes |
| SSE first event after create / largest gap | 0.48 s / 0.53 s | 0.59 s / 0.54 s |

Queue latency includes the worker's poll interval and job claim. Only a single run was measured; **multi-worker throughput and many simultaneous runs were not
benchmarked.** Run and SSE stability under reconnection is covered by the functional tests, not by this benchmark.

## Database and pool

P7's `scripts/bench_database.py` (`artifacts/evaluation/database_bench.json`) measures SQLite vs PostgreSQL query paths and pool behaviour and was not re-run in P10;
the numbers above (API over a real PostgreSQL) are the P10 database evidence. The pool defaults are 10 connections plus 20 overflow per process
(`RESEARCHOS_DB_POOL_SIZE`, `RESEARCHOS_DB_MAX_OVERFLOW`); pool exhaustion under load was not provoked.

## Front-end build and page load

Production build (`npm run build`): `dist/` about 1.7 MB on disk; largest JS chunks (raw / gzip): Overview 380 / 103 KB (contains the charts library),
flow (agent graph) 176 / 57 KB, index 163 / 47 KB, react 157 / 51 KB, markdown 125 / 39 KB, query 47 / 14 KB. Routes are code-split.

Cold-cache initial load in Edge against the real backend on loopback (`page_load.json`; a single load per page, so indicative only):

| Page | Requests | DOMContentLoaded | LCP | JS wire (decoded) | Total wire |
|---|---|---|---|---|---|
| Overview | 19 | 88 ms | 504 ms | 267 KB (902 KB) | 366 KB |
| Library | 16 | 122 ms | 348 ms | 117 KB (378 KB) | 214 KB |
| Agents + graph | 21 | 109 ms | 344 ms | 171 KB (543 KB) | 286 KB |

Loopback hides network cost: on a real connection the wire sizes matter more than the millisecond figures. No Lighthouse score was produced.

## Optimisations made in P10 and their measured effect

| Change | Measured effect | Evidence |
|---|---|---|
| `research.bm25` phrase check: cheap lowercase-substring pre-filter before the tokenised match | phrase search over 3000 passages p50 145 -> 110 ms; results identical to the old code (0 mismatches) | A/B on the same data, same session |
| Corpus package: lazy imports; extractors imported only when the file type needs them | child-process start for a text file about 375 -> 195 ms measured in isolation; end-to-end HTTP upload remains about 350 ms (parse, store and index costs remain), so the end-to-end gain is smaller than the isolated figure | isolated timing of the child start; `api_bench.json` |
| Charts library no longer a shared chunk (it was pre-loaded on every page) | JS pre-loaded on first paint about 752 KB -> 366 KB raw; the charts code now loads only with the Overview | build output before/after (`modulepreload` list) |
| gzip for the SPA's JS/CSS/HTML/JSON/SVG (cached by path and mtime), `immutable` caching for fingerprinted assets | JS wire size roughly 30 % of decoded (267 of 902 KB on the Overview); the `immutable` header is tested, but repeat-visit behaviour in a browser was not measured | `page_load.json`; `tests/test_p9_web.py` |

## Bottlenecks that remain (not fixed)

1. Per-file subprocess start for isolated parsing (about 190 ms floor on this machine). Safe but costly for bulk imports; a persistent sandboxed worker pool would amortise it and was judged out of scope for hardening.
2. Search aggregates all passages for document-level results and rebuilds the in-memory index after changes; p95 grows with library size.
3. A single API process serves searches in Python threads; concurrency scales by adding processes.
4. Nothing here covers real-literature scale, model inference, or Docker.
