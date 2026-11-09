"""End-to-end benchmark of the RESEARCHOS API over real HTTP.

    python scripts/bench_api.py [--postgres postgresql://...] [--sizes 25,100,250] [--out artifacts/evaluation/api_bench.json]

It starts the real server (scripts/serve_api.py, embedded worker) in a subprocess with fresh data directories, signs up users over HTTP and
measures: request latency of cheap endpoints, upload and search latency as the literature library grows, concurrent-user throughput,
research-run queue latency and completion time, and SSE delivery. Rate limits are switched off for the run so the benchmark measures the
server and not the limiter (the limiter is tested in tests/test_p10_security.py).

The documents are SYNTHETIC text generated here (no real papers, no scraped data), so absolute search relevance means nothing; the point is
size and timing. Numbers describe THIS machine (see "environment" in the output) with client and server on loopback, one short run each:
they are indicative, not a statistical benchmark, and say nothing about a networked deployment."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PW = "correct horse battery 42"
WORDS = ("polymer electrolyte conductivity temperature membrane lithium battery cathode anode interface stability degradation humidity "
         "encapsulation efficiency perovskite solar cell lifetime spectroscopy impedance diffusion transport ionic mobility thermal "
         "annealing crystallinity morphology substrate coating deposition measurement uncertainty control sample batch replicate").split()


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summary(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": round(statistics.median(ms), 2), "p95_ms": round(pct(ms, 0.95), 2), "max_ms": round(max(ms), 2)}


def synthetic_doc(i: int, rng: random.Random) -> bytes:
    paras = []
    for p in range(10):
        sent = [" ".join(rng.choices(WORDS, k=rng.randint(10, 18))).capitalize() + "." for _ in range(4)]
        paras.append(" ".join(sent))
    return (f"Synthetic benchmark document {i}\n\nAbstract\n\n" + "\n\n".join(paras) + f"\n\nUnique marker doc{i}x\n").encode()


class Client:
    def __init__(self, base: str, email: str):
        self.http = httpx.Client(base_url=base, timeout=120)
        r = self.http.post("/auth/register", json={"email": email, "password": PW})
        r.raise_for_status()
        self.h = {"X-CSRF-Token": r.json()["csrf_token"]}

    def timed(self, method: str, url: str, **kw) -> tuple[float, httpx.Response]:
        t = time.perf_counter()
        r = self.http.request(method, url, headers=self.h, **kw)
        return (time.perf_counter() - t) * 1000, r


def start_server(port: int, work: Path, postgres: str | None) -> subprocess.Popen:
    env = {**os.environ, "RESEARCHOS_DATA_DIR": str(work / "data"), "RESEARCHOS_EMBEDDED_WORKER": "true", "RESEARCHOS_MAX_DOCUMENTS_PER_PROJECT": "2000",
           "RESEARCHOS_MAX_ACTIVE_RUNS_PER_USER": "4", "RESEARCHOS_REGISTER_ATTEMPTS": "1000", "RESEARCHOS_LOG_LEVEL": "warning"}
    for n in ("UPLOAD", "RUN_CREATE", "SEARCH", "VERIFY", "REPORT", "EXPERIMENT", "EVALUATE", "REPROCESS"):
        env[f"RESEARCHOS_RATE_{n}"] = "off"
    if postgres:
        env.update(RESEARCHOS_DATABASE_URL=postgres, RESEARCHOS_DATABASE_SCHEMA="bench_api", RESEARCHOS_ENGINE_DATABASE_SCHEMA="bench_engine")
    else:
        env.update(RESEARCHOS_DATABASE_URL=f"sqlite:///{(work / 'app.db').as_posix()}", RESEARCHOS_ENGINE_DB=str(work / "engine.db"))
    p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "serve_api.py"), "--port", str(port)], env=env, cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("server did not start: " + (p.stderr.read().decode()[-500:] if p.stderr else ""))


def bench(base: str, sizes: list[int]) -> dict:
    out: dict = {}
    c = Client(base, "bench@example.org")
    pid = c.http.post("/projects", headers=c.h, json={"title": "Bench", "question": "How does temperature affect polymer electrolyte conductivity?"}).json()["id"]

    out["simple_endpoints"] = {}
    for name, url in (("health", "/health"), ("auth_me", "/auth/me"), ("list_projects", "/projects"), ("overview", "/overview"), ("audit", "/audit?limit=20")):
        ms = [c.timed("GET", url)[0] for _ in range(200)]
        out["simple_endpoints"][name] = summary(ms)

    rng, doc_i, scaling = random.Random(7), 0, []
    for size in sizes:
        up = []
        while doc_i < size:
            data = synthetic_doc(doc_i, rng)
            ms, r = c.timed("POST", f"/projects/{pid}/documents", files={"file": (f"doc{doc_i}.txt", data, "text/plain")},
                            data={"title": f"Synthetic {doc_i}", "license": "CC0-1.0", "source": "generated by scripts/bench_api.py", "allowed_use": "search,quote"})
            if r.status_code != 201:
                raise RuntimeError(f"upload failed {r.status_code} {r.text[:200]}")
            up.append(ms)
            doc_i += 1
        row = {"documents": size, "upload": summary(up)}
        queries = ["polymer conductivity", "lithium battery stability", "\"ionic transport\"", f"doc{rng.randrange(size)}x", "perovskite humidity degradation"]
        for level in ("chunk", "document", "metadata"):
            ms = []
            for k in range(60):
                t, r = c.timed("GET", f"/projects/{pid}/search", params={"q": queries[k % len(queries)], "level": level, "k": 8})
                if r.status_code != 200:
                    raise RuntimeError(f"search failed {r.status_code}")
                ms.append(t)
            row[f"search_{level}"] = summary(ms)
        scaling.append(row)
    out["library_scaling"] = scaling
    out["library_note"] = "synthetic documents (~3 KB each); the first search after an upload may rebuild the in-memory index, which shows in p95/max"

    # concurrent users, each with their own project and library (search only, after uploading a small library each)
    def user_load(idx: int, res: list, errs: list):
        try:
            u = Client(base, f"load{idx}@example.org")
            p2 = u.http.post("/projects", headers=u.h, json={"title": "L", "question": "How does temperature affect polymer electrolyte conductivity?"}).json()["id"]
            for d in range(15):
                u.http.post(f"/projects/{p2}/documents", headers=u.h, files={"file": (f"d{d}.txt", synthetic_doc(10_000 * idx + d, random.Random(idx)), "text/plain")},
                            data={"license": "CC0-1.0", "source": "synthetic", "allowed_use": "search,quote"})
            lat = [u.timed("GET", f"/projects/{p2}/search", params={"q": "polymer conductivity temperature"})[0] for _ in range(40)]
            res.extend(lat)
        except Exception as e:  # noqa: BLE001 - recorded, not hidden
            errs.append(repr(e))

    for users in (1, 4, 8):
        res: list[float] = []
        errs: list[str] = []
        t0 = time.perf_counter()
        ts = [threading.Thread(target=user_load, args=(users * 100 + i, res, errs)) for i in range(users)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        wall = time.perf_counter() - t0
        out.setdefault("concurrent_users_search", []).append({"users": users, "requests": len(res), "errors": errs[:3], "error_count": len(errs),
                                                              **(summary(res) if res else {}), "wall_s_including_setup": round(wall, 2)})

    # research run: queue latency, completion, SSE
    t_create = time.perf_counter()
    r = c.http.post(f"/projects/{pid}/runs", headers=c.h, json={})
    r.raise_for_status()
    run_id = r.json()["id"]
    events: list[tuple[float, str]] = []
    done = threading.Event()

    def listen():
        with httpx.Client(base_url=base, timeout=None, cookies=c.http.cookies) as s, s.stream("GET", f"/runs/{run_id}/stream") as resp:
            ev = "message"
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    ev = line[6:].strip()
                elif line.startswith("data:"):
                    events.append((time.perf_counter(), ev))
                    if ev == "end":
                        break
        done.set()

    th = threading.Thread(target=listen, daemon=True)
    th.start()
    first_running = None
    final = None
    while time.perf_counter() - t_create < 600:
        tasks = c.http.get(f"/runs/{run_id}/tasks").json()
        if first_running is None and any(t["status"] != "pending" for t in tasks):
            first_running = time.perf_counter() - t_create
        run = c.http.get(f"/runs/{run_id}").json()
        status = (run.get("state") or {}).get("status")
        if status in {"completed", "failed", "cancelled"}:
            final = status
            break
        time.sleep(0.05)
    total = time.perf_counter() - t_create
    done.wait(30)
    gaps = [b[0] - a[0] for a, b in zip(events, events[1:])]
    out["research_run"] = {"final_status": final, "tasks": len(tasks), "queue_latency_s_to_first_task_started": round(first_running, 3) if first_running else None,
                           "completion_s": round(total, 2), "poll_interval_s": 0.05,
                           "sse": {"events": len(events), "ended_cleanly": bool(events and events[-1][1] == "end"), "max_gap_s": round(max(gaps), 3) if gaps else None,
                                   "first_event_s_after_create": round(events[0][0] - t_create, 3) if events else None}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--postgres", help="postgresql:// URL (schemas bench_api / bench_engine are used); default SQLite")
    ap.add_argument("--sizes", default="25,100,250")
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "evaluation" / "api_bench.json"))
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="researchos-bench-"))
    proc = start_server(a.port, work, a.postgres)
    try:
        res = bench(f"http://127.0.0.1:{a.port}", [int(x) for x in a.sizes.split(",")])
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
    res["environment"] = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                          "python": platform.python_version(), "database": "postgresql (loopback)" if a.postgres else "sqlite (local files)",
                          "client_and_server": "same machine, loopback", "rate_limits": "off for this run", "corpus": "synthetic generated text"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
