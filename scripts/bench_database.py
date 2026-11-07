"""Micro-benchmark of the engine store and the job queue on SQLite and on PostgreSQL.

    python scripts/bench_database.py [--postgres postgresql://...] [--out artifacts/evaluation/database_bench.json]

Without --postgres it starts the pip-installable PostgreSQL used by the tests (pgserver). The numbers describe THIS machine with
PostgreSQL on the loopback interface: they say nothing about a networked production database, and they are single short runs, not a
statistical benchmark. They exist to show that nothing is pathologically slow and to compare the two backends under the same code."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def timed(fn, n):
    t = time.perf_counter()
    for i in range(n):
        fn(i)
    dt = time.perf_counter() - t
    return {"n": n, "total_s": round(dt, 4), "per_s": round(n / dt, 1), "ms_each": round(1000 * dt / n, 3)}


def bench_store(target: str, n_events=400, n_tasks=200) -> dict:
    from agent_engine.state.store import Store
    s = Store(target)
    pid = s.create_project("bench", "q")
    rid = s.create_run(pid, "q", {})
    out = {"add_event": timed(lambda i: s.add_event(rid, "e", {"i": i}), n_events)}
    s.add_tasks(rid, [{"id": f"t{i}", "agent_type": "a", "title": "x", "input": {"i": i}} for i in range(n_tasks)])
    out["task_pending_to_running"] = timed(lambda i: s.update_task(f"t{i}", status="running"), n_tasks)
    out["task_running_to_succeeded"] = timed(lambda i: s.update_task(f"t{i}", status="succeeded", output={"ok": i}), n_tasks)
    out["list_tasks_200"] = timed(lambda i: s.list_tasks(rid), 20)
    out["list_events_after_id"] = timed(lambda i: s.list_events(rid, after_id=i), 50)
    # several writers, one connection each
    def writer(k):
        w = Store(target)
        for i in range(100):
            w.add_event(rid, "w", {"k": k, "i": i})
        w.close()
    t = time.perf_counter()
    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    [x.start() for x in threads]
    [x.join() for x in threads]
    dt = time.perf_counter() - t
    out["4_writers_x_100_events"] = {"n": 400, "total_s": round(dt, 4), "per_s": round(400 / dt, 1)}
    s.close()
    return out


def bench_queue(api_url: str, jobs=60, workers=4) -> dict:
    from api import migrate
    from api.config import Settings
    from api.db import make_engine, make_sessionmaker
    from api.jobs import JobQueue
    from api.models import Project, Run, User
    eng = make_engine(api_url)
    migrate.upgrade(eng)
    sm = make_sessionmaker(eng)
    with sm() as db:
        u = User(email="b@example.org", password_hash="x")
        db.add(u); db.flush()
        p = Project(owner_id=u.id, title="p", question="q?")
        db.add(p); db.flush()
        runs = [Run(project_id=p.id, owner_id=u.id, engine_run_id=f"e{i}", question="q?", workflow="full") for i in range(jobs)]
        db.add_all(runs); db.commit()
        runs = [db.get(Run, r.id) for r in runs]
    q = JobQueue(sm, Settings(job_lease_s=30, job_max_attempts=3))
    enq = timed(lambda i: q.enqueue(runs[i]), jobs)
    claimed, lat = [], []
    gate = threading.Barrier(workers)

    def work(n):
        e = make_engine(api_url)
        wq = JobQueue(make_sessionmaker(e), Settings(job_lease_s=30))
        gate.wait()
        while True:
            t0 = time.perf_counter()
            j = wq.claim(f"w{n}")
            lat.append(time.perf_counter() - t0)
            if j is None:
                break
            claimed.append(j["id"])
        e.dispose()

    t = time.perf_counter()
    threads = [threading.Thread(target=work, args=(n,)) for n in range(workers)]
    [x.start() for x in threads]
    [x.join() for x in threads]
    dt = time.perf_counter() - t
    eng.dispose()
    return {"enqueue": enq, "claim": {"workers": workers, "jobs": jobs, "claimed": len(claimed), "unique": len(set(claimed)),
                                      "total_s": round(dt, 4), "median_claim_ms": round(1000 * statistics.median(lat), 3),
                                      "p95_claim_ms": round(1000 * sorted(lat)[int(0.95 * len(lat)) - 1], 3)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--postgres", help="postgresql:// URL (a throwaway database is strongly advised)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "evaluation" / "database_bench.json"))
    args = ap.parse_args()
    from agent_engine.state import dburl
    base = args.postgres
    if not base:
        from tests import pg_support
        base = pg_support.base_url()
    import uuid
    result: dict = {"note": "single short runs on one machine, PostgreSQL over loopback; indicative only"}
    tmp = Path(tempfile.mkdtemp(prefix="researchos-bench-"))
    for name, engine_t, api_t in (
        ("sqlite", str(tmp / "engine.db"), f"sqlite:///{(tmp / 'api.db').as_posix()}"),
        ("postgresql", dburl.with_schema(base, f"bench_e_{uuid.uuid4().hex[:8]}"), dburl.with_schema(base, f"bench_a_{uuid.uuid4().hex[:8]}")),
    ):
        result[name] = {"store": bench_store(engine_t), "queue": bench_queue(api_t)}
        print(name, json.dumps(result[name], indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
