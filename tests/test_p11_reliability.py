"""P11.6 agent reliability reporting: failures, retries, timeouts and skipped dependents are attributed to their root cause, partial results
are visible, and a clean run says so. The scheduler behaviours themselves are covered in test_agent_engine.py; this checks what is reported."""
import asyncio

from agent_engine.reliability import NOTE_DEGRADED, NOTE_OK, NOTE_RETRIED, run_reliability
from tests.conftest_agents import make_engine, ok, run, start
from tests.test_agent_engine import P, supervisor_proposing, tasks_by_title


def test_a_clean_run_reports_no_failures_and_says_so():
    engine, store = make_engine({"supervisor": supervisor_proposing(P("a"), P("b", deps=["a"])), "worker": lambda c, t: ok(t)})
    rid = start(engine)
    run(engine.run(rid))
    r = run_reliability(store, rid)
    assert r["finished"] and not r["degraded"] and r["failures"] == [] and r["skipped"] == [] and r["partial_results"] == []
    assert r["note"] == NOTE_OK and r["agents"]["worker"]["succeeded"] == 2 and r["agents"]["worker"]["retries"] == 0
    assert r["agents"]["worker"]["attempts"] == 2 and any("bounded" in g for g in r["guarantees"])


def test_a_retried_task_is_visible_even_though_the_run_completed():
    def flaky(ctx, task):
        if task.attempt < 3:
            raise RuntimeError("transient")
        return ok(task)

    engine, store = make_engine({"supervisor": supervisor_proposing(P("w")), "worker": flaky})
    rid = start(engine, max_retries=2)
    run(engine.run(rid))
    r = run_reliability(store, rid)
    assert not r["degraded"] and r["note"] == NOTE_RETRIED
    assert r["agents"]["worker"]["retries"] == 2 and r["agents"]["worker"]["attempts"] == 3
    assert [x["retries_used"] for x in r["retried_tasks"]] == [2]


def test_a_failure_is_attributed_and_skipped_dependents_name_the_root_cause_not_their_parent():
    def broken(ctx, task):
        raise ValueError("always broken")

    engine, store = make_engine({"supervisor": supervisor_proposing(P("bad", "broken"), P("child", deps=["bad"]), P("grandchild", deps=["child"]),
                                                                    P("independent")),
                                 "broken": broken, "worker": lambda c, t: ok(t)})
    rid = start(engine, max_retries=1)
    run(engine.run(rid))
    t = tasks_by_title(store, rid)
    r = run_reliability(store, rid)
    assert r["degraded"] and r["note"] == NOTE_DEGRADED and r["run_status"] == "failed"
    (f,) = r["failures"]
    assert f["task_id"] == t["bad"]["id"] and f["attempts"] == 2 and f["retries_used"] == 1 and "always broken" in f["error"]
    assert len(f["attempt_errors"]) == 2 and not f["timed_out"]
    by_title = {s["title"]: s for s in r["skipped"]}
    assert set(by_title) == {"child", "grandchild"}
    assert [c["task_id"] for c in by_title["grandchild"]["caused_by"]] == [t["bad"]["id"]]  # traced through child to the real failure
    assert by_title["grandchild"]["caused_by"][0]["agent_type"] == "broken"
    titles = {p["title"] for p in r["partial_results"]}
    assert "independent" in titles and "child" not in titles and "bad" not in titles  # unrelated work is kept and listed as partial
    assert r["report_available"] is False


def test_timeouts_are_counted_per_task():
    engine, store = make_engine({"supervisor": supervisor_proposing(P("w")), "worker": lambda c, t: asyncio.sleep(30)})
    rid = start(engine, task_timeout_s=0.05, max_retries=1)
    run(engine.run(rid))
    r = run_reliability(store, rid)
    assert r["agents"]["worker"]["timeouts"] == 2 and r["failures"][0]["timed_out"] is True and "timeout" in r["failures"][0]["error"]


def test_two_independent_failures_are_both_reported():
    def broken(ctx, task):
        raise ValueError(f"broken {task.title}")

    engine, store = make_engine({"supervisor": supervisor_proposing(P("x", "broken"), P("y", "broken"), P("after", deps=["x", "y"])),
                                 "broken": broken, "worker": lambda c, t: ok(t)})
    rid = start(engine, max_retries=0)
    run(engine.run(rid))
    r = run_reliability(store, rid)
    assert {f["title"] for f in r["failures"]} == {"x", "y"}
    (s,) = r["skipped"]
    assert {c["agent_type"] for c in s["caused_by"]} == {"broken"} and len(s["caused_by"]) == 2  # both roots are named


def test_reading_reliability_changes_nothing():
    engine, store = make_engine({"supervisor": supervisor_proposing(P("a")), "worker": lambda c, t: ok(t)})
    rid = start(engine)
    run(engine.run(rid))
    before = (store.list_tasks(rid), len(store.list_events(rid)))
    run_reliability(store, rid)
    assert (store.list_tasks(rid), len(store.list_events(rid))) == before
