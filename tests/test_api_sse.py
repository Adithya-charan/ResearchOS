"""Server-Sent Events: framing safety, stream limits and lifetime, live run / experiment streams, resumption, isolation."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from api.sse import Item, StreamGate, event_stream, sse_format
from research.experiments.examples import example_spec
from tests.test_api_experiments import make_app, new_exp, project, sign_in, approve


def frames(text: str) -> list[dict]:
    out = []
    for block in text.split("\n\n"):
        f = {}
        for line in block.splitlines():
            if ":" in line and not line.startswith(":"):
                k, v = line.split(":", 1)
                f[k] = v.strip()
        if f.get("event"):
            f["json"] = json.loads(f["data"])
            out.append(f)
    return out


def read(c, url, **kw):
    with c.stream("GET", url, **kw) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-store"
        return frames("".join(r.iter_text()))


# ------------------------------------------------------------------ framing and plumbing (no HTTP)
def test_frames_cannot_be_forged_by_untrusted_text():
    nasty = "line1\n\nevent: admin\ndata: pwned\r\nid: 999 x"
    out = sse_format(Item("log\nevent: x", {"text": nasty}, 7))
    assert out.count("\n\n") == 1 and out.endswith("\n\n")  # exactly one frame
    assert [ln.split(":")[0] for ln in out.strip().splitlines()] == ["id", "event", "data"]
    assert not any(ln.startswith(("event: admin", "data: pwned", "id: 999")) for ln in out.splitlines())  # nothing became a field
    assert json.loads(out.splitlines()[2][len("data: "):])["text"] == nasty
    assert len(out.splitlines()) == 4  # id, event, data, blank terminator: the raw newline and U+2028 in the text did not add lines
    assert out.splitlines()[1] == "event: log_event__x"


def test_gate_limits_per_user_and_total_and_releases():
    g = StreamGate(per_user=2, total=3, max_age_s=600)
    a, b = g.acquire("u1"), g.acquire("u1")
    with pytest.raises(Exception) as e:
        g.acquire("u1")
    assert e.value.status_code == 429
    c = g.acquire("u2")
    with pytest.raises(Exception):
        g.acquire("u3")  # total capacity
    g.release(a)
    assert g.acquire("u1") and g.active() == 3
    g.release(b), g.release(c)


def run_stream(poll, **kw):
    async def go():
        async def never():
            return False
        params = dict(is_disconnected=never, still_valid=lambda: True, interval=0.01, heartbeat=0.05, max_seconds=5, reauth_every=0.02)
        params.update(kw)
        return [f async for f in event_stream(poll, **params)]
    return asyncio.run(go())


def test_stream_ends_on_finish_session_loss_and_max_duration_and_sends_heartbeats():
    calls = iter([([Item("a", 1, 1)], False), ([], False), ([], False), ([Item("b", 2, 2)], True)])
    out = "".join(run_stream(lambda: next(calls)))
    assert out.startswith("retry: 3000") and "event: a" in out and "event: b" in out and out.rstrip().endswith('"reason": "finished"}')
    assert ": keepalive" in "".join(run_stream(lambda: ([], False), heartbeat=0.0, max_seconds=0.1))  # idle -> heartbeat
    assert "session_ended" in "".join(run_stream(lambda: ([], False), still_valid=lambda: False))
    assert "max_duration" in "".join(run_stream(lambda: ([], False), max_seconds=0.05))
    closed = []
    run_stream(lambda: ([], True), on_close=lambda: closed.append(1))
    assert closed == [1]


def test_disconnect_stops_polling_and_releases():
    polls, closed = [], []

    async def go():
        async def gone():
            return len(polls) >= 2
        def poll():
            polls.append(1)
            return [], False
        return [f async for f in event_stream(poll, is_disconnected=gone, still_valid=lambda: True, interval=0.01, heartbeat=9,
                                              max_seconds=5, reauth_every=9, on_close=lambda: closed.append(1))]
    asyncio.run(go())
    assert len(polls) == 2 and closed == [1]


# ------------------------------------------------------------------ live streams over HTTP
@pytest.fixture()
def client(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        yield c


def test_run_stream_delivers_ordered_events_then_ends_and_resumes(client):
    h = sign_in(client)
    pid = project(client, h)
    rid = client.post(f"/projects/{pid}/runs", json={}, headers=h).json()["id"]
    fs = read(client, f"/runs/{rid}/stream")  # opened while the run is executing
    assert fs[-1]["event"] == "end" and fs[-1]["json"]["reason"] == "finished"
    ev = [f for f in fs if "id" in f]
    ids = [int(f["id"]) for f in ev]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    types = {f["event"] for f in ev}
    assert {"run_created", "task_succeeded", "run_finished"} <= types
    states = [f for f in fs if f["event"] == "state"]
    assert states and states[-1]["json"]["status"] == "completed" and not states[-1]["json"]["active"]
    mid = ids[len(ids) // 2]
    again = read(client, f"/runs/{rid}/stream", headers={"Last-Event-ID": str(mid)})
    assert [int(f["id"]) for f in again if "id" in f] == [i for i in ids if i > mid]  # no gaps, no repeats
    assert [int(f["id"]) for f in read(client, f"/runs/{rid}/stream", params={"after": mid}) if "id" in f] == [i for i in ids if i > mid]


def test_stream_requires_login_ownership_and_respects_the_cap(client):
    h = sign_in(client)
    pid = project(client, h)
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
    with TestClient(client.app, base_url="http://testserver") as anon:
        assert anon.get(f"/runs/{rid}/stream").status_code == 401
    with TestClient(client.app, base_url="http://testserver") as bob:
        sign_in(bob, "bob@example.org")
        assert bob.get(f"/runs/{rid}/stream").status_code == 404
        assert bob.get("/runs/run_nope/stream").status_code == 404
    client.app.state.settings.sse_max_streams_per_user = 0
    gate = client.app.state.streams
    gate.per_user = 0
    assert client.get(f"/runs/{rid}/stream").status_code == 429
    assert gate.active() == 0


def test_experiment_stream_reports_status_and_untrusted_logs_safely(client):
    h = sign_in(client)
    pid = project(client, h)
    spec = example_spec().model_dump()
    eid = new_exp(client, h, pid, spec)
    assert approve(client, h, eid).status_code == 200
    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 202
    fs = read(client, f"/experiments/{eid}/stream")
    statuses = [f["json"]["status"] for f in fs if f["event"] == "status"]
    assert statuses[-1] == "succeeded" and fs[-1]["event"] == "end"
    logs = [f for f in fs if f["event"] == "log"]
    assert all(f["json"]["stream"] in ("stdout", "stderr") for f in logs)
    with TestClient(client.app, base_url="http://testserver") as bob:
        sign_in(bob, "bob@example.org")
        assert bob.get(f"/experiments/{eid}/stream").status_code == 404
