"""P10 tester pass: cross-user isolation across the whole documented API surface.

User A creates one of every owned resource. User B (a different, signed-in account) then calls EVERY documented operation that takes an id,
with A's real ids, using every HTTP method the operation supports. Nothing may succeed (no 2xx) and nothing may crash (no 5xx), and none
of A's data may appear in a response. A control pass with A's own session proves the same calls do work for the owner, so the test cannot
pass merely because every route is broken."""
import re
import time

import pytest
from fastapi.testclient import TestClient

from tests.test_api_experiments import make_app, new_exp, project, sign_in

PARAM = re.compile(r"{(\w+)}")
SECRET = "Zyxwvutsrqponm unique marker sentence about cell efficiency retained after cycling."


@pytest.fixture()
def world(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app, base_url="http://testserver") as a, TestClient(app, base_url="http://testserver") as b:
        ha = sign_in(a, "owner@example.org")
        hb = sign_in(b, "intruder@example.org")
        pid = project(a, ha, title="Owner's private project")
        up = a.post(f"/projects/{pid}/documents", headers=ha,
                    files={"file": ("notes.txt", (SECRET + " ") * 8 + "\n\nSecond paragraph with more owner text.", "text/plain")})
        assert up.status_code == 201, up.text
        doc_id = up.json()["document_id"]
        rid = a.post(f"/projects/{pid}/runs", json={}, headers=ha).json()["id"]
        end = time.time() + 90
        while time.time() < end:
            st = a.get(f"/runs/{rid}").json()["state"]
            if st["status"] in ("completed", "failed", "cancelled") and not st["active"]:
                break
            time.sleep(0.1)
        report = a.post(f"/runs/{rid}/reports", headers=ha)
        claim = a.post(f"/projects/{pid}/claims", headers=ha, json={"text": "Cells retained efficiency after cycling.", "claim_type": "finding"})
        assert claim.status_code == 201, claim.text
        exp_id = new_exp(a, ha, pid)
        ids = {"project_id": pid, "doc_id": doc_id, "run_id": rid, "exp_id": exp_id, "claim_id": claim.json()["id"],
               "report_id": report.json().get("id", "x") if report.status_code == 201 else "x"}
        yield app, a, ha, b, hb, ids


def operations(app):
    for path, ops in sorted(app.openapi()["paths"].items()):
        if "{" in path:
            for method in ops:
                if method in ("get", "post", "put", "patch", "delete"):
                    yield method.upper(), path


def fill(path, ids):
    return PARAM.sub(lambda m: ids.get(m.group(1), "x" * 8), path)


# Streams stay open on success; for the intruder they must be refused immediately, and the owner control skips them.
STREAMS = ("/stream", "/events")


def call(client, method, url, headers):
    body = {"json": {}} if method in ("POST", "PUT", "PATCH") else {}
    return client.request(method, url, headers=headers, **body)


def test_another_users_ids_are_refused_everywhere(world):
    app, a, ha, b, hb, ids = world
    problems, checked = [], 0
    for method, path in operations(app):
        url = fill(path, ids)
        r = call(b, method, url, hb)
        checked += 1
        leaked = SECRET[:30] in r.text or "Owner's private project" in r.text
        if r.status_code < 400 or r.status_code >= 500 or leaked:
            problems.append(f"{method} {url} -> {r.status_code}{' LEAK' if leaked else ''}: {r.text[:120]}")
    assert checked > 60, checked
    assert not problems, "\n".join(problems)

    # the intruder changed nothing
    assert a.get(f"/projects/{ids['project_id']}").status_code == 200
    assert a.get(f"/projects/{ids['project_id']}/documents/{ids['doc_id']}").status_code == 200
    assert a.get(f"/experiments/{ids['exp_id']}").status_code == 200
    assert a.get(f"/projects/{ids['project_id']}/claims/{ids['claim_id']}").status_code == 200


def test_the_owner_control_works_so_the_refusals_above_are_meaningful(world):
    app, a, ha, b, hb, ids = world
    for url in (f"/projects/{ids['project_id']}", f"/projects/{ids['project_id']}/documents",
                f"/projects/{ids['project_id']}/documents/{ids['doc_id']}", f"/projects/{ids['project_id']}/documents/{ids['doc_id']}/passages",
                f"/projects/{ids['project_id']}/claims/{ids['claim_id']}", f"/experiments/{ids['exp_id']}", f"/runs/{ids['run_id']}",
                f"/runs/{ids['run_id']}/tasks", f"/projects/{ids['project_id']}/search?q=efficiency"):
        r = a.get(url)
        assert r.status_code == 200, (url, r.status_code, r.text[:150])


def test_garbage_ids_never_crash_the_server(world):
    app, a, ha, b, hb, ids = world
    junk = {k: "x" * 8 for k in ids}
    junk_weird = {k: "../../etc/passwd" for k in ids}
    bad = []
    for method, path in operations(app):
        for table in (junk, {k: "' OR 1=1 --" for k in ids}):
            url = fill(path, table)
            if method == "GET" and url.endswith(STREAMS):
                continue
            r = call(a, method, url, ha)
            if r.status_code >= 500:
                bad.append(f"{method} {url} -> {r.status_code}")
        r = call(a, method, fill(path, junk_weird).replace("../../etc/passwd", "..%2f..%2fetc%2fpasswd"), ha)
        if r.status_code >= 500:
            bad.append(f"{method} {path} (encoded traversal) -> {r.status_code}")
    assert not bad, "\n".join(bad)


BODIES = [
    {"json": []}, {"json": "text"}, {"json": None}, {"json": {"text": None, "title": None, "question": None, "spec": None}},
    {"json": {"text": "x" * 200_000, "title": "t" * 100_000, "question": "q" * 100_000}},
    {"json": {"spec": {"code": 7}, "base_version": "one", "limit": -1, "k": 10**12, "temperature": "hot"}},
    {"content": b"\xff\xfe\x00not json", "headers": {"Content-Type": "application/json"}},
    {"content": b"{" * 5000, "headers": {"Content-Type": "application/json"}},
    {"data": {"a": "b"}},
]


def test_malformed_bodies_get_4xx_never_a_server_error(world):
    """Hostile or broken request bodies on every write operation, with the owner's real ids so authorisation passes and the handler runs."""
    app, a, ha, b, hb, ids = world
    bad, seen = [], {}
    for method, path in operations(app):
        if method == "GET" or path.endswith(("/stream", "/events", "/generate/stream")):
            continue
        url = fill(path, ids)
        if method == "DELETE":
            continue  # no body; destroying the owner's fixtures would only hide later cases
        for body in BODIES:
            extra = dict(body)
            headers = {**ha, **extra.pop("headers", {})}
            r = a.request(method, url, headers=headers, **extra)
            seen[r.status_code] = seen.get(r.status_code, 0) + 1
            if r.status_code >= 500:
                bad.append(f"{method} {url} {str(body)[:50]} -> {r.status_code}")
    assert not bad, "\n".join(bad[:20])
    assert seen.get(422, 0) >= 100, seen  # the bodies really reached request validation, not only an earlier refusal
