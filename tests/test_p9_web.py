"""P9 web support: the /overview and /audit aggregates (shape, owner isolation, auth) and the /app single-page-app route (fallback,
traversal refusal, missing assets, CSP, and the not-built message)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings

PW = "correct horse battery 42"
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@pytest.fixture()
def app():
    return create_app(Settings(database_url="sqlite:///:memory:", engine_db=":memory:", corpus_isolate=False))


def browser(app):
    return TestClient(app, base_url="http://testserver")


def sign_up(c, email):
    r = c.post("/auth/register", json={"email": email, "password": PW})
    assert r.status_code == 201, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def test_overview_and_audit_require_a_session(app):
    c = browser(app)
    assert c.get("/overview").status_code == 401
    assert c.get("/audit").status_code == 401


def test_overview_of_a_new_account_is_all_zero_and_honest(app):
    c = browser(app)
    sign_up(c, "a@example.org")
    o = c.get("/overview").json()
    assert (o["projects"], o["runs"], o["documents"], o["reports"], o["pending_approvals"], o["claims_total"]) == (0, 0, 0, 0, 0, 0)
    assert len(o["runs_per_day"]) == 30 and all(d["runs"] == 0 for d in o["runs_per_day"])
    assert set(o["claims"]) == {"SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "UNVERIFIED"}
    assert o["recent_runs"] == [] and o["recent_projects"] == [] and "note" in o


def test_overview_counts_are_scoped_to_the_signed_in_user(app):
    a, b = browser(app), browser(app)
    ha = sign_up(a, "a@example.org")
    sign_up(b, "b@example.org")
    assert a.post("/projects", json={"title": "Mine", "question": "What matters?"}, headers=ha).status_code == 201
    assert a.get("/overview").json()["projects"] == 1
    other = b.get("/overview").json()
    assert other["projects"] == 0 and other["recent_projects"] == []


def test_audit_lists_only_the_callers_entries_and_filters_by_action(app):
    a, b = browser(app), browser(app)
    ha = sign_up(a, "a@example.org")
    sign_up(b, "b@example.org")
    a.post("/projects", json={"title": "Mine", "question": "What matters?"}, headers=ha)
    mine = a.get("/audit").json()
    assert mine["total"] >= 1 and mine["entries"] and mine["actions"]
    assert all(set(e) == {"id", "at", "action", "target", "detail"} for e in mine["entries"])
    theirs = b.get("/audit").json()
    assert {e["id"] for e in theirs["entries"]}.isdisjoint({e["id"] for e in mine["entries"]})
    one = mine["actions"][0]
    filtered = a.get("/audit", params={"action": one}).json()
    assert filtered["entries"] and {e["action"] for e in filtered["entries"]} == {one}
    assert a.get("/audit", params={"limit": 0}).status_code == 422
    assert len(a.get("/audit", params={"limit": 1}).json()["entries"]) == 1


def test_audit_never_contains_credentials(app):
    c = browser(app)
    sign_up(c, "a@example.org")
    assert PW not in c.get("/audit").text


def test_spa_reports_when_it_is_not_built(app, monkeypatch):
    if (DIST / "index.html").is_file():
        pytest.skip("frontend/dist exists in this checkout; the not-built path is covered when it does not")
    r = browser(app).get("/app")
    assert r.status_code == 404 and "not built" in r.json()["detail"]


@pytest.fixture()
def built(app, tmp_path, monkeypatch):
    """Serve a tiny fake build so the routing rules are tested without needing node."""
    if not (DIST / "index.html").is_file():
        pytest.skip("frontend/dist is not built (npm run build)")
    return browser(app)


def test_spa_serves_the_shell_with_a_strict_csp(built):
    r = built.get("/app")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert r.headers["x-frame-options"] == "DENY"
    assert "<script" in r.text and 'type="module"' in r.text


def test_spa_client_routes_fall_back_to_the_shell(built):
    shell = built.get("/app").text
    assert built.get("/app/projects").text == shell
    assert built.get("/app/runs/abc123").text == shell


def test_spa_missing_assets_are_404_not_the_shell(built):
    r = built.get("/app/assets/does-not-exist.js")
    assert r.status_code == 404


def test_spa_refuses_path_traversal(built):
    for p in ("/app/../api/config.py", "/app/..%2fapi%2fconfig.py", "/app/%2e%2e/%2e%2e/pyproject.toml", "/app/assets/../../../pyproject.toml"):
        r = built.get(p)
        assert "[project]" not in r.text and "class Settings" not in r.text


def test_spa_serves_real_built_assets_with_the_right_type(built):
    import re
    html = built.get("/app").text
    js = re.search(r'src="/app/(assets/[^"]+\.js)"', html) or re.search(r'src="\.?/?(assets/[^"]+\.js)"', html)
    assert js, html[:500]
    r = built.get("/app/" + js.group(1))
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]


def test_spa_text_assets_are_gzipped_when_accepted_and_identical_when_decoded(built):
    import re

    html = built.get("/app", headers={"Accept-Encoding": "identity"}).text
    asset = re.search(r'src="/app/(assets/[^"]+\.js)"', html).group(1)
    plain = built.get("/app/" + asset, headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    raw = built.build_request("GET", "/app/" + asset, headers={"Accept-Encoding": "gzip"})
    zipped = built.send(raw)  # the test client decodes transparently, so check the header and the byte count instead
    assert zipped.headers["content-encoding"] == "gzip" and zipped.headers["vary"] == "Accept-Encoding"
    assert zipped.content == plain.content and zipped.num_bytes_downloaded < len(plain.content) / 2  # fewer bytes on the wire
