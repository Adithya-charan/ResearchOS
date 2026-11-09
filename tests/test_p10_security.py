"""P10: security headers on every response, per-user rate limits (configurable, server-side, scoped per user), and config parsing."""
import io

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import RATE_DEFAULTS, Settings, parse_rate, rates_from_env

PW = "correct horse battery 42"


def make(**kw):
    return create_app(Settings(database_url="sqlite:///:memory:", engine_db=":memory:", corpus_isolate=False, **kw))


def sign_up(c, email):
    r = c.post("/auth/register", json={"email": email, "password": PW})
    assert r.status_code == 201, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def test_parse_rate_accepts_counts_and_off_and_rejects_nonsense():
    assert parse_rate("60/600") == (60, 600)
    assert parse_rate("OFF") is None
    for bad in ("0/10", "5/0", "abc", "5"):
        with pytest.raises(ValueError):
            parse_rate(bad)


def test_rates_from_env_overrides_only_what_is_set():
    r = rates_from_env({"RESEARCHOS_RATE_SEARCH": "5/60", "RESEARCHOS_RATE_UPLOAD": "off"})
    assert r["search"] == (5, 60) and r["upload"] is None and r["verify"] == RATE_DEFAULTS["verify"]


def test_every_response_carries_the_baseline_headers():
    c = TestClient(make(), base_url="http://testserver")
    for path in ("/health", "/auth/me", "/nope"):
        h = c.get(path).headers
        assert h["x-content-type-options"] == "nosniff" and h["referrer-policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in h["content-security-policy"] and "unsafe-inline" not in h["content-security-policy"]
        assert "camera=()" in h["permissions-policy"] and h["x-frame-options"] == "DENY"
        assert h["cache-control"] == "no-store" and "strict-transport-security" not in h


def test_hsts_only_when_enabled_and_docs_keep_their_viewer_csp():
    c = TestClient(make(hsts=True), base_url="http://testserver")
    assert c.get("/health").headers["strict-transport-security"].startswith("max-age=")
    assert "content-security-policy" not in c.get("/docs").headers  # Swagger UI loads its viewer from a CDN


def test_search_is_limited_per_user_and_the_limit_is_not_shared():
    app = make(rate_limits={**RATE_DEFAULTS, "search": (3, 60)})
    a, b = TestClient(app, base_url="http://testserver"), TestClient(app, base_url="http://testserver")
    ha, hb = sign_up(a, "a@example.org"), sign_up(b, "b@example.org")
    pa = a.post("/projects", json={"title": "A", "question": "What matters?"}, headers=ha).json()["id"]
    pb = b.post("/projects", json={"title": "B", "question": "What matters?"}, headers=hb).json()["id"]
    codes = [a.get(f"/projects/{pa}/search", params={"q": "efficiency"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    limited = a.get(f"/projects/{pa}/search", params={"q": "efficiency"})
    assert limited.headers["retry-after"] == "60" and "search" in limited.json()["detail"]
    assert b.get(f"/projects/{pb}/search", params={"q": "efficiency"}).status_code == 200  # another user is unaffected


def test_unauthenticated_requests_do_not_consume_budget_and_get_401():
    app = make(rate_limits={**RATE_DEFAULTS, "search": (1, 60)})
    c = TestClient(app, base_url="http://testserver")
    assert [c.get("/projects/x/search", params={"q": "ab"}).status_code for _ in range(3)] == [401, 401, 401]


def test_upload_and_run_creation_are_limited_and_off_disables():
    app = make(rate_limits={**RATE_DEFAULTS, "upload": (2, 60), "run_create": None})
    c = TestClient(app, base_url="http://testserver")
    h = sign_up(c, "a@example.org")
    pid = c.post("/projects", json={"title": "A", "question": "What matters?"}, headers=h).json()["id"]
    codes = [c.post(f"/projects/{pid}/documents", headers=h, files={"file": ("n.txt", io.BytesIO(b"hello world"), "text/plain")}).status_code
             for _ in range(4)]
    assert codes[2:] == [429, 429] and 429 not in codes[:2]
    assert app.state.settings.rate_limits["run_create"] is None


def test_settings_from_env_reads_limits_and_refuses_wildcard_origins(monkeypatch):
    monkeypatch.setenv("RESEARCHOS_MAX_UPLOAD_BYTES", "1048576")
    monkeypatch.setenv("RESEARCHOS_WORKER_CONCURRENCY", "4")
    monkeypatch.setenv("RESEARCHOS_RATE_RUN_CREATE", "3/60")
    s = Settings.from_env()
    assert (s.max_upload_bytes, s.worker_concurrency, s.rate_limits["run_create"]) == (1048576, 4, (3, 60))
    monkeypatch.setenv("RESEARCHOS_ALLOWED_ORIGINS", "https://*.example.org")
    with pytest.raises(ValueError, match="exact origins"):
        Settings.from_env()


def test_risky_configurations_are_reported():
    s = Settings(allowed_origins=("https://r.example.org",), cookie_secure=False, allow_weak_sandbox=True, min_password_length=8)
    text = " ".join(s.warnings())
    assert "COOKIE_SECURE" in text and "WEAK_SANDBOX" in text and "password" in text
    assert Settings(allowed_origins=("https://r.example.org",), cookie_secure=True, hsts=True).warnings() == []


def test_every_api_route_requires_authentication_except_the_public_ones():
    """Regression guard for authorization by omission, checked by behaviour: every documented operation is called with no session and
    must answer 401. A new route that forgets `CurrentAuth` shows up here. (Static pages under /app and /lab are not in the schema.)"""
    import re

    app = make()
    c = TestClient(app, base_url="http://testserver")
    open_ops, checked = [], 0
    for path, ops in app.openapi()["paths"].items():
        for method in ops:
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            checked += 1
            r = c.request(method.upper(), re.sub(r"\{[^}]+\}", "x", path))
            if r.status_code != 401:
                open_ops.append(f"{method.upper()} {path} -> {r.status_code}")
    assert checked > 90, checked  # the scan really saw the API
    assert sorted(o.split(" ->")[0] for o in open_ops) == ["GET /health", "POST /auth/login", "POST /auth/register"], open_ops


def test_api_docs_can_be_switched_off():
    c = TestClient(make(enable_docs=False), base_url="http://testserver")
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
    assert TestClient(make(), base_url="http://testserver").get("/openapi.json").status_code == 200
