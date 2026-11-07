"""P8 Gate G (Model Lab UI): the bundled page is served from a fixed whitelist under a strict CSP, calls only endpoints that exist, holds no
fabricated data, no unsafe DOM sinks and no external resources, and exposes all twelve required sections."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings

LAB = Path(__file__).resolve().parents[1] / "api" / "static" / "lab"
SECTIONS = ["Model overview", "Tokenizer", "Dataset management", "Model architecture", "Training configuration", "Training runs",
            "Live training metrics", "Checkpoint manager", "Model evaluation", "Inference playground", "Model registry", "Agent integration status"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    root = tmp_path_factory.mktemp("p8fe")
    s = Settings(database_url="sqlite:///:memory:", engine_db=":memory:", data_dir=root / "data", corpus_isolate=False)
    return create_app(s)


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app, base_url="http://testserver")


def js() -> str:
    return (LAB / "lab.js").read_text(encoding="utf-8")


def test_lab_files_are_served_with_csp_and_correct_types(client):
    for path, kind in (("/lab", "text/html"), ("/lab/", "text/html"), ("/lab/lab.js", "text/javascript"), ("/lab/lab.css", "text/css")):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith(kind)
        assert "default-src 'self'" in r.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
        assert r.headers["x-content-type-options"] == "nosniff"
        assert "unsafe-inline" not in r.headers["content-security-policy"] and "unsafe-eval" not in r.headers["content-security-policy"]


def test_only_whitelisted_files_exist_and_traversal_is_refused(client):
    for path in ("/lab/app.py", "/lab/../app.py", "/lab/%2e%2e/app.py", "/lab/index.html", "/lab/..%5capp.py", "/lab/lab.js/x", "/lab/static/lab/lab.js"):
        assert client.get(path).status_code == 404, path


def test_page_has_no_inline_script_or_external_resources():
    html, css, script = (LAB / "index.html").read_text(encoding="utf-8"), (LAB / "lab.css").read_text(encoding="utf-8"), js()
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline script would be blocked by the CSP"
    assert not re.search(r"\son\w+\s*=", html), "inline event handler attributes"
    for name, text in (("index.html", html), ("lab.css", css), ("lab.js", script)):
        # a link to a third-party host would break `default-src 'self'` and leak requests; only the SVG namespace string is allowed
        urls = [u for u in re.findall(r"https?://[^\s\"'`)<>]+", text) if u != "http://www.w3.org/2000/svg"]
        assert not urls, f"{name} references external URLs: {urls}"
    assert not re.search(r"@import|url\(\s*['\"]?https?:", css)


def test_script_avoids_unsafe_dom_sinks_and_fake_data():
    script = js()
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "setAttribute(\"style\"", "Math.random", "lorem"):
        assert bad not in script, bad
    # the number formatter shows "-" for a missing value; it never substitutes one
    assert 'n == null || Number.isNaN(n) ? "-"' in script


def test_all_twelve_sections_are_present_in_the_script_and_the_page():
    script = js()
    ids = re.findall(r'\["(\w+)", "(\d+)\. ([^"]+)"\]', script)
    assert len(ids) == 12 and [int(n) for _, n, _ in ids] == list(range(1, 13))
    render_ids = set(re.findall(r"(\w+): sec\w+", re.search(r"const RENDER = \{(.*?)\};", script, re.S).group(1)))
    assert {i for i, _, _ in ids} == render_ids, "every navigation entry needs a renderer"
    headings = re.findall(r'h\("h2", \{\}, "([^"]+)"\)', script)
    for want in SECTIONS:
        assert want in headings, f"section heading missing: {want}"


def test_every_failure_path_has_visible_states():
    script = js()
    for needle in ('role: "alert"', 'role: "status"', "Retry", "Cancelled.", "No training runs yet", "No trained model exists yet", "aria-live", "aria-current"):
        assert needle in script, needle
    assert "viewport" in (LAB / "index.html").read_text(encoding="utf-8")
    assert "@media" in (LAB / "lab.css").read_text(encoding="utf-8")


def _norm(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", re.sub(r"\$\{[^}]*\}", "{}", path.split("?")[0]))


def test_every_endpoint_the_ui_calls_exists(app):
    """Catches UI/API drift: each `api("METHOD", path)` (and the streaming fetch) must match a route of the same method."""
    routes: dict[str, set[str]] = {}
    for path, ops in app.openapi()["paths"].items():
        routes[_norm(path)] = {m.upper() for m in ops}
    calls = re.findall(r'api\("(GET|POST|PUT|DELETE)",\s*[`"]([^`"]+)[`"]', js())
    calls.append(("POST", "/models/${state.modelId}/generate/stream"))
    verbs = set(re.findall(r'act\(j\.id, "(\w+)"\)', js()))  # the run buttons pass a literal verb to act(id, verb)
    assert verbs == {"pause", "cancel", "resume"}
    calls = [c for c in calls if c[1] != "/training/jobs/${id}/${verb}"] + [("POST", f"/training/jobs/${{id}}/{v}") for v in sorted(verbs)]
    assert len(calls) > 25
    for method, path in calls:
        key = _norm(path)
        assert key in routes, f"UI calls {method} {path} but no such route exists"
        assert method in routes[key], f"UI calls {method} {path} but the route allows {routes[key]}"


def test_same_origin_writes_work_and_cross_site_writes_are_still_refused(client):
    body = {"email": "lab@example.org", "password": "correct horse battery 9"}
    evil = client.post("/auth/register", json=body, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403 and evil.json()["detail"] == "origin not allowed"
    spoof = client.post("/auth/register", json=body, headers={"Origin": "http://testserver.evil.example"})
    assert spoof.status_code == 403
    same = client.post("/auth/register", json=body, headers={"Origin": "http://testserver"})  # what a browser sends from /lab
    assert same.status_code == 201, same.text


def test_lab_requires_a_session_for_all_data(client):
    anon = TestClient(client.app, base_url="http://testserver")
    for path in ("/modellab/overview", "/modellab/profiles", "/modellab/datasets", "/modellab/provider", "/training/jobs"):
        assert anon.get(path).status_code == 401, path
