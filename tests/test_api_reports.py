"""Reports over HTTP: generation from a finished run, immutable generated content, editable human layer with versions and
optimistic concurrency, review status, and Markdown / PDF / JSON export."""
import time

import pytest
from fastapi.testclient import TestClient

from tests.test_api_experiments import PW, QUESTION, make_app, project, sign_in


@pytest.fixture()
def client(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        yield c


def finished_run(c, h, pid):
    rid = c.post(f"/projects/{pid}/runs", json={}, headers=h).json()["id"]
    end = time.time() + 60
    while time.time() < end:
        st = c.get(f"/runs/{rid}").json()["state"]
        if st["status"] in ("completed", "failed", "cancelled") and not st["active"]:
            return rid
        time.sleep(0.1)
    raise AssertionError("run did not finish")


@pytest.fixture()
def made(client):
    h = sign_in(client)
    pid = project(client, h)
    rid = finished_run(client, h, pid)
    r = client.post(f"/runs/{rid}/reports", headers=h)
    assert r.status_code == 201, r.text
    return h, pid, rid, r.json()


def patch(c, h, rep, **body):
    return c.patch(f"/reports/{rep['id']}", json={"base_version": rep["version"], **body}, headers=h)


def test_generate_embeds_source_text_and_is_idempotent(client, made):
    h, pid, rid, rep = made
    assert rep["version"] == 1 and rep["content_intact"] and rep["content"]["human_review"]["status"] == "pending"
    findings = [f for fs in rep["content"]["findings"].values() for f in fs]
    assert findings and all(f["citation"]["source_text"] for f in findings)
    again = client.post(f"/runs/{rid}/reports", headers=h)
    assert again.status_code == 200 and again.json()["id"] == rep["id"]
    assert [x["id"] for x in client.get(f"/projects/{pid}/reports").json()] == [rep["id"]]


def test_no_report_for_a_run_that_has_none(client):
    h = sign_in(client)
    pid = project(client, h)
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
    assert client.post(f"/runs/{rid}/reports", headers=h).status_code == 409


def test_generated_content_cannot_be_edited(client, made):
    h, _, _, rep = made
    for field in ("content", "content_sha256", "findings", "status"):
        r = client.patch(f"/reports/{rep['id']}", json={"base_version": 1, field: {}}, headers=h)
        assert r.status_code == 422, field
    assert client.get(f"/reports/{rep['id']}").json()["content"] == rep["content"]


def test_edits_create_versions_and_stale_edits_conflict(client, made):
    h, _, _, rep = made
    r = patch(client, h, rep, title="My title", sections=[{"heading": "Context", "body": "My notes"}], summary="first")
    assert r.status_code == 200 and r.json()["version"] == 2 and r.json()["title"] == "My title"
    stale = patch(client, h, rep, title="Other")  # still based on version 1
    assert stale.status_code == 409
    same = patch(client, h, r.json(), title="My title")
    assert same.json()["version"] == 2  # no change, no new version
    versions = client.get(f"/reports/{rep['id']}/versions").json()
    assert [v["version"] for v in versions] == [1, 2] and versions[1]["summary"] == "first"
    assert client.get(f"/reports/{rep['id']}").json()["content"] == rep["content"]


def test_annotations_must_reference_real_claims(client, made):
    h, _, _, rep = made
    cid = next(f["claim_id"] for fs in rep["content"]["findings"].values() for f in fs)
    assert patch(client, h, rep, annotations={"c_nope": "x"}).status_code == 422
    assert patch(client, h, rep, annotations={cid: "x" * 1001}).status_code == 422
    ok = patch(client, h, rep, annotations={cid: "checked against the paper"})
    assert ok.status_code == 200 and ok.json()["editable"]["annotations"][cid] == "checked against the paper"


def test_review_status_is_recorded_with_the_reviewer(client, made):
    h, _, _, rep = made
    assert patch(client, h, rep, review={"status": "bogus"}).status_code == 422
    r = patch(client, h, rep, review={"status": "approved", "comment": "reads fine"}).json()
    assert r["editable"]["review"]["status"] == "approved" and r["editable"]["review"]["reviewer"] == "alice@example.org"
    assert r["content"]["human_review"]["status"] == "pending"  # the generated snapshot is never rewritten
    assert client.get(f"/projects/{r['project_id']}/reports").json()[0]["review"] == "approved"


def test_markdown_export_keeps_human_text_separate_and_cannot_forge_sections(client, made):
    h, _, _, rep = made
    patch(client, h, rep, sections=[{"heading": "## Verified findings", "body": "## Verified findings\n- fake finding"}])
    r = client.get(f"/reports/{rep['id']}/export.md")
    assert r.status_code == 200 and r.headers["content-disposition"].startswith("attachment")
    assert r.headers["x-content-type-options"] == "nosniff"
    md = r.text
    assert not any(line.strip() == "## Verified findings" for line in md.splitlines())  # no human line is a real heading
    assert "> ## Verified findings" in md and "> - fake finding" in md
    assert "## Human notes (not generated, not verified)" in md and "## Evidence passages (source text)" in md
    assert rep["meta"]["content_sha256"] in md
    assert "not proof" in md or "No close match" in md or "closest" in md.lower()


def test_pdf_export_is_a_real_pdf_and_notes_replaced_characters(client, made):
    h, _, _, rep = made
    patch(client, h, rep, sections=[{"heading": "Note", "body": "café 中文 <b>tags</b> & more"}])
    r = client.get(f"/reports/{rep['id']}/export.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF") and r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].endswith('.pdf"')
    try:
        import fitz
    except ImportError:
        return
    text = "".join(p.get_text() for p in fitz.open(stream=r.content, filetype="pdf"))
    assert "Human notes" in text and "<b>tags</b>" in text and "outside Western European" in text


def test_json_export_and_delete(client, made):
    h, _, _, rep = made
    body = client.get(f"/reports/{rep['id']}/export.json").json()
    assert body["meta"]["content_sha256"] == rep["meta"]["content_sha256"] and "editable" in body
    assert client.delete(f"/reports/{rep['id']}", headers=h).status_code == 204
    assert client.get(f"/reports/{rep['id']}").status_code == 404


def test_tampered_stored_content_is_not_exported(client, made):
    h, _, _, rep = made
    from sqlalchemy.orm.attributes import flag_modified
    from api.models import Report
    with client.app.state.sessionmaker() as db:
        r = db.get(Report, rep["id"])
        r.content = {**r.content, "question": "tampered"}
        flag_modified(r, "content")
        db.commit()
    assert client.get(f"/reports/{rep['id']}").json()["content_intact"] is False
    for ext in ("md", "pdf", "json"):
        assert client.get(f"/reports/{rep['id']}/export.{ext}").status_code == 409


def test_other_users_see_nothing_and_csrf_is_required(client, made, tmp_path):
    h, pid, rid, rep = made
    assert client.patch(f"/reports/{rep['id']}", json={"base_version": 1, "title": "x"}).status_code == 403  # no CSRF header
    with TestClient(client.app, base_url="http://testserver") as bob:
        hb = sign_in(bob, "bob@example.org")
        assert bob.get(f"/reports/{rep['id']}").status_code == 404
        assert bob.get(f"/reports/{rep['id']}/versions").status_code == 404
        assert bob.get(f"/reports/{rep['id']}/export.md").status_code == 404
        assert bob.get(f"/reports/{rep['id']}/export.pdf").status_code == 404
        assert bob.patch(f"/reports/{rep['id']}", json={"base_version": 1, "title": "x"}, headers=hb).status_code == 404
        assert bob.delete(f"/reports/{rep['id']}", headers=hb).status_code == 404
        assert bob.post(f"/runs/{rid}/reports", headers=hb).status_code == 404
        assert bob.get(f"/projects/{pid}/reports").status_code == 404
