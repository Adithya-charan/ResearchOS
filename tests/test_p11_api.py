"""P11 over HTTP: corpus gap analysis and the opt-in advanced ranker, with ownership, licensing and default-behaviour checks."""
import json
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from fastapi.testclient import TestClient  # noqa: E402

from tests.test_api_corpus import make_app, new_project, sign_in, upload  # noqa: E402

CORPUS = {d["id"]: d for d in json.loads((Path(__file__).resolve().parents[1] / "datasets" / "p11_eval" / "corpus.json").read_text("utf-8"))["docs"]}


def put(c, h, pid, doc_id, **form):
    d = CORPUS[doc_id]
    r = upload(c, h, pid, f"{doc_id}.md", d["text"].encode("utf-8"), title=d["title"], license="CC0-1.0", source="invented test document", **form)
    assert r.status_code == 201, r.text
    return r.json()["document_id"]


@pytest.fixture()
def lib(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        ids = {k: put(c, h, pid, k) for k in ("bat1", "bat2", "bat4", "bat5", "soil1", "soil2")}
        yield c, h, pid, ids


def test_gaps_endpoint_returns_traceable_gaps_and_finds_the_opposing_biochar_trials(lib):
    c, h, pid, ids = lib
    r = c.get(f"/projects/{pid}/gaps")
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["documents"] == 6 and rep["traceability_problems"] == [] and rep["excluded_not_quotable"] == 0
    assert rep["counts"]["observed"] > 0 and "outside the library" in rep["scope"]
    conflicts = [g for g in rep["gaps"] if g["type"] == "conflicting_findings"]
    assert {frozenset(g["documents"]) for g in conflicts} >= {frozenset({ids["soil1"], ids["soil2"]}), frozenset({ids["bat4"], ids["bat5"]})}
    passages = {p["passage_id"] for d in ids.values() for p in c.get(f"/projects/{pid}/documents/{d}/passages").json()}
    assert all(e["passage_id"] in passages for g in rep["gaps"] for e in g["evidence"])  # every link resolves to a real stored passage
    assert not any("novel" in g["statement"].lower() for g in rep["gaps"])


def test_documents_not_licensed_for_quoting_are_left_out_and_counted(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        put(c, h, pid, "bat1")
        hidden = put(c, h, pid, "bat2", allowed_use="search")
        rep = c.get(f"/projects/{pid}/gaps").json()
        assert rep["documents"] == 1 and rep["excluded_not_quotable"] == 1
        assert hidden not in {d for g in rep["gaps"] for d in g["documents"]}


def test_an_empty_project_gets_an_honest_empty_answer(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        rep = c.get(f"/projects/{pid}/gaps").json()
        assert rep["gaps"] == [] and rep["documents"] == 0 and "nothing to analyse" in rep["note"]


def test_gaps_and_advanced_search_respect_ownership_and_authentication(lib):
    c, h, pid, _ = lib
    c.post("/auth/logout", headers=h)
    assert c.get(f"/projects/{pid}/gaps").status_code == 401
    assert c.get(f"/projects/{pid}/search", params={"q": "swelling", "ranker": "advanced"}).status_code == 401
    sign_in(c, "mallory@example.org")
    assert c.get(f"/projects/{pid}/gaps").status_code == 404
    assert c.get(f"/projects/{pid}/search", params={"q": "swelling", "ranker": "advanced"}).status_code == 404


def test_the_default_ranker_is_unchanged_and_advanced_is_opt_in_with_explanations(lib):
    c, h, pid, ids = lib
    base = c.get(f"/projects/{pid}/search", params={"q": "silicon anode volume expansion", "k": 5}).json()
    assert base["hits"] and all("explanation" not in x for x in base["hits"])
    same = c.get(f"/projects/{pid}/search", params={"q": "silicon anode volume expansion", "k": 5, "ranker": "bm25"}).json()
    assert [x["passage_id"] for x in same["hits"]] == [x["passage_id"] for x in base["hits"]]
    adv = c.get(f"/projects/{pid}/search", params={"q": "silicon anode volume expansion", "k": 5, "ranker": "advanced"}).json()
    assert adv["hits"] and "opt-in" in adv["method"] and all(x["explanation"]["components"] for x in adv["hits"])
    assert adv["hits"][0]["document_id"] == ids["bat1"]
    assert c.get(f"/projects/{pid}/search", params={"q": "swelling", "ranker": "nonsense"}).status_code == 422
    # vocabulary the document does not use: advanced reaches bat1 through expansion terms it names; the default ranker cannot
    plain = {x["document_id"] for x in c.get(f"/projects/{pid}/search", params={"q": "swelling", "k": 20}).json()["hits"]}
    exp = c.get(f"/projects/{pid}/search", params={"q": "swelling", "k": 20, "ranker": "advanced"}).json()["hits"]
    assert ids["bat1"] not in plain and ids["bat1"] in {x["document_id"] for x in exp}
