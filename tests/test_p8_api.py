"""P8 Gate G (API): datasets with rights, tokenizer preparation, training on a user dataset, evaluation, lifecycle with real evidence,
provider selection and its effect on runs, authorization and limits. Training runs in a real subprocess on a tiny model (CPU)."""
import contextlib
import json
import random
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings
from api.models import ModelEntry
from tests.test_api import QUESTION, make_index, wait_done
from tests.test_api_models import PW, done, sign_in, wait_for


@contextlib.contextmanager
def guest(app):
    """A second browser. Deliberately NOT `with TestClient(app)`: leaving that block would run the app's shutdown (stopping the
    embedded worker that the module's main client still needs)."""
    yield TestClient(app, base_url="http://testserver")


def account(c, email):
    """Register, or sign in if an earlier test already created the account (accounts live as long as the module's app)."""
    r = c.post("/auth/register", json={"email": email, "password": PW})
    if r.status_code != 201:
        r = c.post("/auth/login", json={"email": email, "password": PW})
        assert r.status_code == 200, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}

WORDS = ("temperature conductivity electrolyte sample measured increased decreased protocol replication uncertainty instrument "
         "calibration annealing viscosity catalyst spectrum absorption regression coefficient significant control group placebo "
         "treatment outcome variance estimate cohort survey review evidence claim status supported contradicted hypothesis "
         "falsifiable variable assumption criterion analysis phase transition calorimetry cycle degradation activity yield").split()


def make_docs(n=60, seed=0):
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        sentences = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 14))).capitalize() + "." for _ in range(rng.randint(6, 9))]
        docs.append({"title": f"doc {i}", "text": " ".join(sentences)})
    return docs


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("p8api")
    cfgs = root / "configs"
    cfgs.mkdir()
    cfg = {"description": "tiny dataset-training config",
           "model": {"vocab_size": "auto", "max_seq_len": 32, "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "dropout": 0.0},
           "training": {"run_name": "x", "seed": 1, "batch_size": 8, "grad_accum_steps": 1, "max_steps": 150, "learning_rate": 0.004,
                        "warmup_steps": 5, "eval_interval": 50, "eval_batches": 4, "checkpoint_interval": 50, "log_interval": 10,
                        "mixed_precision": "none", "device": "cpu"},
           "data": {"processed_dir": str(root / "unused"), "tokenizer_dir": str(root / "unused")}}
    (cfgs / "tiny.yaml").write_text(yaml.safe_dump(cfg))
    s = Settings(database_url="sqlite:///:memory:", engine_db=":memory:", data_dir=root / "data", corpus_isolate=False,
                 models_dir=root / "operator_models", tokenizer_dir=root / "optok", train_configs_dir=cfgs, training_enabled=True,
                 max_train_steps=5000, train_stop_grace_s=30.0, admin_emails=("alice@example.org",), dataset_max_docs=100,
                 dataset_max_total_chars=60_000, dataset_max_per_user=3)
    app = create_app(s, index_provider=make_index)
    with TestClient(app, base_url="http://testserver") as c:
        h = sign_in(c)
        body = {"name": "synthetic words", "license": "CC0-1.0", "attribution": "generated for tests", "rights_confirmed": True,
                "documents": make_docs()}
        ds = c.post("/modellab/datasets", json=body, headers=h)
        assert ds.status_code == 201, ds.text
        did = ds.json()["id"]
        assert c.post(f"/modellab/datasets/{did}/prepare", json={"vocab_size": 400, "min_chars": 50}, headers=h).status_code == 202
        prepared = wait_for(c, f"/modellab/datasets/{did}", lambda b: b["status"] in ("prepared", "failed"), 120)
        assert prepared["status"] == "prepared", prepared
        job = c.post("/training/jobs", json={"name": "on user data", "base_config": "tiny", "dataset_id": did}, headers=h)
        assert job.status_code == 201, job.text
        final = done(c, job.json()["id"], 240)
        assert final["status"] == "succeeded", final
        yield {"app": app, "c": c, "h": h, "dataset": prepared, "job": final, "root": root, "settings": s, "mid": final["id"], "did": did}


# ------------------------------------------------------------------ datasets and tokenizer
def test_dataset_requires_rights_license_attribution_and_respects_limits(env):
    c, h = env["c"], env["h"]
    ok = {"name": "x", "license": "CC0-1.0", "attribution": "me", "rights_confirmed": True, "documents": [{"text": "hello world " * 20}]}
    assert c.post("/modellab/datasets", json={**ok, "rights_confirmed": False}, headers=h).status_code == 422
    assert c.post("/modellab/datasets", json={**ok, "license": "all-rights-reserved"}, headers=h).status_code == 422
    assert c.post("/modellab/datasets", json={**ok, "license": "CC-BY-4.0", "attribution": ""}, headers=h).status_code == 422
    assert c.post("/modellab/datasets", json={**ok, "name": "../etc"}, headers=h).status_code == 422
    big = c.post("/modellab/datasets", json={**ok, "documents": [{"text": "x" * 40_000}, {"text": "y" * 40_000}]}, headers=h)
    assert big.status_code == 422 and "limited" in big.json()["detail"]
    assert c.post("/modellab/datasets", json={**ok, "documents": []}, headers=h).status_code == 422
    assert c.post("/modellab/datasets", json=ok).status_code in (401, 403)  # CSRF/auth enforced


def test_prepared_dataset_records_manifest_split_and_tokenizer(env):
    c, ds = env["c"], env["dataset"]
    st = ds["stats"]
    assert st["documents"]["train"] > 0 and st["documents"]["val"] > 0 and st["documents"]["test"] > 0
    assert st["tokenizer_corpus_sha256"] and st["manifest_sha256"] == ds["manifest_sha256"] and st["pretokenizer"] == "v2"
    tk = c.get(f"/modellab/datasets/{ds['id']}/tokenizer").json()
    assert tk["manifest"]["fingerprint"] == st["tokenizer_fingerprint"]
    assert tk["manifest"]["training_corpus"]["corpus_sha256"] == st["tokenizer_corpus_sha256"]
    assert tk["statistics"]["merged_tokens"] > 0 and "training split only" in tk["trained_on"]
    proc = env["settings"].data_dir / "datasets" / ds["id"] / "processed"
    splits = json.loads((proc / "splits.json").read_text())
    assert not (set(splits["train"]) & set(splits["val"])) and not (set(splits["train"]) & set(splits["test"]))


def test_datasets_are_private_to_their_owner_and_deletable(env):
    c, app = env["c"], env["app"]
    with guest(app) as other:
        h2 = account(other, "bob@example.org")
        assert other.get(f"/modellab/datasets/{env['did']}").status_code == 404
        assert other.delete(f"/modellab/datasets/{env['did']}", headers=h2).status_code == 404
        assert other.post(f"/modellab/datasets/{env['did']}/prepare", json={}, headers=h2).status_code == 404
        r = other.post("/training/jobs", json={"name": "steal", "base_config": "tiny", "dataset_id": env["did"]}, headers=h2)
        assert r.status_code == 404
    assert env["did"] in [d["id"] for d in c.get("/modellab/datasets").json()["datasets"]]
    extra = c.post("/modellab/datasets", json={"name": "tmp", "license": "user-owned", "rights_confirmed": True,
                                               "documents": [{"text": "short text here " * 30}]}, headers=env["h"])
    assert extra.status_code == 201
    assert c.delete(f"/modellab/datasets/{extra.json()['id']}", headers=env["h"]).status_code == 204
    assert c.get(f"/modellab/datasets/{extra.json()['id']}").status_code == 404


def test_unprepared_or_tiny_datasets_cannot_start_training(env):
    c, h = env["c"], env["h"]
    raw = c.post("/modellab/datasets", json={"name": "unprepared", "license": "user-owned", "rights_confirmed": True,
                                             "documents": [{"text": "alpha beta gamma " * 40}]}, headers=h).json()
    r = c.post("/training/jobs", json={"name": "t", "base_config": "tiny", "dataset_id": raw["id"]}, headers=h)
    assert r.status_code == 409 and "prepare" in r.json()["detail"]
    assert c.post("/training/jobs", json={"name": "t", "base_config": "tiny", "dataset_id": "dts_notreal"}, headers=h).status_code == 422
    c.delete(f"/modellab/datasets/{raw['id']}", headers=h)


def test_dataset_quota(env):
    c, h = env["c"], env["h"]
    made = []
    for i in range(5):
        r = c.post("/modellab/datasets", json={"name": f"q{i}", "license": "user-owned", "rights_confirmed": True,
                                               "documents": [{"text": "quota filler " * 30}]}, headers=h)
        made.append(r)
        if r.status_code != 201:
            break
    assert any(r.status_code == 429 for r in made)
    for r in made:
        if r.status_code == 201:
            c.delete(f"/modellab/datasets/{r.json()['id']}", headers=h)


# ------------------------------------------------------------------ training on the dataset
def test_job_used_the_datasets_own_tokenizer_and_data(env):
    c, mid, did = env["c"], env["mid"], env["did"]
    assert env["job"]["config"]["dataset_id"] == did
    d = c.get(f"/modellab/models/{mid}").json()
    tk = c.get(f"/modellab/datasets/{did}/tokenizer").json()["manifest"]
    assert d["config"]["tokenizer"]["fingerprint"] == tk["fingerprint"]
    assert d["config"]["lineage_root"] == "random_init" and d["config"]["memory_estimate"]["basis"] == "estimate"
    assert d["run"]["completion_state"] == "completed" and d["run"]["dataset_fingerprint"]
    gen = c.post(f"/models/{mid}/generate", json={"prompt": "temperature", "max_new_tokens": 8, "temperature": 0}, headers=env["h"])
    assert gen.status_code == 200, gen.text  # served with the dataset's tokenizer, not the operator's


# ------------------------------------------------------------------ lifecycle, evaluation, selection
def test_lifecycle_follows_the_artifacts_and_refuses_unearned_states(env):
    c, h, mid = env["c"], env["h"], env["mid"]
    lc = c.get(f"/modellab/models/{mid}/lifecycle").json()
    assert lc["state"] == "TRAINED"  # derived from a completed run, not from initialisation
    assert [x["to"] for x in lc["history"]][:3] == ["INITIALIZED", "TRAINING", "TRAINED"]
    r = c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "VALIDATED"}, headers=h)
    assert r.status_code == 409 and "baseline" in r.json()["detail"]  # no evaluation yet
    assert c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "DEPLOYED"}, headers=h).status_code == 409
    assert c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "NONSENSE"}, headers=h).status_code == 422
    assert c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "TRAINED"}, headers=h).status_code == 409


def test_evaluation_validates_and_a_deploy_needs_an_administrator(env):
    c, h, mid, app = env["c"], env["h"], env["mid"], env["app"]
    assert c.get(f"/modellab/models/{mid}/evaluation").json()["state"] == "none"
    assert c.post(f"/modellab/models/{mid}/evaluate", headers=h).status_code == 202
    ev = wait_for(c, f"/modellab/models/{mid}/evaluation", lambda b: b["state"] in ("done", "failed"), 180)
    assert ev["state"] == "done", ev
    rep = ev["report"]
    assert rep["beats_random_baseline"] and rep["validation_loss"] < rep["baseline_validation_loss"]
    assert set(rep["tasks"]) >= {"general_lm", "structured_output"} and rep["agent_integration"]["stage_reached"] in (0, 1, 2)
    assert rep["inference"]["cached"]["tokens_per_second"] > 0 and "none" in rep["comparison_with_other_models"]
    ok = c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "VALIDATED", "note": "tests"}, headers=h)
    assert ok.status_code == 200, ok.text
    assert ok.json()["state"] == "VALIDATED" and ok.json()["evidence"]["val_loss"] == rep["validation_loss"]
    # a non-admin owner cannot deploy: use a second account that owns nothing here
    with guest(app) as other:
        h2 = account(other, "bob@example.org")
        assert other.post(f"/modellab/models/{mid}/lifecycle", json={"target": "DEPLOYED"}, headers=h2).status_code == 404  # not theirs
    dep = c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "DEPLOYED"}, headers=h)  # alice is in admin_emails
    assert dep.status_code == 200 and dep.json()["state"] == "DEPLOYED" and dep.json()["evidence"]["authorized_by"]
    back = c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "VALIDATED"}, headers=h)
    assert back.status_code == 200
    ev2 = c.get(f"/modellab/models/{mid}/evaluation").json()["report"]
    assert ev2["checkpoint_sha256"]


def test_validation_is_withdrawn_when_the_checkpoint_changes(env):
    app, mid = env["app"], env["mid"]
    with app.state.sessionmaker() as db:
        row = db.get(ModelEntry, mid)
        row.evaluation = {**row.evaluation, "checkpoint_sha256": "0" * 64}  # evaluation of some other weights
        db.commit()
    c, h = env["c"], env["h"]
    c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "TRAINED"}, headers=h)
    r = c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "VALIDATED"}, headers=h)
    assert r.status_code == 409 and "baseline" in r.json()["detail"]
    with app.state.sessionmaker() as db:  # restore for later tests
        row = db.get(ModelEntry, mid)
        from model.common import sha256_file  # noqa: F401
        ck = env["settings"].data_dir / "training" / mid / "run"
        best = json.loads((ck / "best.json").read_text())["path"]
        sha = json.loads((ck / best / "meta.json").read_text())["model_sha256"]
        row.evaluation = {**row.evaluation, "checkpoint_sha256": sha}
        db.commit()
    assert c.post(f"/modellab/models/{mid}/lifecycle", json={"target": "VALIDATED"}, headers=h).status_code == 200


def test_selection_rules_and_provider_status(env):
    c, h, mid = env["c"], env["h"], env["mid"]
    prov = c.get("/modellab/provider").json()
    assert prov["effective"]["provider"] == "deterministic" and prov["never_uses_hosted_api"] is True
    assert c.put("/modellab/selection", json={"provider": "custom_transformer"}, headers=h).status_code == 422
    r = c.put("/modellab/selection", json={"provider": "custom_transformer", "model_id": mid, "mode": "experimental"}, headers=h)
    assert r.status_code == 409 and "allow_experimental" in r.json()["detail"]
    r = c.put("/modellab/selection", json={"provider": "custom_transformer", "model_id": mid, "mode": "production"}, headers=h)
    assert r.status_code == 409  # VALIDATED, not DEPLOYED, and no measured stage 3
    r = c.put("/modellab/selection", json={"provider": "custom_transformer", "model_id": mid, "mode": "experimental",
                                           "allow_experimental": True}, headers=h)
    assert r.status_code == 200 and r.json()["provider"] == "custom_transformer"
    prov = c.get("/modellab/provider").json()
    assert prov["effective"]["provider"] == "custom_transformer" and prov["effective"]["explicit_experimental"] is True
    assert prov["effective"]["capability"] == "validated" and len(prov["roles"]) == 9


def test_selected_model_attaches_untrusted_notes_without_changing_the_verdict(env):
    c, h = env["c"], env["h"]
    pid = c.post("/projects", json={"title": "with model", "question": QUESTION}, headers=h).json()["id"]
    rid = c.post(f"/projects/{pid}/runs", json={}, headers=h).json()["id"]
    wait_done(c, rid, h, 120)
    with_model = c.get(f"/runs/{rid}/report").json()
    assert c.put("/modellab/selection", json={"provider": "deterministic"}, headers=h).status_code == 200
    pid2 = c.post("/projects", json={"title": "plain", "question": QUESTION}, headers=h).json()["id"]
    rid2 = c.post(f"/projects/{pid2}/runs", json={}, headers=h).json()["id"]
    wait_done(c, rid2, h, 120)
    plain = c.get(f"/runs/{rid2}/report").json()
    key = lambda r: [(f["citation"]["passage_id"], f.get("text")) for fs in r["findings"].values() for f in fs]  # noqa: E731
    assert key(with_model) == key(plain)  # evidence layer identical with and without the model
    events = c.get(f"/runs/{rid}/events").json()
    assert any(e["type"] in ("model_call", "model_call_skipped", "model_note_rejected") for e in events)
    assert not any(e["type"].startswith("model_") for e in c.get(f"/runs/{rid2}/events").json())
    assert all(n.get("trusted") is False for n in with_model.get("model_notes", []))


def test_stale_selection_degrades_to_deterministic_and_is_reported(env):
    c, h, mid, app = env["c"], env["h"], env["mid"], env["app"]
    assert c.put("/modellab/selection", json={"provider": "custom_transformer", "model_id": mid, "allow_experimental": True},
                 headers=h).status_code == 200
    with app.state.sessionmaker() as db:
        row = db.get(ModelEntry, mid)
        row.state = "ARCHIVED"
        db.commit()
    prov = c.get("/modellab/provider").json()
    assert prov["effective"]["provider"] == "deterministic" and prov["error"] and "ARCHIVED" in prov["error"]
    from api.models import Project
    with app.state.sessionmaker() as db:
        pid = db.query(Project).first().id
    assert app.state.modellab.provider_for_project(pid) == (None, None)
    with app.state.sessionmaker() as db:
        row = db.get(ModelEntry, mid)
        row.state = "VALIDATED"
        db.commit()
    c.put("/modellab/selection", json={"provider": "deterministic"}, headers=h)


# ------------------------------------------------------------------ overview, profiles, checkpoints, authorization
def test_overview_and_profiles_report_real_values(env):
    c, mid = env["c"], env["mid"]
    ov = c.get("/modellab/overview").json()
    assert ov["hardware"]["cpu_count"] and ov["training_enabled"] and ov["is_admin"] is True
    assert mid in [m["model_id"] for m in ov["models"]] and ov["datasets"] >= 1
    profs = {p["name"]: p for p in c.get("/modellab/profiles").json()}
    assert set(profs) == {"CPU_DEBUG", "CONSUMER_SMALL", "CONSUMER_MEDIUM"}
    assert profs["CPU_DEBUG"]["parameters"] < profs["CONSUMER_MEDIUM"]["parameters"]
    assert all(p["fit_check"]["basis"] == "estimate" for p in profs.values())
    assert c.get("/modellab/hardware-benchmarks").status_code == 200


def test_checkpoint_deletion_rules(env):
    c, h, mid = env["c"], env["h"], env["mid"]
    cks = c.get(f"/models/{mid}/checkpoints").json()
    keep = [k for k in cks if k["is_best"] or k["is_latest"]][0]
    assert c.delete(f"/modellab/models/{mid}/checkpoints/{keep['name']}", headers=h).status_code == 409
    assert c.delete(f"/modellab/models/{mid}/checkpoints/..%2F..%2Fx", headers=h).status_code in (404, 422)
    assert c.delete(f"/modellab/models/{mid}/checkpoints/not_a_checkpoint", headers=h).status_code == 404
    spare = [k for k in cks if not k["is_best"] and not k["is_latest"]]
    if spare:
        assert c.delete(f"/modellab/models/{mid}/checkpoints/{spare[0]['name']}", headers=h).status_code == 204
        assert spare[0]["name"] not in [k["name"] for k in c.get(f"/models/{mid}/checkpoints").json()]
    assert c.delete("/modellab/models/sys_anything/checkpoints/step_0000001", headers=h).status_code == 403


def test_every_modellab_route_requires_authentication(env):
    app = env["app"]
    with guest(app) as anon:
        for method, url in [("get", "/modellab/overview"), ("get", "/modellab/profiles"), ("get", f"/modellab/models/{env['mid']}"),
                            ("get", "/modellab/datasets"), ("get", "/modellab/provider"), ("get", f"/modellab/models/{env['mid']}/evaluation")]:
            assert getattr(anon, method)(url).status_code == 401, url
        for method, url in [("post", f"/modellab/models/{env['mid']}/evaluate"), ("put", "/modellab/selection"),
                            ("post", "/modellab/datasets")]:
            assert getattr(anon, method)(url, json={}).status_code in (401, 403), url


def test_other_users_cannot_touch_my_model(env):
    app, mid = env["app"], env["mid"]
    with guest(app) as other:
        h2 = account(other, "mallory@example.org")
        for method, url, body in [("get", f"/modellab/models/{mid}", None), ("get", f"/modellab/models/{mid}/lifecycle", None),
                                  ("post", f"/modellab/models/{mid}/evaluate", None),
                                  ("post", f"/modellab/models/{mid}/lifecycle", {"target": "ARCHIVED"}),
                                  ("delete", f"/modellab/models/{mid}/checkpoints/step_0000050", None)]:
            r = getattr(other, method)(url, headers=h2, **({"json": body} if body else {}))
            assert r.status_code == 404, (method, url, r.status_code)
        r = other.put("/modellab/selection", json={"provider": "custom_transformer", "model_id": mid, "allow_experimental": True}, headers=h2)
        assert r.status_code == 404
        assert other.get("/modellab/overview").json()["models"] == []


def test_selection_rejects_paths_and_unknown_ids(env):
    c, h = env["c"], env["h"]
    for bad in ["../../etc/passwd", "sys_../x", "trn_ZZZ", "C:\\models\\x"]:
        assert c.put("/modellab/selection", json={"provider": "custom_transformer", "model_id": bad}, headers=h).status_code == 422
    assert c.put("/modellab/selection", json={"provider": "custom_transformer", "model_id": "trn_" + "0" * 16,
                                              "allow_experimental": True}, headers=h).status_code == 404


def test_audit_log_records_modellab_actions(env):
    from api.models import AuditLog
    with env["app"].state.sessionmaker() as db:
        actions = {a.action for a in db.query(AuditLog).all()}
    assert {"dataset_create", "dataset_prepare", "model_lifecycle", "model_evaluate", "model_select"} <= actions
    assert "model_lifecycle_refused" in actions or "model_select_refused" in actions
