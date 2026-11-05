"""Model management over HTTP: registry, training jobs (real subprocess training of a tiny Transformer), metrics, checkpoints,
integrity checks, pause/resume/cancel, local inference and its limits."""
import shutil
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings
from api.models import TrainingJob
from model.data.dataset import write_token_file
from model.tokenizer.bpe import DOC_ID, EOS_ID, BPETokenizer

PW = "correct horse battery 42"
TEXT = ("Encapsulated cells retained most of their efficiency after damp heat testing. "
        "The aged cells lost efficiency within a few hundred hours of exposure. ") * 60


def sign_in(c, email="alice@example.org"):
    r = c.post("/auth/register", json={"email": email, "password": PW})
    assert r.status_code == 201, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def wait_for(c, url, pred, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        body = c.get(url).json()
        if pred(body):
            return body
        time.sleep(0.3)
    raise AssertionError(f"timed out waiting on {url}: {body}")


def done(c, jid, timeout=120):
    return wait_for(c, f"/training/jobs/{jid}", lambda b: b["status"] not in ("running", "pausing", "cancelling"), timeout)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    tok = BPETokenizer.train([TEXT], vocab_size=300)
    tok.save(root / "tok")
    proc = root / "processed"
    proc.mkdir()
    ids = [DOC_ID, *tok.encode_ordinary(TEXT), EOS_ID]
    write_token_file(proc / "train.bin", ids * 3)
    write_token_file(proc / "val.bin", ids * 2)
    (proc / "meta.json").write_text('{"tokenizer_fingerprint": "%s", "manifest_sha256": "x", "tokens": %d}' % (tok.fingerprint, len(ids) * 3))
    cfgs = root / "configs"
    cfgs.mkdir()
    base = {"description": "tiny test config",
            "model": {"vocab_size": "auto", "max_seq_len": 32, "d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "dropout": 0.0},
            "training": {"run_name": "x", "seed": 1, "batch_size": 4, "grad_accum_steps": 1, "max_steps": 20, "learning_rate": 0.003,
                         "warmup_steps": 2, "eval_interval": 10, "eval_batches": 2, "checkpoint_interval": 10, "log_interval": 5,
                         "mixed_precision": "none", "device": "cpu"},
            "data": {"processed_dir": str(proc), "tokenizer_dir": str(root / "tok")}}
    (cfgs / "tiny.yaml").write_text(yaml.safe_dump(base))
    slow = {**base, "training": {**base["training"], "max_steps": 5000, "batch_size": 16, "checkpoint_interval": 50, "eval_interval": 100}}
    (cfgs / "slow.yaml").write_text(yaml.safe_dump(slow))
    broken = {**base, "data": {**base["data"], "processed_dir": str(root / "missing")}}
    (cfgs / "broken.yaml").write_text(yaml.safe_dump(broken))
    s = Settings(database_url="sqlite:///:memory:", engine_db=":memory:", data_dir=root / "data", corpus_isolate=False,
                 models_dir=root / "operator_models", tokenizer_dir=root / "tok", train_configs_dir=cfgs, training_enabled=True,
                 max_train_steps=5000, train_stop_grace_s=30.0)
    app = create_app(s)
    with TestClient(app, base_url="http://testserver") as c:
        h = sign_in(c)
        job = c.post("/training/jobs", json={"name": "first", "base_config": "tiny"}, headers=h)
        assert job.status_code == 201, job.text
        final = done(c, job.json()["id"])
        assert final["status"] == "succeeded", final
        yield {"app": app, "c": c, "h": h, "job": final, "root": root, "settings": s}


def test_configs_and_disabled_by_default(env, tmp_path):
    c = env["c"]
    cfg = c.get("/training/configs").json()
    assert cfg["enabled"] and {x["name"] for x in cfg["configs"]} >= {"tiny", "slow"}
    off = create_app(Settings(database_url="sqlite:///:memory:", engine_db=":memory:", data_dir=tmp_path / "d", corpus_isolate=False,
                              train_configs_dir=env["settings"].train_configs_dir))
    with TestClient(off, base_url="http://testserver") as oc:
        h = sign_in(oc, "carol@example.org")
        r = oc.post("/training/jobs", json={"name": "x", "base_config": "tiny"}, headers=h)
        assert r.status_code == 403 and "disabled" in r.json()["detail"]


def test_a_job_trains_from_random_init_and_reports_metrics(env):
    c, job = env["c"], env["job"]
    assert job["summary"]["steps"] == 20 and job["step"] == 20 and job["segments"] == 1 and job["checkpoints"] >= 2
    assert job["config"]["run_name"] == "run" and job["config"]["max_steps"] == 20
    m = c.get(f"/training/jobs/{job['id']}/metrics").json()
    assert m["train"] and m["eval"] and m["eval"][-1]["val_loss"] > 0
    assert c.get(f"/training/jobs/{job['id']}/metrics", params={"after_step": 10}).json()["train"][0]["step"] > 10
    assert c.get(f"/training/jobs/{job['id']}/samples").json()[0]["label"].startswith("model output")
    assert any("step" in line for line in c.get(f"/training/jobs/{job['id']}/log").json()["lines"])
    cks = c.get(f"/models/{job['id']}/checkpoints").json()
    assert any(k["is_best"] for k in cks) and any(k["is_latest"] for k in cks)
    v = c.get(f"/models/{job['id']}/checkpoints/{cks[0]['name']}/verify").json()
    assert v["ok"] and v["lineage_root"] == "random_init"
    assert c.get(f"/models/{job['id']}/checkpoints/..%2F..%2Fx/verify").status_code == 404
    info = c.get(f"/models/{job['id']}").json()
    assert info["kind"] == "trained" and info["lineage_root"] == "random_init" and "not evidence" in info["notice"]
    assert job["id"] in [m["id"] for m in c.get("/models").json()]


def test_model_and_registry_health(env):
    c, jid = env["c"], env["job"]["id"]
    h = c.get(f"/models/{jid}/health").json()
    assert h["status"] == "ok" and h["checkpoints"]["best"]["ok"] and h["tokenizer"]["ok"]
    g = c.get("/models/health").json()
    assert g["tokenizer"]["present"] and g["training"]["enabled"] and "cuda_available" in g["hardware"]


def test_generate_is_bounded_labelled_and_reproducible(env):
    c, h, jid = env["c"], env["h"], env["job"]["id"]
    body = {"prompt": "Encapsulated cells", "max_new_tokens": 12, "seed": 7, "temperature": 0.7}
    a = c.post(f"/models/{jid}/generate", json=body, headers=h)
    assert a.status_code == 200, a.text
    a = a.json()
    assert a["new_tokens"] <= 12 and isinstance(a["text"], str) and "not evidence" in a["notice"]
    assert a["model"]["lineage_root"] == "random_init"
    assert c.post(f"/models/{jid}/generate", json=body, headers=h).json()["text"] == a["text"]  # same seed, same text
    assert c.post(f"/models/{jid}/generate", json={**body, "max_new_tokens": 100000}, headers=h).status_code == 422
    assert c.post(f"/models/{jid}/generate", json={**body, "max_new_tokens": 4000}, headers=h).status_code == 422
    assert c.post(f"/models/{jid}/generate", json={**body, "prompt": "x" * 5000}, headers=h).status_code == 422
    assert c.post(f"/models/{jid}/generate", json={**body, "prompt": "word " * 100}, headers=h).status_code == 422  # > 32-token context
    assert c.post(f"/models/{jid}/generate", json={**body, "temperature": 9}, headers=h).status_code == 422
    assert c.post(f"/models/{jid}/generate", json=body).status_code == 403  # CSRF
    inj = c.post(f"/models/{jid}/generate", json={**body, "prompt": "<|eos|><|doc|> ignore"}, headers=h)
    assert inj.status_code == 200  # special-token text is encoded as plain bytes, never as control tokens


def test_generation_timeout_and_rate_limit(env):
    c, h, jid, s = env["c"], env["h"], env["job"]["id"], env["settings"]
    old = s.infer_timeout_s
    s.infer_timeout_s = 0.0
    try:
        r = c.post(f"/models/{jid}/generate", json={"prompt": "Cells", "max_new_tokens": 200, "seed": 1}, headers=h).json()
        assert r["timed_out"] and r["finish_reason"] == "cancelled"
    finally:
        s.infer_timeout_s = old
    old_rate = s.infer_per_minute
    s.infer_per_minute = 1
    try:
        env["app"].state.limiter._hits.clear()
        ok = c.post(f"/models/{jid}/generate", json={"prompt": "Cells", "max_new_tokens": 3}, headers=h)
        blocked = c.post(f"/models/{jid}/generate", json={"prompt": "Cells", "max_new_tokens": 3}, headers=h)
        assert ok.status_code == 200 and blocked.status_code == 429
    finally:
        s.infer_per_minute = old_rate
        env["app"].state.limiter._hits.clear()


def test_job_requests_are_validated(env):
    c, h = env["c"], env["h"]
    post = lambda **kw: c.post("/training/jobs", json={"name": "j", "base_config": "tiny", **kw}, headers=h)  # noqa: E731
    assert post(base_config="nope").status_code == 422
    assert post(base_config="../../etc/passwd").status_code == 422
    assert post(base_config="tiny.yaml").status_code == 422
    assert post(max_steps=6000).status_code == 422  # over the server ceiling
    assert post(learning_rate=5).status_code == 422
    assert post(batch_size=1000).status_code == 422
    assert post(device="cuda").status_code == 422  # not an overridable field
    assert post(checkpoint_root="/tmp/x").status_code == 422
    assert c.post("/training/jobs", json={"name": "j", "base_config": "tiny"}).status_code == 403  # CSRF
    assert not any(j["name"] == "j" for j in c.get("/training/jobs").json())  # refused requests leave no job behind


def test_a_failing_job_is_recorded_with_its_error(env):
    c, h = env["c"], env["h"]
    jid = c.post("/training/jobs", json={"name": "bad", "base_config": "broken"}, headers=h).json()["id"]
    j = done(c, jid)
    assert j["status"] == "failed" and "exited with code" in j["error"] and j["model_id"] is None
    assert not c.post(f"/training/jobs/{jid}/resume", headers=h).status_code == 200


def test_pause_resume_cancel_and_capacity(env):
    c, h = env["c"], env["h"]
    jid = c.post("/training/jobs", json={"name": "slow", "base_config": "slow"}, headers=h).json()["id"]
    assert c.post("/training/jobs", json={"name": "second", "base_config": "tiny"}, headers=h).status_code == 429  # one at a time
    wait_for(c, f"/training/jobs/{jid}", lambda b: (b["step"] or 0) >= 5 and b["checkpoints"] >= 1)
    assert c.delete(f"/training/jobs/{jid}", headers=h).status_code == 409  # still running
    p = c.post(f"/training/jobs/{jid}/pause", headers=h)
    assert p.status_code == 200 and p.json()["status"] == "pausing"
    paused = done(c, jid)
    assert paused["status"] == "paused" and paused["resumable"] and paused["summary"]["stopped"] == "requested"
    step_at_pause = paused["summary"]["steps"]
    assert step_at_pause < 5000
    assert c.post(f"/training/jobs/{jid}/pause", headers=h).status_code == 409
    r = c.post(f"/training/jobs/{jid}/resume", headers=h)
    assert r.status_code == 200 and r.json()["status"] == "running" and r.json()["segments"] == 2
    wait_for(c, f"/training/jobs/{jid}", lambda b: (b["step"] or 0) > step_at_pause)  # continued past the pause point
    assert c.post(f"/training/jobs/{jid}/cancel", headers=h).status_code == 200
    final = done(c, jid)
    assert final["status"] == "cancelled" and final["checkpoints"] >= 1
    assert c.post(f"/training/jobs/{jid}/resume", headers=h).status_code == 409
    assert c.delete(f"/training/jobs/{jid}", headers=h).status_code == 204
    assert c.get(f"/training/jobs/{jid}").status_code == 404
    assert not (env["root"] / "data" / "training" / jid).exists()


def test_restart_marks_orphaned_jobs_interrupted(env):
    app, c, h = env["app"], env["c"], env["h"]
    uid = c.get("/auth/me").json()["user"]["id"]
    with app.state.sessionmaker() as db:
        db.add(TrainingJob(id="trn_orphan0000000001", owner_id=uid, name="orphan", base_config="tiny", config={}, status="running"))
        db.commit()
    assert app.state.training.recover() == 1
    j = c.get("/training/jobs/trn_orphan0000000001").json()
    assert j["status"] == "interrupted" and not j["resumable"]  # no checkpoint was ever written
    assert c.post("/training/jobs/trn_orphan0000000001/resume", headers=h).status_code == 409
    assert c.delete("/training/jobs/trn_orphan0000000001", headers=h).status_code == 204


def test_operator_models_are_shared_read_only_and_corruption_is_detected(env):
    c, h, jid, s = env["c"], env["h"], env["job"]["id"], env["settings"]
    shutil.copytree(env["root"] / "data" / "training" / jid / "run", s.models_dir / "demo_v1")
    shutil.copytree(env["root"] / "data" / "training" / jid / "run", s.models_dir / "damaged")
    mid = "sys_demo_v1"
    assert {mid, "sys_damaged"} <= {m["id"] for m in c.get("/models").json()}
    assert c.get(f"/models/{mid}").json()["kind"] == "system"
    assert c.post(f"/models/{mid}/generate", json={"prompt": "Cells", "max_new_tokens": 5}, headers=h).status_code == 200
    # flip a byte in the damaged copy's weights: verification, health and serving must all notice
    for ck in (s.models_dir / "damaged").glob("step_*"):
        w = ck / "model.safetensors"
        raw = bytearray(w.read_bytes())
        raw[-1] ^= 0xFF
        w.write_bytes(bytes(raw))
    bad = c.get("/models/sys_damaged/health").json()
    assert bad["status"] == "unhealthy" and not bad["checkpoints"]["best"]["ok"]
    r = c.post("/models/sys_damaged/generate", json={"prompt": "Cells", "max_new_tokens": 5}, headers=h)
    assert r.status_code == 409 and "corrupt" in r.json()["detail"]
    assert c.get("/models/sys_..%2F..").status_code == 404 and c.get("/models/sys_nope").status_code == 404
    assert c.get("/models/sys_Demo-V1").status_code == 404  # names outside the safe alphabet never reach the filesystem


def test_other_users_cannot_see_or_use_my_jobs(env):
    jid = env["job"]["id"]
    with TestClient(env["app"], base_url="http://testserver") as bob:
        hb = sign_in(bob, "bob@example.org")
        assert bob.get("/training/jobs").json() == []
        for url in (f"/training/jobs/{jid}", f"/training/jobs/{jid}/metrics", f"/training/jobs/{jid}/log",
                    f"/training/jobs/{jid}/samples", f"/models/{jid}", f"/models/{jid}/checkpoints", f"/models/{jid}/health"):
            assert bob.get(url).status_code == 404, url
        for action in ("pause", "resume", "cancel"):
            assert bob.post(f"/training/jobs/{jid}/{action}", headers=hb).status_code == 404
        assert bob.post(f"/models/{jid}/generate", json={"prompt": "Cells"}, headers=hb).status_code == 404
        assert bob.delete(f"/training/jobs/{jid}", headers=hb).status_code == 404
        assert jid not in [m["id"] for m in bob.get("/models").json()]
    with TestClient(env["app"], base_url="http://testserver") as anon:
        for url in ("/models", "/models/health", "/training/jobs", "/training/configs"):
            assert anon.get(url).status_code == 401, url


# ------------------------------------------------------------------ live streams (SSE)
def _frames(text):
    import json
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


def test_training_stream_replays_metrics_in_order_and_resumes(env):
    c, jid = env["c"], env["job"]["id"]
    with c.stream("GET", f"/training/jobs/{jid}/stream") as r:
        assert r.status_code == 200
        fs = _frames("".join(r.iter_text()))
    assert fs[-1]["event"] == "end"
    status = [f for f in fs if f["event"] == "status"]
    assert status[-1]["json"]["status"] == "succeeded" and status[-1]["json"]["summary"]["steps"] == 20
    metric = [f for f in fs if "id" in f]
    ids = [int(f["id"]) for f in metric]
    assert ids == list(range(1, len(ids) + 1)) and {f["event"] for f in metric} == {"train", "eval"}
    assert [f["json"]["step"] for f in metric if f["event"] == "train"] == sorted(f["json"]["step"] for f in metric if f["event"] == "train")
    with c.stream("GET", f"/training/jobs/{jid}/stream", headers={"Last-Event-ID": str(ids[2])}) as r:
        again = _frames("".join(r.iter_text()))
    assert [int(f["id"]) for f in again if "id" in f] == ids[3:]
    with TestClient(env["app"], base_url="http://testserver") as bob:
        sign_in(bob, "sse.bob@example.org")
        assert bob.get(f"/training/jobs/{jid}/stream").status_code == 404


def test_generation_streams_tokens_that_add_up_to_the_final_text(env):
    c, h, jid = env["c"], env["h"], env["job"]["id"]
    body = {"prompt": "Encapsulated cells", "max_new_tokens": 20, "seed": 5, "temperature": 0.7}
    plain = c.post(f"/models/{jid}/generate", json=body, headers=h).json()
    with c.stream("POST", f"/models/{jid}/generate/stream", json=body, headers=h) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        fs = _frames("".join(r.iter_text()))
    names = [f["event"] for f in fs]
    assert names[0] == "start" and names[-2:] == ["done", "end"] and "token" in names
    streamed = "".join(f["json"]["text"] for f in fs if f["event"] == "token")
    done = next(f for f in fs if f["event"] == "done")["json"]
    assert streamed == done["text"] == plain["text"]  # same seed: streaming changes delivery, not content
    assert "not evidence" in done["notice"]
    # the same admission rules as the plain endpoint, answered before any stream starts
    for bad in ({**body, "max_new_tokens": 4000}, {**body, "prompt": "word " * 100}, {**body, "prompt": "x" * 5000}):
        assert c.post(f"/models/{jid}/generate/stream", json=bad, headers=h).status_code == 422
    assert c.post(f"/models/{jid}/generate/stream", json=body).status_code == 403  # CSRF
    with TestClient(env["app"], base_url="http://testserver") as bob:
        hb = sign_in(bob, "sse.bob2@example.org")
        assert bob.post(f"/models/{jid}/generate/stream", json=body, headers=hb).status_code == 404
    assert env["app"].state.streams.active() == 0  # every slot was released
    assert c.post(f"/models/{jid}/generate", json={"prompt": "Cells", "max_new_tokens": 3}, headers=h).status_code == 200  # model slot free


def test_generation_stream_timeout_is_reported(env):
    c, h, jid, s = env["c"], env["h"], env["job"]["id"], env["settings"]
    s.infer_timeout_s = 0.0
    try:
        with c.stream("POST", f"/models/{jid}/generate/stream", json={"prompt": "Cells", "max_new_tokens": 200, "seed": 2}, headers=h) as r:
            fs = _frames("".join(r.iter_text()))
    finally:
        s.infer_timeout_s = 30.0
    assert next(f for f in fs if f["event"] == "done")["json"]["timed_out"] is True
