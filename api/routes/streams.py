"""Server-Sent Events: live run events, training metrics, experiment status/logs and token-by-token generation.

All of these stream state that the REST endpoints can also read, so a client can always fall back to polling; reconnecting with
`Last-Event-ID` (EventSource does this automatically) resumes without gaps for runs and training."""
from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from api.deps import DB, CurrentAuth, audit
from api.model_service import ACTIVE, NOTICE, Busy, ModelNotFound, pick_checkpoint
from api.routes.experiments import owned_experiment
from api.routes.models import _resolve, _svc, admit_generation, owned_job
from api.routes.runs import owned_run
from api.schemas import GenerateIn
from api.sse import HEADERS, Item, resume_point, sse_format, start_stream, session_checker

router = APIRouter(tags=["streams"])
RUN_TERMINAL = ("completed", "failed", "cancelled")
EXP_TERMINAL = ("succeeded", "failed", "timed_out", "error")
LOG_CHUNK = 16_384


# ------------------------------------------------------------------ runs
@router.get("/runs/{run_id}/stream")
def stream_run(run_id: str, request: Request, auth: CurrentAuth, db: DB, after: int = Query(0, ge=0)):
    r = owned_run(db, run_id, auth.user.id)
    rt, engine_run_id = request.app.state.runtime, r.engine_run_id
    cursor = [resume_point(request, after)]
    last_state: list[str | None] = [None]

    def poll():
        events = rt.store.list_events(engine_run_id, cursor[0], 200)
        items = [Item(e["type"], e, e["id"]) for e in events]
        if events:
            cursor[0] = events[-1]["id"]
        state = rt.status(engine_run_id)
        snap = json.dumps(state, sort_keys=True, default=str)
        if snap != last_state[0]:  # progress counters change without an event; send the snapshot only when it differs
            last_state[0] = snap
            items.append(Item("state", state))
        done = state["status"] in RUN_TERMINAL and not state["active"] and not events
        return items, done

    return start_stream(request, auth, poll)


# ------------------------------------------------------------------ training
@router.get("/training/jobs/{job_id}/stream")
def stream_training(job_id: str, request: Request, auth: CurrentAuth, db: DB, after: int = Query(0, ge=0)):
    job = owned_job(db, job_id, auth.user.id)
    training, _, _ = _svc(request)
    metrics = training.root / job.id / "run" / "metrics.jsonl"
    maker = request.app.state.sessionmaker
    cursor = [resume_point(request, after)]  # = number of metric lines already delivered; also the SSE id
    last_status: list[str | None] = [None]

    def poll():
        from api.models import TrainingJob
        with maker() as s:  # status first, then metrics: if the job is already inactive the file below is final
            j = s.get(TrainingJob, job_id)
            status, error, summary = (j.status, j.error, j.summary) if j else ("deleted", "", None)
        items: list[Item] = []
        if metrics.is_file():
            lines = metrics.read_text(encoding="utf-8", errors="replace").splitlines()
            complete = lines if (not lines or lines[-1].endswith("}")) else lines[:-1]  # skip a half-written line
            for n in range(cursor[0], min(len(complete), cursor[0] + 200)):
                try:
                    rec = json.loads(complete[n])
                except ValueError:
                    rec = None
                cursor[0] = n + 1
                if isinstance(rec, dict):
                    items.append(Item(rec.get("type", "metric"), rec, n + 1))
        if status != last_status[0]:
            last_status[0] = status
            items.append(Item("status", {"status": status, "error": error, "summary": summary}))
        more = metrics.is_file() and cursor[0] < len(metrics.read_text(encoding="utf-8", errors="replace").splitlines())
        return items, status not in ACTIVE and not more

    return start_stream(request, auth, poll)


# ------------------------------------------------------------------ experiments
@router.get("/experiments/{exp_id}/stream")
def stream_experiment(exp_id: str, request: Request, auth: CurrentAuth, db: DB, stdout: int = Query(0, ge=0), stderr: int = Query(0, ge=0)):
    e = owned_experiment(db, exp_id, auth.user.id)
    run_dir: Path = request.app.state.experiments.run_dir(e.id)
    maker = request.app.state.sessionmaker
    offsets = {"stdout": stdout, "stderr": stderr}
    last_status: list[str | None] = [None]

    def poll():
        from api.models import Experiment
        with maker() as s:
            x = s.get(Experiment, exp_id)
            status = x.status if x else "deleted"
        items: list[Item] = []
        for name in ("stdout", "stderr"):
            p = run_dir / f"{name}.log"
            if p.is_file() and not p.is_symlink():
                with open(p, "rb") as f:
                    f.seek(offsets[name])
                    chunk = f.read(LOG_CHUNK)
                if chunk:  # untrusted program output: only ever sent as an escaped JSON string
                    items.append(Item("log", {"stream": name, "offset": offsets[name], "text": chunk.decode("utf-8", errors="replace")}))
                    offsets[name] += len(chunk)
        if status != last_status[0]:
            last_status[0] = status
            items.append(Item("status", {"status": status}))
        drained = all(not (run_dir / f"{n}.log").is_file() or offsets[n] >= (run_dir / f"{n}.log").stat().st_size for n in offsets)
        return items, (status in EXP_TERMINAL or status == "deleted") and drained

    return start_stream(request, auth, poll)


# ------------------------------------------------------------------ token streaming
@router.post("/models/{model_id}/generate/stream")
async def stream_generate(model_id: str, body: GenerateIn, request: Request, auth: CurrentAuth, db: DB):
    """Generation as an SSE stream (use fetch, since EventSource cannot POST). Events: `start`, `token` (a text delta), `done`."""
    _, _, inference = _svc(request)
    s = request.app.state.settings
    run_dir, _, _ = _resolve(request, db, auth, model_id)
    admit_generation(request, auth, body)
    from model.inference.runtime import ContextOverflowError, TokenizerMismatchError
    from model.tokenizer.bpe import DOC_ID
    from model.training.checkpoint import CheckpointError
    gate = request.app.state.streams
    slot = gate.acquire(auth.user.id)
    gate_released = False
    try:
        try:
            ck = pick_checkpoint(run_dir, body.checkpoint)
            rt = await anyio.to_thread.run_sync(inference.runtime_for, ck)
            ids = [DOC_ID, *rt.tokenizer.encode(body.prompt)]
            if len(ids) >= rt.model.cfg.max_seq_len:
                raise ContextOverflowError(f"prompt has {len(ids)} tokens; model context is {rt.model.cfg.max_seq_len}")
            await anyio.to_thread.run_sync(inference.acquire_slot)
        except ModelNotFound as e:
            raise HTTPException(404, str(e)) from e
        except Busy as e:
            raise HTTPException(429, str(e)) from e
        except ContextOverflowError as e:
            raise HTTPException(422, str(e)) from e
        except (TokenizerMismatchError, CheckpointError) as e:
            raise HTTPException(409, f"this checkpoint cannot be served: {e}") from e
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(503, f"model is not available: {e}") from e
    except BaseException:
        gate.release(slot)
        raise
    audit(db, request, "model_generate_stream", auth.user.id, model_id, prompt_chars=len(body.prompt))

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    timer = threading.Timer(s.infer_timeout_s, cancel.set)
    timer.daemon = True

    def work() -> None:  # runs the model in a worker thread and hands events to the event loop
        sent = ""
        all_ids: list[int] = []
        put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)  # noqa: E731
        try:
            for emit, finish in rt.stream_events(ids, body.max_new_tokens, body.temperature, body.top_k, top_p=body.top_p,
                                                 repetition_penalty=body.repetition_penalty, seed=body.seed, cancel_event=cancel):
                all_ids.extend(emit)
                text = rt.tokenizer.decode(all_ids)
                stable = text if finish is not None else text.rstrip("�")  # hold back a half-decoded UTF-8 character
                if len(stable) > len(sent) and stable.startswith(sent):
                    put(Item("token", {"index": len(all_ids), "text": stable[len(sent):]}))
                    sent = stable
                if finish is not None:
                    put(Item("done", {"finish_reason": finish, "prompt_tokens": len(ids), "new_tokens": len(all_ids),
                                      "timed_out": finish == "cancelled", "text": text,
                                      "model": {"step": rt.meta.get("step"), "model_sha256": rt.meta.get("model_sha256")},
                                      "notice": NOTICE}))
        except Exception as e:
            put(Item("error", {"message": f"generation failed: {type(e).__name__}"}))
        finally:
            put(None)
            timer.cancel()
            inference.release_slot()

    check = session_checker(request, auth.session.id)

    async def body_gen():
        threading.Thread(target=work, daemon=True, name="generate-stream").start()
        timer.start()
        yield "retry: 3000\n\n"
        yield sse_format(Item("start", {"model_id": model_id, "checkpoint": ck.name, "max_new_tokens": body.max_new_tokens}))
        waited = 0.0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    waited += 1.0
                    if await request.is_disconnected() or (waited % s.sse_reauth_s < 1.0 and not await anyio.to_thread.run_sync(check)):
                        return
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    yield sse_format(Item("end", {"reason": "finished"}))
                    return
                yield sse_format(item)
        finally:
            cancel.set()  # client left (or finished): stop generating and free the model slot
            gate.release(slot)

    return StreamingResponse(body_gen(), media_type="text/event-stream", headers=HEADERS)
