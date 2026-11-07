"""Provider selection for the custom Transformer, with capability state, staged acceptance and an output guard.

There are exactly two providers: the **deterministic** provider (no language model; every agent runs on deterministic
software only - the default and the fallback) and the **custom Transformer** provider (the model RESEARCHOS trained itself).
No hosted AI API exists anywhere in this module and none is ever used as a fallback.

Rules enforced here (tests: tests/test_p8_provider.py):
* Production use needs a DEPLOYED model (lifecycle: validated against a random baseline, then explicitly authorised).
* Experimental use needs an explicit opt-in per resolution and a model that has trained weights; an untrained model needs a
  second explicit opt-in (`allow_untrained`) and exists for pipeline tests only.
* A model that is not eligible raises `NoValidatedModelError`; the caller may choose `on_unavailable="deterministic"` to degrade
  to the deterministic provider, which is recorded in the returned status. Nothing silently switches.
* Model text is always an untrusted `ModelNote`. `guard_model_text` drops any note that looks like a citation, passage id or
  DOI: the model may never mint evidence, and verification stays with the deterministic evidence layer.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Literal

from agent_engine.model_gateway import LocalBackend
from model.lifecycle import ModelState, experimental_eligible, production_eligible

ProviderName = Literal["deterministic", "custom_transformer"]
Mode = Literal["production", "experimental"]

AGENT_ROLES = (
    "supervisor", "discovery", "literature", "gap", "hypothesis", "experiment_design",
    "reviewer", "verifier", "synthesis",
)
# `experiment` (sandbox execution) is deliberately absent: running code is never delegated to model text.

ROLE_PROMPTS: dict[str, str] = {
    "supervisor": "Title: {q}\n",
    "discovery": "Title: Literature on {q}\n",
    "literature": "Title: Methods and limitations: {q}\n",
    "gap": "Title: Open questions about {q}\n",
    "hypothesis": "Title: Hypothesis on {q}\n",
    "experiment_design": "Title: Experiment design for {q}\n",
    "reviewer": "Title: Review of {q}\n",
    "verifier": "Title: Verification notes on {q}\n",
    "synthesis": "Title: Summary of {q}\n",
}


class NoValidatedModelError(RuntimeError):
    """No model is eligible for the requested use. The message says why and what would make it eligible."""


# --------------------------------------------------------------------------- output guard
_CITATION_PATTERNS = [
    re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]"),                      # [1] [1, 2] [3-5]
    re.compile(r"\[[^\]]{0,80}\b(?:doc|passage|page|p\.|pp\.|ref|source)\b[^\]]{0,80}\]", re.I),  # [doc-1, page 3]
    re.compile(r"\([A-Z][A-Za-z\-']+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-']+|\s+et\s+al\.?)?,?\s+(?:19|20)\d{2}[a-z]?\)"),  # (Doe et al., 2019)
    re.compile(r"\b[A-Za-z0-9_\-]+:p\d{3}\b"),                                  # RESEARCHOS passage ids
    re.compile(r"\b10\.\d{4,9}/\S+"),                                           # DOI
    re.compile(r"\bdoi\s*:", re.I),
    re.compile(r"\barxiv\s*:?\s*\d{4}\.\d{4,5}", re.I),
    re.compile(r"https?://", re.I),
]


def guard_model_text(text: str) -> tuple[str, list[str]]:
    """Return (text, flags). Non-empty flags mean the text must not be used: it contains citation-like or source-like
    strings, which a language model cannot legitimately produce (it cannot read the corpus)."""
    flags = [f"citation_like:{p.pattern[:24]}" for p in _CITATION_PATTERNS if p.search(text)]
    return text, flags


# --------------------------------------------------------------------------- structured-output validator (Stage 2)
STATUS_VALUES = ("supported", "contradicted", "insufficient evidence", "unverified")
_STRUCT_RE = re.compile(r"^Claim (?P<id>C\d+): (?P<claim>[^\n]{3,300})\nStatus: (?P<status>[a-z ]+?)\.\s*$", re.M)


def parse_structured_note(text: str) -> dict | None:
    """Parse the claim/status block format. Returns None (rejected) for anything malformed, with an unknown status, or
    containing citation-like text. Never raises."""
    try:
        if guard_model_text(text)[1]:
            return None
        m = _STRUCT_RE.search(text)
        if not m or m["status"] not in STATUS_VALUES:
            return None
        return {"claim_id": m["id"], "claim": m["claim"].strip(), "status": m["status"]}
    except Exception:  # defensive: parsing model text must never take an agent down
        return None


# --------------------------------------------------------------------------- backend
class CustomTransformerBackend(LocalBackend):
    """`TextBackend` over the from-scratch Transformer, extended with streaming, cancellation, limits and metadata.

    Satisfies the existing `TextBackend` protocol unchanged (count_tokens / generate), so the agent engine needs no change."""

    def __init__(self, runtime, *, model_id: str = "", state: ModelState | str = ModelState.INITIALIZED, mode: Mode = "experimental",
                 temperature: float = 0.7, top_k: int = 40, max_new_tokens_limit: int = 128, max_prompt_tokens_limit: int | None = None):
        super().__init__(runtime, temperature=temperature, top_k=top_k)
        self.model_id, self.state, self.mode = model_id, ModelState(state), mode
        self.max_new_tokens_limit = max_new_tokens_limit
        self.max_prompt_tokens_limit = max_prompt_tokens_limit or (runtime.model.cfg.max_seq_len - 1 if runtime.model else 255)
        self._cancel = threading.Event()

    # -- limits
    def _clamp(self, prompt: str, max_new_tokens: int) -> int:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.count_tokens(prompt) > self.max_prompt_tokens_limit:
            raise ValueError(f"prompt exceeds the {self.max_prompt_tokens_limit}-token limit")
        return min(max_new_tokens, self.max_new_tokens_limit)

    # -- TextBackend protocol
    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, int]:
        n = self._clamp(prompt, max_new_tokens)
        self._cancel.clear()
        res = self.runtime.generate_text(prompt, n, temperature=self.temperature, top_k=self.top_k, seed=seed,
                                         cancel_event=self._cancel)
        return res.text or "", len(res.token_ids)

    # -- extras
    def stream(self, prompt: str, max_new_tokens: int, seed: int) -> Iterator[str]:
        """Yield text pieces as tokens arrive. Decoding is incremental on bytes, so a multi-byte character split across
        tokens is emitted once complete."""
        from model.tokenizer.bpe import DOC_ID

        n = self._clamp(prompt, max_new_tokens)
        self._cancel.clear()
        tok = self.runtime.tokenizer
        ids = [DOC_ID] + tok.encode(prompt)
        pending = b""
        for t in self.runtime.stream(ids, n, temperature=self.temperature, top_k=self.top_k, seed=seed, cancel_event=self._cancel):
            pending += tok.token_bytes(t)
            try:
                piece, pending = pending.decode("utf-8"), b""
            except UnicodeDecodeError as e:
                piece, pending = pending[:e.start].decode("utf-8"), pending[e.start:]
                if len(pending) > 4:  # genuinely invalid bytes, not a split character
                    piece, pending = piece + pending.decode("utf-8", errors="replace"), b""
            if piece:
                yield piece
        if pending:
            yield pending.decode("utf-8", errors="replace")

    def cancel(self) -> None:
        """Ask the in-flight generation (if any) to stop at the next token."""
        self._cancel.set()

    def metadata(self) -> dict:
        rt = self.runtime
        meta = rt.meta or {}
        cfg = rt.model.cfg
        return {
            "provider": "custom_transformer",
            "model_id": self.model_id,
            "lifecycle_state": self.state.value,
            "mode": self.mode,
            "parameters": rt.model.num_parameters(),
            "context_length": cfg.max_seq_len,
            "vocab_size": cfg.vocab_size,
            "model_sha256": meta.get("model_sha256"),
            "checkpoint_step": meta.get("step"),
            "tokenizer_fingerprint": rt.tokenizer.fingerprint if rt.tokenizer else None,
            "device": str(rt.device),
            "limits": {"max_new_tokens": self.max_new_tokens_limit, "max_prompt_tokens": self.max_prompt_tokens_limit},
            "note": "experimental small language model trained from random initialisation; output is untrusted",
        }


# --------------------------------------------------------------------------- selection
@dataclass
class ProviderSelection:
    provider: ProviderName = "deterministic"
    model_id: str | None = None
    mode: Mode = "experimental"
    allow_experimental: bool = False  # the explicit opt-in for non-DEPLOYED models
    allow_untrained: bool = False  # second opt-in: a model with no completed training (pipeline tests only)
    on_unavailable: Literal["error", "deterministic"] = "error"


@dataclass
class ModelRecordView:
    """What provider selection needs to know about a registry entry (decoupled from the database)."""
    model_id: str
    state: ModelState
    stage_reached: int = 0  # measured integration stage (see evaluate_stage); 0 = never measured


@dataclass
class ProviderStatus:
    provider: ProviderName
    capability: str  # none | untrained | trained_unvalidated | validated | deployed
    mode: str | None
    model_id: str | None
    lifecycle_state: str | None
    stage_reached: int
    fallback_reason: str | None = None
    explicit_experimental: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_CAPABILITY = {
    ModelState.INITIALIZED: "untrained", ModelState.TRAINING: "untrained", ModelState.TRAINED: "trained_unvalidated",
    ModelState.VALIDATED: "validated", ModelState.DEPLOYED: "deployed", ModelState.FAILED: "none", ModelState.ARCHIVED: "none",
}


def capability_of(state: ModelState | str | None) -> str:
    return "none" if state is None else _CAPABILITY[ModelState(state)]


def deterministic_status(reason: str | None = None) -> ProviderStatus:
    return ProviderStatus("deterministic", "none", None, None, None, 0, fallback_reason=reason,
                          notes=["agents run on deterministic software only"])


def resolve_provider(
    sel: ProviderSelection,
    lookup: Callable[[str], ModelRecordView | None],
    load: Callable[[str], CustomTransformerBackend],
) -> tuple[CustomTransformerBackend | None, ProviderStatus]:
    """Decide which backend an engine gets. `lookup(model_id)` reads the registry; `load(model_id)` builds the backend (only
    called after eligibility passed). Returns `(None, status)` for the deterministic provider."""
    if sel.provider == "deterministic":
        return None, deterministic_status()
    if sel.provider != "custom_transformer":
        raise ValueError(f"unknown provider {sel.provider!r}")

    def refuse(msg: str) -> tuple[None, ProviderStatus]:
        if sel.on_unavailable == "deterministic":
            return None, deterministic_status(reason=msg)
        raise NoValidatedModelError(msg)

    if not sel.model_id:
        return refuse("custom_transformer was selected without a model_id; choose a model explicitly (none is picked for you)")
    rec = lookup(sel.model_id)
    if rec is None:
        return refuse(f"model {sel.model_id!r} is not in the registry")
    state = ModelState(rec.state)
    if sel.mode == "production":
        if not production_eligible(state):
            return refuse(
                f"model {sel.model_id!r} is {state.value}; production agent reasoning needs a DEPLOYED model "
                "(train, validate against the random baseline, then deploy with explicit authorisation)"
            )
        if rec.stage_reached < 3:
            return refuse(
                f"model {sel.model_id!r} has measured integration stage {rec.stage_reached}; production needs stage >= 3 "
                "(structured-task performance compared with deterministic baselines)"
            )
    else:
        if not sel.allow_experimental:
            return refuse(
                f"model {sel.model_id!r} ({state.value}) can only be used experimentally and allow_experimental was not set"
            )
        trained = experimental_eligible(state)
        if not trained and not (sel.allow_untrained and state in (ModelState.INITIALIZED, ModelState.TRAINING)):
            return refuse(
                f"model {sel.model_id!r} is {state.value}: it has no completed training. "
                + ("Pass allow_untrained for pipeline tests only." if state in (ModelState.INITIALIZED, ModelState.TRAINING)
                   else "It cannot be used.")
            )
    try:
        backend = load(sel.model_id)
    except Exception as e:  # missing/corrupt checkpoint, tokenizer mismatch, out of memory: explicit, never silent
        return refuse(f"model {sel.model_id!r} could not be loaded: {type(e).__name__}: {str(e)[:200]}")
    backend.model_id, backend.state, backend.mode = sel.model_id, state, sel.mode
    status = ProviderStatus(
        "custom_transformer", capability_of(state), sel.mode, sel.model_id, state.value, rec.stage_reached,
        explicit_experimental=sel.mode == "experimental",
        notes=(["explicit experimental use: output is untrusted and not evidence"] if sel.mode == "experimental" else []),
    )
    if state in (ModelState.INITIALIZED, ModelState.TRAINING):
        status.notes.append("model has no completed training: output is noise; pipeline testing only")
    return backend, status


# --------------------------------------------------------------------------- staged acceptance
STRUCTURED_PROMPTS = [f"Claim C{i}: conductivity increased with temperature.\nStatus:" for i in range(1, 9)]
STAGE_DESCRIPTIONS = {
    1: "model loads, provider interface works, generation is reproducible",
    2: "model follows simple structured prompts; output parsing and validation work; invalid output is rejected",
    3: "trained on permitted instruction/research data; structured-task performance evaluated against deterministic baselines",
    4: "experimental agent use enabled only after the acceptance criteria are met",
    5: "responsibilities expanded gradually from measured results",
}
STRUCTURED_PASS_THRESHOLD = 0.8
_BAD_SAMPLES = ["", "Status: supported.", "Claim C1: x\nStatus: banana.", "Claim C1: see [1]\nStatus: supported.", "Claim C1: " + "x" * 400 + "\nStatus: supported.",
                "Claim C1: ok claim\nStatus: supported (Doe et al., 2019)."]


def evaluate_stage(backend: CustomTransformerBackend, *, instruction_eval: dict | None = None, seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)) -> dict:
    """Measure which integration stages the model has actually reached. Every field comes from running the backend now (stages
    1-2) or from a recorded evaluation artifact (stage 3); nothing is assumed.

    `instruction_eval` is the artifact a Stage-3 run would produce:
    `{"trained_on_permitted_data": bool, "dataset_licenses": [...], "structured_score": float, "deterministic_baseline_score": float}`.
    """
    checks: dict[str, dict] = {}

    def check(name: str, fn: Callable[[], object]) -> bool:
        try:
            ok = bool(fn())
            checks[name] = {"ok": ok}
        except Exception as e:
            ok = False
            checks[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        return ok

    s1 = [
        check("loads", lambda: backend.runtime.model is not None and backend.runtime.tokenizer is not None),
        check("metadata", lambda: backend.metadata()["parameters"] > 0),
        check("token_counts", lambda: backend.count_tokens("hello") >= 2),
        check("reproducible_generation", lambda: backend.generate("Title: test\n", 8, 123) == backend.generate("Title: test\n", 8, 123)),
    ]
    stage = 1 if all(s1) else 0

    outputs = []
    if stage >= 1:
        for s in seeds:
            try:
                outputs.append(backend.generate(STRUCTURED_PROMPTS[s % len(STRUCTURED_PROMPTS)], 24, s)[0])
            except Exception:
                outputs.append("")
    parsed = [parse_structured_note(STRUCTURED_PROMPTS[i % len(STRUCTURED_PROMPTS)] + o) for i, o in enumerate(outputs)]
    pass_rate = (sum(p is not None for p in parsed) / len(parsed)) if parsed else 0.0
    checks["structured_pass_rate"] = {"value": pass_rate, "threshold": STRUCTURED_PASS_THRESHOLD, "n": len(parsed),
                                      "ok": pass_rate >= STRUCTURED_PASS_THRESHOLD}
    check("invalid_output_rejected", lambda: all(parse_structured_note(b) is None for b in _BAD_SAMPLES))
    if stage >= 1 and checks["structured_pass_rate"]["ok"] and checks["invalid_output_rejected"]["ok"]:
        stage = 2

    s3 = bool(instruction_eval) and bool(instruction_eval.get("trained_on_permitted_data")) and bool(instruction_eval.get("dataset_licenses")) \
        and isinstance(instruction_eval.get("structured_score"), (int, float)) \
        and instruction_eval["structured_score"] > instruction_eval.get("deterministic_baseline_score", float("inf"))
    checks["instruction_training_and_baseline_comparison"] = {
        "ok": s3, "detail": "recorded instruction-tuning evaluation beating the deterministic baseline" if s3
        else "no recorded instruction-tuning run that beats the deterministic baseline"}
    if stage >= 2 and s3:
        stage = 3
    # Stages 4 and 5 are decisions, not measurements: 4 needs stage 3 AND a VALIDATED+ model AND an explicit switch; 5 needs
    # per-role measured results, which this project has not produced.
    return {
        "stage_reached": stage,
        "stage_description": STAGE_DESCRIPTIONS.get(stage, "model does not meet stage 1"),
        "next_stage": min(stage + 1, 5), "next_stage_requires": STAGE_DESCRIPTIONS.get(min(stage + 1, 5)),
        "checks": checks,
        "stage_4_5": "not enabled: stage 4 needs stage 3 plus a VALIDATED model and an explicit switch; stage 5 needs per-role measurements",
    }


# --------------------------------------------------------------------------- role policy
@dataclass
class RolePolicy:
    """Which agent roles may attach an (untrusted) model note. `enabled` roles only; everything else stays deterministic."""
    enabled: frozenset[str] = frozenset()
    auto_notes: bool = False  # engine-level notes for roles whose own code has no model hook
    max_new_tokens: int = 40

    def allows(self, role: str | None) -> bool:
        return role in self.enabled

    @classmethod
    def for_status(cls, status: ProviderStatus, roles: tuple[str, ...] = AGENT_ROLES) -> RolePolicy:
        """Experimental custom-model use enables draft notes for the listed roles; the deterministic provider enables none."""
        if status.provider != "custom_transformer":
            return cls()
        return cls(enabled=frozenset(roles), auto_notes=True)
