"""Budgeted access to the locally trained language model.

The model is an *optional, untrusted helper*: it can add a labelled draft note, never evidence, never control flow.
Every call reserves tokens from the run's budget first; when the budget is spent (or no model is attached) the
call returns None and the agent carries on deterministically.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Protocol

from agent_engine.contracts import ModelNote
from agent_engine.state.store import Store

MAX_PROMPT_CHARS = 500


class TextBackend(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, int]:
        """Return (text, number_of_new_tokens)."""
        ...


class LocalBackend:
    """Adapter over model.inference.ResearchOSModel (the model RESEARCHOS trained itself)."""

    def __init__(self, runtime, temperature: float = 0.7, top_k: int = 40):
        if runtime.tokenizer is None:
            raise ValueError("runtime needs a tokenizer attached")
        self.runtime, self.temperature, self.top_k = runtime, temperature, top_k

    def count_tokens(self, text: str) -> int:
        return 1 + len(self.runtime.tokenizer.encode(text))  # +1 for the <DOC> start token

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, int]:
        res = self.runtime.generate_text(
            prompt, max_new_tokens, temperature=self.temperature, top_k=self.top_k, seed=seed
        )
        return res.text or "", len(res.token_ids)


class ModelGateway:
    """Serialises access to one backend (one generation at a time keeps GPU memory bounded)."""

    def __init__(self, backend: TextBackend | None, store: Store, policy=None):
        self.backend, self.store = backend, store
        self.policy = policy  # agent_engine.model_provider.RolePolicy | None; None = unrestricted (original behaviour)
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.backend is not None

    def handle(self, run_id: str, token_limit: int, emit: Callable[[str, dict], None], agent_type: str | None = None) -> ModelHandle:
        return ModelHandle(self, run_id, token_limit, emit, agent_type)

    def _generate_locked(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, int]:
        with self._lock:
            return self.backend.generate(prompt, max_new_tokens, seed)  # type: ignore[union-attr]


class ModelHandle:
    """A task's view of the gateway, bound to one run's token budget."""

    def __init__(self, gateway: ModelGateway, run_id: str, token_limit: int, emit: Callable[[str, dict], None],
                 agent_type: str | None = None):
        self._gw, self._run_id, self._limit, self._emit = gateway, run_id, token_limit, emit
        self._agent_type = agent_type

    async def note(self, prompt: str, max_new_tokens: int = 48, seed: int = 0) -> ModelNote | None:
        gw = self._gw
        if gw.backend is None:
            return None
        if gw.policy is not None and not gw.policy.allows(self._agent_type):
            self._emit("model_call_skipped", {"reason": "role_not_enabled", "agent_type": self._agent_type})
            return None
        prompt = prompt[:MAX_PROMPT_CHARS]
        try:
            reserve = gw.backend.count_tokens(prompt) + max_new_tokens
        except Exception as e:  # tokenizer problem: treat the model as unavailable
            self._emit("model_call_failed", {"error": str(e)[:200]})
            return None
        if not gw.store.reserve_model_tokens(self._run_id, reserve, self._limit):
            self._emit("model_call_skipped", {"reason": "token_budget_exhausted", "requested": reserve})
            return None
        try:
            text, new_tokens = await asyncio.to_thread(gw._generate_locked, prompt, max_new_tokens, seed)
        except Exception as e:
            gw.store.add_model_tokens(self._run_id, -reserve)  # refund: nothing was generated
            self._emit("model_call_failed", {"error": str(e)[:200]})
            return None
        used = reserve - max_new_tokens + new_tokens  # prompt tokens + tokens actually generated
        gw.store.add_model_tokens(self._run_id, used - reserve)
        self._emit("model_call", {"prompt_chars": len(prompt), "tokens_used": used})
        from agent_engine.model_provider import guard_model_text  # local import: model_provider imports this module

        _, flags = guard_model_text(text)
        if flags:  # the model cannot read the corpus, so citation-like text is invention: drop the whole note
            self._emit("model_note_rejected", {"reason": "citation_like_text", "flags": flags[:4]})
            return None
        return ModelNote(text=text.strip(), tokens_used=used)
