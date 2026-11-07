"""Model lifecycle state machine (pure logic; persistence lives in the API layer).

States: INITIALIZED -> TRAINING -> TRAINED -> VALIDATED -> DEPLOYED, with ARCHIVED (terminal) and FAILED.

Rules that matter more than the arrows:
* A model is **never TRAINED merely because it was initialised**: TRAINING -> TRAINED needs evidence of a completed run with
  at least one applied optimizer step and a finite final loss.
* VALIDATED needs a *successful* validation report: finite validation loss, checkpoint integrity verified, random-init lineage
  verified, no train/test leakage found, and a better validation loss than a random-initialised baseline of the same
  architecture. Nothing else grants VALIDATED.
* DEPLOYED (eligible for production agent reasoning) needs VALIDATED plus an explicit human authorisation recorded with
  the transition. Experimental use of a non-deployed model is a separate, explicit, per-request choice made elsewhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class ModelState(str, Enum):
    INITIALIZED = "INITIALIZED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    VALIDATED = "VALIDATED"
    DEPLOYED = "DEPLOYED"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


S = ModelState

ALLOWED: dict[ModelState, frozenset[ModelState]] = {
    S.INITIALIZED: frozenset({S.TRAINING, S.ARCHIVED}),
    S.TRAINING: frozenset({S.TRAINED, S.FAILED, S.INITIALIZED}),  # INITIALIZED only if no step was ever applied
    S.TRAINED: frozenset({S.VALIDATED, S.TRAINING, S.ARCHIVED}),
    S.VALIDATED: frozenset({S.DEPLOYED, S.TRAINING, S.TRAINED, S.ARCHIVED}),  # TRAINED = validation withdrawn
    S.DEPLOYED: frozenset({S.VALIDATED, S.ARCHIVED}),
    S.FAILED: frozenset({S.TRAINING, S.ARCHIVED}),
    S.ARCHIVED: frozenset(),
}


class LifecycleError(ValueError):
    """The requested transition is not allowed (unknown arrow or missing evidence)."""


@dataclass
class Evidence:
    """Facts supplied by the caller, derived from real run/evaluation artifacts - never defaults that grant a state."""
    steps_completed: int = 0
    completion_state: str | None = None  # from run.json
    final_loss: float | None = None
    checkpoint_verified: bool = False
    lineage_verified: bool = False
    val_loss: float | None = None
    baseline_val_loss: float | None = None  # random-initialised model, same architecture, same data
    leakage_checked: bool = False
    leakage_found: bool = True  # pessimistic default
    authorized_by: str | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def check_transition(current: ModelState | str, target: ModelState | str, ev: Evidence | None = None) -> list[str]:
    """Return the reasons a transition is refused (empty list = allowed)."""
    cur, tgt = ModelState(current), ModelState(target)
    ev = ev or Evidence()
    if tgt not in ALLOWED[cur]:
        return [f"transition {cur.value} -> {tgt.value} is not allowed"]
    problems: list[str] = []
    if cur == S.TRAINING and tgt == S.INITIALIZED and ev.steps_completed > 0:
        problems.append("a model that has applied optimizer steps cannot return to INITIALIZED")
    if tgt == S.TRAINED and cur == S.TRAINING:
        if ev.steps_completed < 1:
            problems.append("TRAINED requires at least one completed optimizer step")
        if ev.completion_state != "completed":
            problems.append(f"TRAINED requires a completed run (completion_state={ev.completion_state!r})")
        if not _finite(ev.final_loss):
            problems.append("TRAINED requires a finite final training loss")
        if not ev.lineage_verified:
            problems.append("TRAINED requires verified random-initialisation lineage")
    if tgt == S.VALIDATED:
        if not ev.checkpoint_verified:
            problems.append("VALIDATED requires a verified checkpoint (hash and tokenizer compatibility)")
        if not ev.lineage_verified:
            problems.append("VALIDATED requires verified random-initialisation lineage")
        if not _finite(ev.val_loss):
            problems.append("VALIDATED requires a finite validation loss from a successful evaluation")
        if not _finite(ev.baseline_val_loss):
            problems.append("VALIDATED requires a random-initialised baseline evaluation")
        elif _finite(ev.val_loss) and not ev.val_loss < ev.baseline_val_loss:
            problems.append(
                f"validation loss {ev.val_loss:.4f} does not beat the random-initialised baseline {ev.baseline_val_loss:.4f}"
            )
        if not ev.leakage_checked:
            problems.append("VALIDATED requires a train/validation/test leakage check")
        elif ev.leakage_found:
            problems.append("train/validation/test leakage was found")
    if tgt == S.DEPLOYED and not ev.authorized_by:
        problems.append("DEPLOYED requires an explicit authorisation (authorized_by)")
    return problems


def transition(current: ModelState | str, target: ModelState | str, ev: Evidence | None = None) -> ModelState:
    problems = check_transition(current, target, ev)
    if problems:
        raise LifecycleError("; ".join(problems))
    return ModelState(target)


def production_eligible(state: ModelState | str) -> bool:
    """Only a DEPLOYED model may drive production agent reasoning."""
    return ModelState(state) == S.DEPLOYED


def experimental_eligible(state: ModelState | str) -> bool:
    """Models that hold trained weights may be used *experimentally* when a caller explicitly asks."""
    return ModelState(state) in (S.TRAINED, S.VALIDATED, S.DEPLOYED)
