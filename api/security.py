"""Password hashing (Argon2id), session tokens, password policy and a small sliding-window rate limiter."""
from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from argon2.low_level import Type

_hasher = PasswordHasher(type=Type.ID)
_DUMMY_HASH = _hasher.hash("researchos-dummy-password")  # verified against when the account is unknown (timing parity)
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s]{2,24}$")
_COMMON = ("password", "123456", "qwerty", "letmein", "iloveyou", "researchos")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored: str | None, password: str) -> bool:
    try:
        ok = _hasher.verify(stored or _DUMMY_HASH, password)
    except (VerificationError, InvalidHashError):
        return False
    return bool(ok) and stored is not None


def normalise_email(email: str) -> str:
    e = email.strip().lower()
    if not _EMAIL.match(e):
        raise ValueError("invalid email address")
    return e


def check_password(password: str, email: str, min_length: int) -> None:
    problems = []
    if len(password) < min_length:
        problems.append(f"at least {min_length} characters")
    if len(password) > 256:
        problems.append("at most 256 characters")
    if len(set(password)) < 5:
        problems.append("at least 5 different characters")
    if any(c in password.lower() for c in _COMMON):
        problems.append("must not contain a common password")
    local = email.split("@")[0].lower()
    if len(local) >= 4 and local in password.lower():
        problems.append("must not contain your email name")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"[\d\W_]", password)):
        problems.append("letters plus at least one digit or symbol")
    if problems:
        raise ValueError("password needs: " + "; ".join(problems))


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class RateLimiter:
    """In-process sliding window. Fine for one API process; use Redis-backed limiting when running several."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _trim(q: deque[float], window: float, now: float) -> None:
        while q and now - q[0] > window:
            q.popleft()

    def blocked(self, key: str, limit: int, window: float) -> bool:
        with self._lock:
            q = self._hits[key]
            self._trim(q, window, time.monotonic())
            return len(q) >= limit

    def hit(self, key: str, window: float) -> None:
        with self._lock:
            q = self._hits[key]
            now = time.monotonic()
            self._trim(q, window, now)
            q.append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)
