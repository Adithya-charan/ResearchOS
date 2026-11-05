"""Request-scoped dependencies: database session, the authenticated user, CSRF enforcement, audit logging."""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.db import utcnow
from api.models import AuditLog, SessionRow, User
from api.security import token_hash

UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def get_db(request: Request):
    db = request.app.state.sessionmaker()
    try:
        yield db
    finally:
        db.close()


DB = Annotated[Session, Depends(get_db)]


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@dataclass
class Auth:
    user: User
    session: SessionRow


def current_auth(request: Request, db: DB) -> Auth:
    token = request.cookies.get(request.app.state.settings.cookie_name)
    if not token:
        raise HTTPException(401, "not authenticated")
    row = db.get(SessionRow, token_hash(token))
    if row is None or row.expires_at <= utcnow():
        if row is not None:
            db.delete(row)
            db.commit()
        raise HTTPException(401, "session expired or invalid")
    user = db.get(User, row.user_id)
    if user is None or not user.is_active:
        raise HTTPException(401, "not authenticated")
    if request.method in UNSAFE:  # CSRF: a cookie alone is never enough to change state
        sent = request.headers.get("x-csrf-token", "")
        if not sent or not secrets.compare_digest(sent, row.csrf_token):
            raise HTTPException(403, "missing or invalid CSRF token")
    return Auth(user, row)


CurrentAuth = Annotated[Auth, Depends(current_auth)]


def audit(db: Session, request: Request, action: str, user_id: str | None, target: str = "", **detail) -> None:
    db.add(AuditLog(user_id=user_id, action=action, target=target, ip=client_ip(request), detail=detail))
    db.commit()
