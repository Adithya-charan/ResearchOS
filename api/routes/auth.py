from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select

from api.deps import DB, CurrentAuth, audit, client_ip
from api.db import utcnow
from api.models import SessionRow, User
from api.schemas import LoginIn, RegisterIn, SessionOut, UserOut
from api.security import (check_password, hash_password, new_token, normalise_email, token_hash, verify_password)

router = APIRouter(prefix="/auth", tags=["auth"])


def _out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, display_name=user.display_name)


def _open_session(request: Request, response: Response, db, user: User) -> SessionOut:
    s = request.app.state.settings
    token = new_token()
    row = SessionRow(id=token_hash(token), user_id=user.id, csrf_token=new_token(),
                     expires_at=utcnow() + timedelta(seconds=s.session_ttl_s),
                     user_agent=request.headers.get("user-agent", "")[:200])
    db.add(row)
    db.commit()
    response.set_cookie(s.cookie_name, token, max_age=s.session_ttl_s, httponly=True, samesite="lax",
                        secure=s.cookie_secure, path="/")
    return SessionOut(user=_out(user), csrf_token=row.csrf_token, expires_at=row.expires_at)


@router.post("/register", status_code=201, response_model=SessionOut)
def register(body: RegisterIn, request: Request, response: Response, db: DB):
    s, limiter, ip = request.app.state.settings, request.app.state.limiter, client_ip(request)
    if limiter.blocked(f"reg:{ip}", s.register_attempts, s.register_window_s):
        raise HTTPException(429, "too many registrations from this address; try later")
    limiter.hit(f"reg:{ip}", s.register_window_s)
    try:
        email = normalise_email(body.email)
        check_password(body.password, email, s.min_password_length)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "an account with this email already exists")
    user = User(email=email, password_hash=hash_password(body.password), display_name=body.display_name)
    db.add(user)
    db.commit()
    audit(db, request, "register", user.id)
    return _open_session(request, response, db, user)


@router.post("/login", response_model=SessionOut)
def login(body: LoginIn, request: Request, response: Response, db: DB):
    s, limiter, ip = request.app.state.settings, request.app.state.limiter, client_ip(request)
    email = body.email.strip().lower()
    key = f"login:{ip}:{email}"
    if limiter.blocked(key, s.login_attempts, s.login_window_s):
        audit(db, request, "login_blocked", None, detail_email=email)
        raise HTTPException(429, "too many failed attempts; try again later")
    user = db.scalar(select(User).where(User.email == email))
    ok = verify_password(user.password_hash if user else None, body.password)  # always hashes: no timing oracle for unknown emails
    if not (ok and user and user.is_active):
        limiter.hit(key, s.login_window_s)
        audit(db, request, "login_failed", user.id if user else None)
        raise HTTPException(401, "invalid email or password")
    limiter.reset(key)
    audit(db, request, "login", user.id)
    return _open_session(request, response, db, user)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, auth: CurrentAuth, db: DB):
    db.delete(auth.session)
    db.commit()
    audit(db, request, "logout", auth.user.id)
    response.delete_cookie(request.app.state.settings.cookie_name, path="/")


@router.get("/me", response_model=SessionOut)
def me(auth: CurrentAuth):
    return SessionOut(user=_out(auth.user), csrf_token=auth.session.csrf_token, expires_at=auth.session.expires_at)
