"""Server-side, per-user rate limits for the expensive or abusable endpoints (P10).

Limits are counted per signed-in user, never per client claim, in the in-process sliding window of `RateLimiter`. With several API
processes each process counts separately, so the effective ceiling is limit x processes; put a shared limiter at the proxy for a hard
global cap (docs/SECURITY.md). Every limit is configurable: RESEARCHOS_RATE_<NAME>="<count>/<window seconds>", or "off" to disable.
Defaults are deliberately generous for one researcher working normally and stop scripted abuse."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from api.config import RATE_DEFAULTS, parse_rate, rates_from_env
from api.deps import Auth, current_auth


DEFAULTS = RATE_DEFAULTS
parse_policy = parse_rate
policies_from_env = rates_from_env


def rate_limit(name: str):
    """Dependency factory: `dependencies=[Depends(rate_limit("upload"))]`. Authentication runs first (a 401 never consumes budget)."""
    if name not in DEFAULTS:
        raise KeyError(name)

    def dep(request: Request, auth: Auth = Depends(current_auth)) -> None:
        policy = request.app.state.settings.rate_limits.get(name)
        if policy is None:
            return
        limit, window = policy
        limiter, key = request.app.state.limiter, f"rl:{name}:{auth.user.id}"
        if limiter.blocked(key, limit, window):
            raise HTTPException(429, f"too many {name.replace('_', ' ')} requests; limit is {limit} per {window} s",
                                headers={"Retry-After": str(window)})
        limiter.hit(key, window)
    return dep
