"""Read-only description of the agent system: who the agents are, what each may do, and the contract each is held to."""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.deps import CurrentAuth

router = APIRouter(tags=["agents"])


@router.get("/agents")
def list_agents(request: Request, auth: CurrentAuth):
    rt = request.app.state.runtime
    engine = rt.controller()
    return {"workflow": rt.settings.workflow, "agents": engine.registry.describe(engine.policy)}
