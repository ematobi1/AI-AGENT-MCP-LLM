"""
FastAPI wrapper (roadmap Phase 8) -- lets the pipeline serve HTTP requests
instead of only running as `python -m src.graph` from a terminal.

Endpoints:
  POST /tasks              start a new task; returns either the final
                            result or a pending-approval payload
  GET  /tasks/{task_id}     check status of a task (completed -> reads
                            from agent_runs.db; still pending approval ->
                            reads from the in-memory PENDING_INTERRUPTS)
  POST /tasks/{task_id}/resume   approve or reject a pending approval
  GET  /health              liveness check, also reports kill switch state

IMPORTANT LIMITATION: PENDING_INTERRUPTS below is an in-memory dict. That
is fine for local development and a single-process deployment, but it does
NOT survive a process restart and does NOT work across multiple worker
processes (e.g. `uvicorn --workers 4`) since each worker has its own
memory. For a real multi-process production deployment, replace this with
a shared store (Redis, or a dedicated table in the same database
persistence.py already writes to) so any worker can see any task's
pending-approval state. The LangGraph checkpointer itself (InMemorySaver
in graph.py) has the identical limitation for the same reason -- swap it
for SqliteSaver/PostgresSaver at the same time.

Run:
    uvicorn src.api:app --reload
Then:
    curl -X POST http://localhost:8000/tasks -H "Content-Type: application/json" -d "{\"user_goal\": \"...\"}"
"""
from __future__ import annotations
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langgraph.types import Command

from . import config
from . import persistence
from .graph import build_graph
from .state import new_task_state

app = FastAPI(title="Multi-Agent System API", version="1.0.0")

# See module docstring's IMPORTANT LIMITATION note.
PENDING_INTERRUPTS: dict[str, dict] = {}

_graph_app = None  # built lazily on first request / at startup


def _get_graph_app():
    global _graph_app
    if _graph_app is None:
        _graph_app = build_graph()
    return _graph_app


@app.on_event("startup")
def on_startup():
    """Fails fast at process startup if required secrets are missing --
    same config.validate_startup_config() the CLI entry point uses."""
    config.validate_startup_config()
    persistence.init_db()


class StartTaskRequest(BaseModel):
    user_goal: str
    domain_tags: Optional[list[str]] = None


class ResumeTaskRequest(BaseModel):
    approved: bool


def _run_config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


def _invoke_and_handle_result(task_id: str, invoke_fn) -> dict:
    """Shared logic between starting and resuming a task -- both end up
    either interrupted again or finished, and both need the same
    persist/cleanup handling. invoke_fn is a zero-arg callable that
    performs the actual app.invoke(...) call."""
    result = invoke_fn()

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        PENDING_INTERRUPTS[task_id] = {"result_snapshot": result, "payload": payload}
        return {
            "task_id": task_id,
            "status": "needs_approval",
            "approval_request": payload,
        }

    PENDING_INTERRUPTS.pop(task_id, None)
    persistence.persist_task_run(result)
    total_cost = sum(e.get("cost_usd") or 0 for e in result["agent_log"])

    return {
        "task_id": task_id,
        "status": "blocked" if result.get("pending_approvals") else "complete",
        "final_output": result.get("final_output"),
        "total_cost_usd": round(total_cost, 6),
        "agent_log": result.get("agent_log", []),
    }


@app.post("/tasks")
def start_task(req: StartTaskRequest):
    if config.is_kill_switch_active():
        raise HTTPException(
            status_code=503,
            detail="System kill switch is active. No new tasks can be started until it's cleared.",
        )

    task_id = str(uuid.uuid4())
    graph_app = _get_graph_app()
    state = new_task_state(user_goal=req.user_goal, domain_tags=req.domain_tags or [])

    return _invoke_and_handle_result(
        task_id,
        lambda: graph_app.invoke(state, config=_run_config(task_id)),
    )


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    if task_id in PENDING_INTERRUPTS:
        return {
            "task_id": task_id,
            "status": "needs_approval",
            "approval_request": PENDING_INTERRUPTS[task_id]["payload"],
        }

    summary = persistence.get_task_summary(task_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"No task found with id {task_id}")

    return summary


@app.post("/tasks/{task_id}/resume")
def resume_task(task_id: str, req: ResumeTaskRequest):
    if task_id not in PENDING_INTERRUPTS:
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} has no pending approval to resume "
                   f"(already completed, doesn't exist, or this worker "
                   f"process doesn't have it in memory -- see module "
                   f"docstring's IMPORTANT LIMITATION note).",
        )

    graph_app = _get_graph_app()
    return _invoke_and_handle_result(
        task_id,
        lambda: graph_app.invoke(Command(resume=req.approved), config=_run_config(task_id)),
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "kill_switch_active": config.is_kill_switch_active(),
        "disabled_agents": sorted(config.get_disabled_agents()),
    }
