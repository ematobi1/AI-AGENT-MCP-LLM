"""
Shared task state ("the blackboard").
Every agent reads from and writes to this structure instead of passing
raw conversation history around. This is the single source of truth
for a task's progress, and what the security analyzer audits.
"""
from __future__ import annotations
from typing import Literal, Optional, TypedDict
from datetime import datetime, timezone
import uuid


Status = Literal["complete", "blocked", "needs_approval"]
Severity = Literal["informational", "caution", "block"]


class AgentLogEntry(TypedDict):
    agent: str
    provider: str          # which LLM/model actually served this call
    action: str
    status: Status
    output: dict
    timestamp: str


class SecurityFinding(TypedDict):
    category: str
    description: str
    evidence: str


class TaskState(TypedDict):
    task_id: str
    user_goal: str
    domain_tags: list[str]
    risk_level: Literal["low", "medium", "high"]
    context: dict            # relevant_facts, user_preferences, prior_decisions
    agent_log: list[AgentLogEntry]
    pending_approvals: list[dict]
    security_findings: list[SecurityFinding]
    final_output: Optional[str]


def new_task_state(user_goal: str, domain_tags: list[str], risk_level: str = "low") -> TaskState:
    return TaskState(
        task_id=str(uuid.uuid4()),
        user_goal=user_goal,
        domain_tags=domain_tags,
        risk_level=risk_level,  # type: ignore
        context={"relevant_facts": [], "user_preferences": {}, "prior_decisions": []},
        agent_log=[],
        pending_approvals=[],
        security_findings=[],
        final_output=None,
    )


def log_agent_action(state: TaskState, agent: str, provider: str, action: str,
                      status: Status, output: dict) -> None:
    state["agent_log"].append({
        "agent": agent,
        "provider": provider,
        "action": action,
        "status": status,
        "output": output,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
