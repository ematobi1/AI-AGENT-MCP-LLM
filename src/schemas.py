"""
Structured output schemas for every agent in the roster (architecture doc §5).
Each agent's system prompt should be updated to describe its schema's fields
exactly — pydantic will then reject (not silently pass through) a malformed
response, which is a stricter version of the regex-based parse_json_response
fallback in llm_adapter.py.

pip install pydantic  (already in requirements.txt)
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


Status = Literal["complete", "blocked", "needs_approval"]
RiskTier = Literal["low", "medium", "high"]


class AgentOutputBase(BaseModel):
    """Every agent's response extends this. Keeps the contract from
    architecture doc §4 (status discipline) enforced at the code level,
    not just described in a prompt."""
    status: Status
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


# ──────────────────────────────── Orchestrator ────────────────────────────

class OrchestratorRoutingOutput(AgentOutputBase):
    """Orchestrator doesn't do domain work, so it never emits its own
    'status' — it's always routing. Default to 'complete' so validation
    doesn't require the model to produce a field it was never asked for."""
    status: Status = "complete"
    domain_tags: list[str] = Field(default_factory=list)
    next_agents: list[str] = Field(default_factory=list)
    risk_level: RiskTier
    reasoning: str


# ─────────────────────────── Business domain (9) ───────────────────────────

class ResearchOutput(AgentOutputBase):
    """Shared by the business Research agent and the cross-cutting
    Research/web agent — same schema, same job."""
    summary: str
    sources: list[str] = Field(default_factory=list)


class FinanceOutput(AgentOutputBase):
    summary: str
    key_figures: dict[str, float] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class LegalOutput(AgentOutputBase):
    status: Literal["needs_approval", "blocked"] = "needs_approval"  # never auto-completes
    flags: list[dict] = Field(default_factory=list)   # {clause, risk, explanation}
    redline_summary: str
    open_questions: list[str] = Field(default_factory=list)


class MarketingOutput(AgentOutputBase):
    draft_content: str
    channel: str                      # e.g. "email", "social", "landing_page"
    tone: str
    call_to_action: Optional[str] = None


class SalesOutput(AgentOutputBase):
    lead_summary: str
    outreach_draft: Optional[str] = None
    pipeline_notes: list[str] = Field(default_factory=list)
    requires_send_approval: bool = True


class OpsProjectOutput(AgentOutputBase):
    tasks: list[dict] = Field(default_factory=list)   # {title, owner, due_date, status}
    timeline_summary: str
    blockers: list[str] = Field(default_factory=list)


class HROutput(AgentOutputBase):
    document_type: str                 # e.g. "job_description", "policy_draft"
    draft_content: str
    compliance_notes: list[str] = Field(default_factory=list)


class CodingOutput(AgentOutputBase):
    code_diff: str
    files_changed: list[str] = Field(default_factory=list)
    tests_added: bool = False
    review_notes: list[str] = Field(default_factory=list)


class DataAnalyticsOutput(AgentOutputBase):
    analysis_summary: str
    key_metrics: dict[str, float] = Field(default_factory=dict)
    chart_spec: Optional[dict] = None  # handed to the visualization agent


# ─────────────────────────── Personal domain (7) ───────────────────────────

class SchedulingOutput(AgentOutputBase):
    proposed_events: list[dict] = Field(default_factory=list)  # {title, start, end}
    conflicts: list[str] = Field(default_factory=list)
    confirmation_needed: bool = True


class PersonalFinanceOutput(AgentOutputBase):
    budget_summary: str
    spending_flags: list[str] = Field(default_factory=list)
    subscriptions_found: list[dict] = Field(default_factory=list)


class HealthWellnessOutput(AgentOutputBase):
    general_guidance: str
    disclaimer: str = "Not medical advice. Consult a licensed professional for personal guidance."
    sources: list[str] = Field(default_factory=list)


class TravelOutput(AgentOutputBase):
    itinerary: list[dict] = Field(default_factory=list)  # {day, location, notes}
    booking_options: list[dict] = Field(default_factory=list)
    logistics_notes: list[str] = Field(default_factory=list)


class LearningTutorOutput(AgentOutputBase):
    explanation: str
    study_plan: Optional[list[str]] = None
    quiz_items: list[dict] = Field(default_factory=list)  # {question, answer}


class HomeAdminOutput(AgentOutputBase):
    reminders: list[dict] = Field(default_factory=list)   # {task, due_date}
    bills_summary: Optional[str] = None
    renewals_due: list[str] = Field(default_factory=list)


class CorrespondenceOutput(AgentOutputBase):
    draft_message: str
    tone: str
    recipient: Optional[str] = None
    send_ready: bool = False           # always False — sending needs explicit approval


# ───────────────────────── Cross-cutting agents (6) ─────────────────────────
# (Research/web is merged into ResearchOutput above — same schema, same job)

class WritingEditingOutput(AgentOutputBase):
    edited_content: str
    changes_summary: list[str] = Field(default_factory=list)
    style_notes: Optional[str] = None


class DocumentOutput(AgentOutputBase):
    file_path: str
    file_type: str                     # "docx" | "pdf" | "pptx" | "xlsx"
    summary: str


class DataVisualizationOutput(AgentOutputBase):
    chart_type: str                    # "bar" | "line" | "pie" | ...
    chart_spec: dict
    data_summary: str


class QAReviewOutput(AgentOutputBase):
    review_verdict: Literal["approved", "needs_revision", "rejected"]
    issues_found: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class MemoryContextOutput(AgentOutputBase):
    stored_facts: list[str] = Field(default_factory=list)
    retrieved_context: dict = Field(default_factory=dict)
    expiry: Optional[str] = None       # ISO date, if the memory is time-bound


class SecurityFindingItem(BaseModel):
    category: str
    description: str
    evidence: str


class SecurityAnalyzerOutput(AgentOutputBase):
    severity: Literal["informational", "caution", "block"]
    findings: list[SecurityFindingItem] = Field(default_factory=list)
    recommendation: str


# ──────────────────────────────── Registry ────────────────────────────────

AGENT_SCHEMA_REGISTRY: dict[str, type[AgentOutputBase]] = {
    "orchestrator": OrchestratorRoutingOutput,
    # business
    "research_agent": ResearchOutput,
    "finance_agent": FinanceOutput,
    "legal_agent": LegalOutput,
    "marketing_agent": MarketingOutput,
    "sales_agent": SalesOutput,
    "ops_project_agent": OpsProjectOutput,
    "hr_agent": HROutput,
    "coding_agent": CodingOutput,
    "data_analytics_agent": DataAnalyticsOutput,
    # personal
    "scheduling_agent": SchedulingOutput,
    "personal_finance_agent": PersonalFinanceOutput,
    "health_wellness_agent": HealthWellnessOutput,
    "travel_agent": TravelOutput,
    "learning_tutor_agent": LearningTutorOutput,
    "home_admin_agent": HomeAdminOutput,
    "correspondence_agent": CorrespondenceOutput,
    # cross-cutting
    "writing_editing_agent": WritingEditingOutput,
    "document_agent": DocumentOutput,
    "data_visualization_agent": DataVisualizationOutput,
    "qa_review_agent": QAReviewOutput,
    "memory_context_agent": MemoryContextOutput,
    "security_analyzer": SecurityAnalyzerOutput,
}


# ────────────────────────────── Risk tiers (Phase 5) ──────────────────────
# Per architecture doc §10 / roadmap Phase 5: medium and high risk agents
# get reviewed by security_analyzer BEFORE they're allowed to run at all,
# not just after. Low risk agents keep the existing post-execution-only
# review (added in earlier phases) -- reviewing every single low-risk
# action twice would double API cost for negligible safety benefit, since
# these agents can't take an irreversible or high-stakes action.
#
# "high" = can materially affect legal standing, money, or an external
#          party's inbox/systems if misused.
# "medium" = produces something a human might act on with real consequences
#            (code that gets merged, HR policy language) but isn't itself
#            the irreversible step.
# "low" = informational, draft-only, or purely internal-facing.
AGENT_RISK_TIER: dict[str, str] = {
    "research_agent": "low",
    "finance_agent": "medium",
    "legal_agent": "high",
    "marketing_agent": "low",
    "sales_agent": "high",              # drafts outreach to real external parties
    "ops_project_agent": "low",
    "hr_agent": "medium",
    "coding_agent": "medium",
    "data_analytics_agent": "low",
    "scheduling_agent": "low",
    "personal_finance_agent": "medium",
    "health_wellness_agent": "medium",  # not medical advice, but health-adjacent
    "travel_agent": "low",
    "learning_tutor_agent": "low",
    "home_admin_agent": "low",
    "correspondence_agent": "high",     # drafts messages to real people
    "writing_editing_agent": "low",
    "document_agent": "low",
    "data_visualization_agent": "low",
    "qa_review_agent": "low",
    "memory_context_agent": "low",
}


def validate_agent_output(agent_name: str, raw: dict) -> AgentOutputBase:
    """Call this instead of trusting parse_json_response's dict directly.
    Raises pydantic.ValidationError on a malformed response — catch that
    at the call site and route to a retry or a 'blocked' status, rather
    than silently passing bad data downstream."""
    schema = AGENT_SCHEMA_REGISTRY.get(agent_name)
    if schema is None:
        raise KeyError(f"No schema registered for agent '{agent_name}'")
    return schema.model_validate(raw)
