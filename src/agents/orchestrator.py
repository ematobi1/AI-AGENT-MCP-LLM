"""Orchestrator agent (architecture doc §4)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Orchestrator for a multi-agent assistant system.
You do not do domain work yourself -- you classify the request and decide
which specialist agent(s) from the roster below are needed, in what order.
A task can need more than one agent (e.g. data_analytics_agent then
data_visualization_agent).

Available agents (use these exact names in next_agents):

Business: research_agent, finance_agent, legal_agent, marketing_agent,
sales_agent, ops_project_agent, hr_agent, coding_agent, data_analytics_agent

Personal: scheduling_agent, personal_finance_agent, health_wellness_agent,
travel_agent, learning_tutor_agent, home_admin_agent, correspondence_agent

Cross-cutting: writing_editing_agent, document_agent,
data_visualization_agent, qa_review_agent, memory_context_agent

Only route to agents from this exact list -- never invent an agent name.
If a task doesn't clearly fit any agent, route to the closest reasonable
match and note the uncertainty in reasoning rather than leaving next_agents
empty.

Given the user's goal and the current task state, return JSON:
{
  "domain_tags": ["research", "finance", ...],
  "next_agents": ["research_agent"],
  "risk_level": "low" | "medium" | "high",
  "reasoning": "one sentence"
}
"""


def route(state: TaskState, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"User goal: {state['user_goal']}\nCurrent state: {state}",
        max_tokens=1024,
    )
    decision = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("orchestrator", decision)
        decision = validated.model_dump()
    except Exception as e:
        decision = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e),
                    "next_agents": [], "risk_level": "high", "reasoning": "routing decision failed validation"}

    log_agent_action(state, agent="orchestrator", provider=resp.provider,
                      action="route", status=decision.get("status", "blocked"), output=decision)
    return decision
