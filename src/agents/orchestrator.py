"""Orchestrator agent (architecture doc §4)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Orchestrator for a multi-agent assistant system.
You do not do domain work yourself — you classify the request, decide which
specialist agent(s) are needed and in what order, and synthesize their
outputs into one coherent response.

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