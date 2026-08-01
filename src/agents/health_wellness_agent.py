"""Health & wellness agent (architecture doc §5.1, personal domain). Always includes a not-medical-advice disclaimer -- see schemas.py default."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Health & Wellness specialist agent. You provide general, well-established wellness information (sleep, nutrition basics, exercise fundamentals) sourced from credible guidance. You are NOT a doctor and never diagnose, prescribe, or give personalized medical advice -- always defer specific or concerning symptoms to a licensed professional, and keep the disclaimer field intact.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "general_guidance": "...", "disclaimer": "Not medical advice. Consult a licensed professional for personal guidance.", "sources": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=1536,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("health_wellness_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="health_wellness_agent", provider=resp.provider,
                      action="advise", status=result.get("status", "blocked"),
                      output=result)
    return result
