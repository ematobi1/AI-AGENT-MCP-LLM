"""Personal finance agent (architecture doc §5.1, personal domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Personal Finance specialist agent. You summarize spending patterns, flag unusual or concerning spending, and identify recurring subscriptions from data given to you. You are not a licensed financial advisor -- avoid specific investment recommendations, and flag anything that looks like it needs a professional's input.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "budget_summary": "...", "spending_flags": ["..."], "subscriptions_found": [{"name": "...", "amount": 0.0, "frequency": "..."}]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=1536,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("personal_finance_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="personal_finance_agent", provider=resp.provider, response=resp,
                      action="analyze", status=result.get("status", "blocked"),
                      output=result)
    return result
