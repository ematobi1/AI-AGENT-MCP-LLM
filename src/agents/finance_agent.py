"""Finance agent (architecture doc §5.1, business domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Finance specialist agent. You analyze financial questions, model scenarios, and summarize figures for a business or personal context. State every assumption you make explicitly -- never present an assumed number as a confirmed fact. Flag anything that looks like a compliance, tax, or reporting risk rather than resolving it yourself.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "summary": "...", "key_figures": {"figure_name": 0.0}, "assumptions": ["..."], "flags": ["..."], "sources": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2048,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("finance_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="finance_agent", provider=resp.provider, response=resp,
                      action="analyze", status=result.get("status", "blocked"),
                      output=result)
    return result
