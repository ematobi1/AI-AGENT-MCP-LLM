"""Writing & editing agent (architecture doc §5.1, cross-cutting)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Writing & Editing specialist agent. You edit or polish provided text for clarity, grammar, and tone, and explain what you changed and why. Preserve the author's voice and intent -- don't rewrite meaning, only improve expression, unless explicitly asked to change the substance.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "edited_content": "...", "changes_summary": ["..."], "style_notes": "..."}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=3072,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("writing_editing_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="writing_editing_agent", provider=resp.provider, response=resp,
                      action="edit", status=result.get("status", "blocked"),
                      output=result)
    return result
