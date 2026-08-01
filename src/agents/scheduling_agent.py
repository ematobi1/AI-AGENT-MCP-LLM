"""Scheduling agent (architecture doc §5.1, personal domain). Never confirms a calendar write itself -- confirmation_needed stays true."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Scheduling specialist agent. You propose calendar events and flag conflicts -- you never actually write to a calendar yourself in this implementation, so confirmation_needed must stay true. Be explicit about timezone assumptions if the task doesn't state one.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "proposed_events": [{"title": "...", "start": "...", "end": "..."}], "conflicts": ["..."], "confirmation_needed": true}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=1536,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("scheduling_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="scheduling_agent", provider=resp.provider,
                      action="schedule", status=result.get("status", "blocked"),
                      output=result)
    return result
