"""Home admin agent (architecture doc §5.1, personal domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Home Admin specialist agent. You organize reminders, summarize bills, and track renewals due from information given to you. Don't infer a due date or amount that wasn't provided or clearly implied -- leave it out rather than guessing.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "reminders": [{"task": "...", "due_date": "..."}], "bills_summary": "...", "renewals_due": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=1536,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("home_admin_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="home_admin_agent", provider=resp.provider,
                      action="organize", status=result.get("status", "blocked"),
                      output=result)
    return result
