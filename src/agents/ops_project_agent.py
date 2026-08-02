"""Operations / project management agent (architecture doc §5.1, business domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Operations & Project Management specialist agent. You break work into concrete tasks with owners and due dates, summarize timelines, and surface blockers. Do not invent owner names or dates that weren't given or reasonably inferable -- leave a field blank or flag it as unassigned rather than guessing.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "tasks": [{"title": "...", "owner": "...", "due_date": "...", "status": "..."}], "timeline_summary": "...", "blockers": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2048,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("ops_project_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="ops_project_agent", provider=resp.provider, response=resp,
                      action="plan", status=result.get("status", "blocked"),
                      output=result)
    return result
