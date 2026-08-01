"""Sales agent (architecture doc §5.1, business domain). Never sends outreach itself -- always requires explicit send approval."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Sales specialist agent. You summarize leads and draft outreach messages -- you never send anything yourself, and every outreach draft you produce requires explicit human approval before sending (requires_send_approval must stay true). Keep pipeline_notes factual, not speculative about a prospect's intent.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "lead_summary": "...", "outreach_draft": "...", "pipeline_notes": ["..."], "requires_send_approval": true}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2048,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("sales_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="sales_agent", provider=resp.provider,
                      action="draft", status=result.get("status", "blocked"),
                      output=result)
    return result
