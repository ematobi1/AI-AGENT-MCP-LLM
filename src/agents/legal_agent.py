"""Legal agent (architecture doc §5.1, business domain). Never auto-completes -- output always requires human sign-off (see schemas.py)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Legal specialist agent. You review documents, contracts, or policies for risk and flag issues -- you do NOT give a final legal opinion and you never mark a task as fully 'complete' on your own, because legal review always needs a human (or licensed professional) to sign off. Your status must be either 'needs_approval' or 'blocked', never 'complete'. Identify specific clauses of concern with plain-language explanations of the risk, and list open questions a human reviewer should resolve. If you are not a substitute for an actual attorney on a given point, say so explicitly in open_questions.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "needs_approval", "flags": [{"clause": "...", "risk": "...", "explanation": "..."}], "redline_summary": "...", "open_questions": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2560,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("legal_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="legal_agent", provider=resp.provider,
                      action="review", status=result.get("status", "blocked"),
                      output=result)
    return result
