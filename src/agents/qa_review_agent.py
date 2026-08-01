"""QA / review agent (architecture doc §5.1, cross-cutting). Independent quality check on another agent's output, separate from the security_analyzer's safety review."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the QA Review specialist agent. You check another agent's output for quality, completeness, and correctness -- this is a quality review, distinct from the security_analyzer's safety/risk review. Be specific about issues found; don't approve something with unaddressed problems just to be agreeable.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "review_verdict": "approved|needs_revision|rejected", "issues_found": ["..."], "suggestions": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2048,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("qa_review_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="qa_review_agent", provider=resp.provider,
                      action="review", status=result.get("status", "blocked"),
                      output=result)
    return result
