"""Coding agent (architecture doc §5.1, business domain).

PHASE 8 NOTE (execution sandboxing): this agent produces a code diff as
TEXT ONLY -- it never calls exec(), subprocess, eval(), or any other
execution path, and never will as currently designed. There is
consequently no code-execution sandboxing gap to close today. If a future
version adds an actual "run this code and report the result" capability,
that execution MUST happen in an isolated environment (a disposable Docker
container, gVisor, Firecracker, or equivalent) with no direct network path
to production credentials or the host filesystem -- do not add real
execution to this agent without that isolation in place first."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Coding specialist agent. You produce code changes as a diff, list every file touched, and note whether tests were added. Never write malicious code, obfuscated code whose purpose you wouldn't disclose, or anything designed to exfiltrate data or bypass security controls. If the request asks for that, set status to 'blocked' and explain why in review_notes instead of complying.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "code_diff": "...", "files_changed": ["..."], "tests_added": false, "review_notes": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=4096,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("coding_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="coding_agent", provider=resp.provider, response=resp,
                      action="code", status=result.get("status", "blocked"),
                      output=result)
    return result
