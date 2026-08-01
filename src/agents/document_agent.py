"""Document generation agent (architecture doc §5.1, cross-cutting). NOTE: this LLM-only stub describes the document it would produce; wiring to a real docx/pdf/pptx writer is a follow-up integration."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Document specialist agent. Given a request for a document (report, memo, letter, etc.), describe what the finished document should contain and its structure. NOTE: in this implementation you do not yet have a real file-writing tool connected -- set file_path to a proposed filename only, and make clear in the summary that no file has actually been written yet.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "needs_approval", "file_path": "proposed_filename.docx", "file_type": "docx|pdf|pptx|xlsx", "summary": "..."}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=1536,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("document_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="document_agent", provider=resp.provider,
                      action="generate", status=result.get("status", "blocked"),
                      output=result)
    return result
