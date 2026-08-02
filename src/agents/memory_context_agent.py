"""Memory / context agent (architecture doc §5.1, cross-cutting). NOTE: this LLM-only stub proposes what to store/retrieve; wiring to a real persistent store is a follow-up integration."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Memory & Context specialist agent. Given the current task state, identify facts worth remembering for future tasks and note what context would be useful to retrieve. NOTE: in this implementation there is no real persistent store wired up yet -- you are only proposing what should be stored/retrieved, not actually writing to durable storage.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "stored_facts": ["..."], "retrieved_context": {}, "expiry": null}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=1536,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("memory_context_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="memory_context_agent", provider=resp.provider, response=resp,
                      action="recall", status=result.get("status", "blocked"),
                      output=result)
    return result
