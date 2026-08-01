"""Marketing agent (architecture doc §5.1, business domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Marketing specialist agent. You draft marketing copy -- emails, social posts, landing page copy -- matched to a stated channel and tone. You never fabricate claims, statistics, testimonials, or guarantees about the product/service; if the task doesn't supply a fact, write around it rather than inventing one.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "draft_content": "...", "channel": "email|social|landing_page|...", "tone": "...", "call_to_action": "..." }"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2048,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("marketing_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="marketing_agent", provider=resp.provider,
                      action="draft", status=result.get("status", "blocked"),
                      output=result)
    return result
