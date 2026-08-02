"""Data analytics agent (architecture doc §5.1, business domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Data Analytics specialist agent. You analyze provided data and summarize findings with concrete metrics. Never fabricate a metric that wasn't derivable from the given data -- if you don't have enough data to compute something, say so instead of estimating a plausible-looking number.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "analysis_summary": "...", "key_metrics": {"metric_name": 0.0}, "chart_spec": null}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2048,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("data_analytics_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="data_analytics_agent", provider=resp.provider, response=resp,
                      action="analyze", status=result.get("status", "blocked"),
                      output=result)
    return result
