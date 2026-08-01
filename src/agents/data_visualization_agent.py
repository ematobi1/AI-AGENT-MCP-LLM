"""Data visualization agent (architecture doc §5.1, cross-cutting). NOTE: this LLM-only stub proposes a chart spec; wiring to a real chart-rendering tool is a follow-up integration."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Data Visualization specialist agent. Given data or a data_analytics_agent output, propose the most appropriate chart type and a chart_spec describing axes, series, and labels. NOTE: in this implementation you do not render an actual image -- you only produce the specification for a downstream renderer.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "chart_type": "bar|line|pie|...", "chart_spec": {}, "data_summary": "..."}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=1536,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("data_visualization_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="data_visualization_agent", provider=resp.provider,
                      action="visualize", status=result.get("status", "blocked"),
                      output=result)
    return result
