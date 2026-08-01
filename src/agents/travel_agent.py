"""Travel agent (architecture doc §5.1, personal domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Travel specialist agent. You build day-by-day itineraries and surface booking options and logistics notes. Never invent specific prices, flight numbers, or availability you can't verify -- describe booking_options in general terms (e.g. 'direct flights typically available') rather than fabricating a specific price or confirmation.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "itinerary": [{"day": 1, "location": "...", "notes": "..."}], "booking_options": [{"type": "...", "notes": "..."}], "logistics_notes": ["..."]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2048,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("travel_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="travel_agent", provider=resp.provider,
                      action="plan", status=result.get("status", "blocked"),
                      output=result)
    return result
