"""Learning / tutoring agent (architecture doc §5.1, personal domain)."""
from ..llm_adapter import get_adapter, parse_json_response
from ..state import TaskState, log_agent_action
from ..schemas import validate_agent_output

SYSTEM_PROMPT = """You are the Learning Tutor specialist agent. You explain concepts clearly, optionally propose a study plan, and can generate quiz questions with answers to check understanding. Adjust depth to what the task implies about the learner's level; when unsure, explain at an accessible level and offer to go deeper.\n\nRespond with ONLY a JSON object and nothing else -- no preamble, no explanation, no markdown fences.\nFormat: {"status": "complete", "explanation": "...", "study_plan": ["..."], "quiz_items": [{"question": "...", "answer": "..."}]}"""


def run(state: TaskState, task_packet: dict, provider: str = "anthropic") -> dict:
    adapter = get_adapter(provider)
    resp = adapter.complete(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Task: {task_packet.get('goal')}\nContext: {task_packet.get('context', {})}",
        max_tokens=2560,
    )
    result = parse_json_response(resp.content)

    try:
        validated = validate_agent_output("learning_tutor_agent", result)
        result = validated.model_dump()
    except Exception as e:
        result = {"status": "blocked", "error": "schema_validation_failed", "detail": str(e)}

    log_agent_action(state, agent="learning_tutor_agent", provider=resp.provider, response=resp,
                      action="teach", status=result.get("status", "blocked"),
                      output=result)
    return result
