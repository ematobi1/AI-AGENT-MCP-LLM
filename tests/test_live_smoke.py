"""
Live smoke tests -- these DO hit real APIs (small cost, ~10-30 seconds each).
Run manually after any change to graph.py, llm_adapter.py, or an agent's
system prompt -- structural tests can't catch a prompt that produces
malformed JSON or a routing decision that picks the wrong agent.

Run:
    python tests/test_live_smoke.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.graph import build_graph
from src.state import new_task_state


def run_case(label: str, user_goal: str, expected_agent_in_log: str) -> bool:
    print(f"\n--- {label} ---")
    print(f"Goal: {user_goal}")
    app = build_graph()
    state = new_task_state(user_goal=user_goal, domain_tags=[])
    result = app.invoke(state)

    agents_run = [e["agent"] for e in result["agent_log"]]
    statuses = {e["agent"]: e["status"] for e in result["agent_log"]}
    print(f"Agents run: {agents_run}")
    print(f"Statuses: {statuses}")

    ok = True
    if expected_agent_in_log not in agents_run:
        print(f"FAIL: expected '{expected_agent_in_log}' in agent log, got {agents_run}")
        ok = False
    if result["pending_approvals"]:
        print(f"NOTE: run ended blocked -- {result['pending_approvals']}")
        # Not automatically a failure -- a real security block is correct
        # behavior, not a bug. Read the reason before assuming this is broken.
    else:
        print(f"Final output (truncated): {str(result['final_output'])[:200]}...")

    return ok


if __name__ == "__main__":
    cases = [
        ("Research routing", "Summarize recent developments in solid-state batteries", "research_agent"),
        ("Marketing routing", "Draft a short marketing email announcing a new productivity app", "marketing_agent"),
        ("Legal routing (should never self-complete)", "Review this NDA clause for risk: the contractor retains all IP rights indefinitely", "legal_agent"),
    ]

    passed, failed = 0, 0
    for label, goal, expected in cases:
        try:
            if run_case(label, goal, expected):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR in '{label}': {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}\n{passed} passed, {failed} failed (of {len(cases)} live cases)")
    sys.exit(1 if failed else 0)
