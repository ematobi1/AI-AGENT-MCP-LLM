"""
Wires the orchestrator, all 22 roster agents, and the security analyzer into
a LangGraph StateGraph (architecture doc §12), with a REAL human approval
pause (architecture doc Phase 4) -- not just a status label.

Previously, an agent returning status="needs_approval" (e.g. legal_agent,
correspondence_agent) didn't actually stop anything -- the chain just kept
going. That defeats the purpose of the status. Now, dispatch_node calls
LangGraph's interrupt() the moment that happens: execution genuinely
pauses, state is checkpointed, and the caller must explicitly approve or
reject before anything continues.

pip install langgraph
"""
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # older langgraph versions used this name
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from dotenv import load_dotenv
load_dotenv()

from .state import TaskState, new_task_state
from .agents import (
    orchestrator,
    security_analyzer,
    research_agent,
    finance_agent,
    legal_agent,
    marketing_agent,
    sales_agent,
    ops_project_agent,
    hr_agent,
    coding_agent,
    data_analytics_agent,
    scheduling_agent,
    personal_finance_agent,
    health_wellness_agent,
    travel_agent,
    learning_tutor_agent,
    home_admin_agent,
    correspondence_agent,
    writing_editing_agent,
    document_agent,
    data_visualization_agent,
    qa_review_agent,
    memory_context_agent,
)

# Every domain/cross-cutting agent the orchestrator is allowed to route to.
# Keys MUST match schemas.AGENT_SCHEMA_REGISTRY keys exactly -- that's what
# ties routing, execution, and validation together.
AGENT_REGISTRY = {
    "research_agent": research_agent.run,
    "finance_agent": finance_agent.run,
    "legal_agent": legal_agent.run,
    "marketing_agent": marketing_agent.run,
    "sales_agent": sales_agent.run,
    "ops_project_agent": ops_project_agent.run,
    "hr_agent": hr_agent.run,
    "coding_agent": coding_agent.run,
    "data_analytics_agent": data_analytics_agent.run,
    "scheduling_agent": scheduling_agent.run,
    "personal_finance_agent": personal_finance_agent.run,
    "health_wellness_agent": health_wellness_agent.run,
    "travel_agent": travel_agent.run,
    "learning_tutor_agent": learning_tutor_agent.run,
    "home_admin_agent": home_admin_agent.run,
    "correspondence_agent": correspondence_agent.run,
    "writing_editing_agent": writing_editing_agent.run,
    "document_agent": document_agent.run,
    "data_visualization_agent": data_visualization_agent.run,
    "qa_review_agent": qa_review_agent.run,
    "memory_context_agent": memory_context_agent.run,
}

# Fields checked in priority order to build a human-readable synthesis line
# out of whatever shape a given agent's output happens to be.
_SUMMARY_FIELDS = (
    "summary", "draft_content", "explanation", "redline_summary",
    "budget_summary", "analysis_summary", "general_guidance",
    "draft_message", "edited_content", "timeline_summary",
    "data_summary", "bills_summary",
)


def orchestrator_node(state: TaskState) -> TaskState:
    decision = orchestrator.route(state)
    state["domain_tags"] = decision.get("domain_tags", [])
    state["risk_level"] = decision.get("risk_level", "low")
    state["context"]["next_agents"] = decision.get("next_agents", [])
    return state


def dispatch_node(state: TaskState) -> TaskState:
    """Runs every agent the orchestrator selected, in order. After each one:

    1. If the agent's own output has status "needs_approval" (e.g. legal_agent,
       correspondence_agent), execution genuinely PAUSES here via interrupt().
       The caller must resume with Command(resume=True) to approve or
       Command(resume=False) to reject -- there is no way to silently skip
       past this. Rejecting halts the chain immediately.
    2. If the agent's own output has status "blocked", halt immediately --
       a failed result never flows into the next agent.
    3. Otherwise, security_analyzer reviews the action; if it flags "block",
       halt immediately too.
    """
    next_agents = state["context"].get("next_agents", [])
    task_packet = {"goal": state["user_goal"], "context": state["context"]}
    state["context"].setdefault("agent_results", {})

    for agent_name in next_agents:
        agent_fn = AGENT_REGISTRY.get(agent_name)
        if agent_fn is None:
            state["context"].setdefault("unrouted_agents", []).append(agent_name)
            continue

        result = agent_fn(state, task_packet)
        state["context"]["agent_results"][agent_name] = result

        if result.get("status") == "needs_approval":
            approved = interrupt({
                "type": "approval_required",
                "agent": agent_name,
                "output": result,
                "message": f"{agent_name} requires human approval before the "
                           f"pipeline can continue. Review the output above "
                           f"and resume with Command(resume=True) to approve "
                           f"or Command(resume=False) to reject.",
            })
            if not approved:
                state["pending_approvals"].append({
                    "agent": agent_name,
                    "reason": f"Human rejected {agent_name}'s output during approval review.",
                    "findings": [],
                })
                break
            # Approved: fall through to the normal blocked/security checks
            # below -- approval does not skip the security review.

        if result.get("status") == "blocked":
            state["pending_approvals"].append({
                "agent": agent_name,
                "reason": f"{agent_name} failed: {result.get('error') or result.get('detail') or 'blocked status'}",
                "findings": [],
            })
            break

        last_action = state["agent_log"][-1] if state["agent_log"] else {}
        verdict = security_analyzer.review(state, last_action)
        if verdict.get("severity") == "block":
            state["pending_approvals"].append({
                "agent": agent_name,
                "reason": verdict.get("recommendation"),
                "findings": verdict.get("findings", []),
            })
            break

    return state


def synthesis_node(state: TaskState) -> TaskState:
    results = state["context"].get("agent_results", {})
    if not results:
        state["final_output"] = "No agent produced output for this task."
        return state

    parts = []
    for agent_name, result in results.items():
        text = next((result[f] for f in _SUMMARY_FIELDS if result.get(f)), None)
        parts.append(f"[{agent_name}] {text if text else result}")
    state["final_output"] = "\n\n".join(parts)
    return state


def blocked_node(state: TaskState) -> TaskState:
    """Populates final_output even when the chain stopped early, so the
    caller always gets a legible message instead of None."""
    reasons = "; ".join(
        f"{pa.get('agent', 'unknown')}: {pa.get('reason', 'no reason given')}"
        for pa in state["pending_approvals"]
    )
    state["final_output"] = f"Blocked pending review: {reasons}"
    return state


def route_after_dispatch(state: TaskState) -> str:
    return "blocked" if state["pending_approvals"] else "continue"


def build_graph():
    """A checkpointer is REQUIRED for interrupt() to work -- without one,
    LangGraph has nowhere to persist state across the pause. InMemorySaver
    is fine for a single process's lifetime (testing, a CLI session); if
    the process restarts mid-approval, that state is lost. For a real
    deployment where approvals might span a restart, swap in
    langgraph.checkpoint.sqlite.SqliteSaver (or Postgres) instead -- same
    interface, durable storage."""
    graph = StateGraph(TaskState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("blocked", blocked_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "dispatch")
    graph.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {"continue": "synthesis", "blocked": "blocked"},
    )
    graph.add_edge("synthesis", END)
    graph.add_edge("blocked", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_with_cli_approval(app, initial_state, thread_id: str):
    """Runs the graph and handles any approval interrupt with a real CLI
    prompt -- this is the reference implementation of the human side of the
    pause/resume loop. A production system would replace input() with
    whatever surfaces the pending approval to an actual reviewer (a UI, a
    Slack message, an email) and whatever captures their response -- the
    interrupt()/Command(resume=...) mechanics underneath stay the same."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n" + "=" * 60)
        print("APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Agent: {payload.get('agent')}")
        print(f"Output: {payload.get('output')}")
        print(f"\n{payload.get('message')}")
        answer = input("\nApprove? [y/N]: ").strip().lower()
        approved = answer == "y"
        result = app.invoke(Command(resume=approved), config=config)

    return result


if __name__ == "__main__":
    import uuid

    app = build_graph()
    state = new_task_state(
        user_goal="Summarize recent developments in solid-state batteries",
        domain_tags=[],
    )
    result = run_with_cli_approval(app, state, thread_id=str(uuid.uuid4()))
    print(result["final_output"])
    print("\n--- agent log ---")
    for entry in result["agent_log"]:
        print(entry)
