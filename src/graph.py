"""
Wires the orchestrator, domain agents, and security analyzer into a
LangGraph StateGraph. This is Step 3 onward from the build order
(architecture doc Â§12) â€” start with orchestrator + research_agent only,
then add nodes one at a time.

pip install langgraph
"""
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
load_dotenv()

from .state import TaskState, new_task_state
from .agents import orchestrator, research_agent, security_analyzer


def orchestrator_node(state: TaskState) -> TaskState:
    decision = orchestrator.route(state)
    state["domain_tags"] = decision.get("domain_tags", [])
    state["risk_level"] = decision.get("risk_level", "low")
    state["context"]["next_agents"] = decision.get("next_agents", [])
    return state


def research_node(state: TaskState) -> TaskState:
    task_packet = {"goal": state["user_goal"], "context": state["context"]}
    result = research_agent.run(state, task_packet)
    state["context"]["research_result"] = result
    return state


def security_node(state: TaskState) -> TaskState:
    last_action = state["agent_log"][-1] if state["agent_log"] else {}
    verdict = security_analyzer.review(state, last_action)
    if verdict.get("severity") == "block":
        reason = verdict.get("recommendation") or "Security review could not validate this action."
        state["pending_approvals"].append({
            "reason": reason,
            "findings": verdict.get("findings", []),
        })
        state["final_output"] = f"Blocked pending review: {reason}"
    return state


def synthesis_node(state: TaskState) -> TaskState:
    research = state["context"].get("research_result", {})
    state["final_output"] = research.get("summary", "No output produced.")
    return state


def route_after_security(state: TaskState) -> str:
    return "blocked" if state["pending_approvals"] else "continue"


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("security_analyzer", security_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "research_agent")
    graph.add_edge("research_agent", "security_analyzer")
    graph.add_conditional_edges(
        "security_analyzer",
        route_after_security,
        {"continue": "synthesis", "blocked": END},  # blocked â†’ surface to user instead
    )
    graph.add_edge("synthesis", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    state = new_task_state(user_goal="Summarize recent developments in solid-state batteries", domain_tags=["research"])
    result = app.invoke(state)
    print(result["final_output"])
    print("\n--- agent log ---")
    for entry in result["agent_log"]:
        print(entry)

