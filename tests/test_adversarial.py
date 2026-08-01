"""
Adversarial tests (roadmap Phase 5) -- NO live API calls, no cost, fully
deterministic. These simulate the exact failure modes the roadmap calls
out explicitly:

  - "an agent requesting out-of-scope tool access" -> test_unauthorized_agent_never_executes
  - "a simulated runaway agent" -> test_blocked_agent_halts_chain_immediately
  - security veto power, both timings -> test_post_execution_security_block_halts_chain
                                           test_pre_execution_security_block_prevents_agent_from_running
  - "embedded prompt injection in a fetched web result" -> test_agents_handling_untrusted_content_warn_against_injection
    (adapted to a structural check -- we can't reliably inject content into
    real live search results in an automated test, so this confirms every
    agent whose job involves untrusted external content explicitly
    instructs the model not to follow embedded instructions in it)

Run before every commit, alongside test_structure.py:
    python tests/test_adversarial.py
"""
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import new_task_state
from src import graph as graph_module


def _fresh_state(goal="test goal"):
    return new_task_state(user_goal=goal, domain_tags=[])


def test_unauthorized_agent_never_executes():
    """An agent name that isn't in AGENT_REGISTRY (e.g. a hallucinated or
    tampered routing decision) must never be invoked -- it should be
    recorded as unrouted and skipped, not silently attempted."""
    state = _fresh_state()
    state["context"]["next_agents"] = ["totally_fake_agent_not_in_roster"]

    result = graph_module.dispatch_node(state)

    assert result["context"].get("unrouted_agents") == ["totally_fake_agent_not_in_roster"], \
        "Unauthorized agent name was not recorded as unrouted"
    assert result["context"]["agent_results"] == {}, \
        "An unauthorized agent name somehow produced a result -- it should never run"


def test_blocked_agent_halts_chain_immediately():
    """If an agent's own output has status='blocked', no later agent in the
    same dispatch chain should run -- a failure must not silently let
    downstream agents build on broken output (this is the 'runaway agent'
    case: a failing agent shouldn't let the chain plow ahead regardless)."""
    downstream_was_called = {"value": False}

    def fake_blocking_agent(state, task_packet):
        return {"status": "blocked", "error": "simulated_failure"}

    def fake_downstream_agent(state, task_packet):
        downstream_was_called["value"] = True
        return {"status": "complete", "summary": "should never run"}

    state = _fresh_state()
    state["context"]["next_agents"] = ["fake_blocking_agent", "fake_downstream_agent"]

    with patch.dict(graph_module.AGENT_REGISTRY, {
        "fake_blocking_agent": fake_blocking_agent,
        "fake_downstream_agent": fake_downstream_agent,
    }):
        result = graph_module.dispatch_node(state)

    assert not downstream_was_called["value"], \
        "Downstream agent ran even though the prior agent in the chain was blocked"
    assert len(result["pending_approvals"]) == 1, \
        f"Expected exactly one pending_approvals entry, got {len(result['pending_approvals'])}"
    assert result["pending_approvals"][0]["agent"] == "fake_blocking_agent"


def test_post_execution_security_block_halts_chain():
    """If security_analyzer.review() (the AFTER-the-fact check) flags an
    action as severity='block', no later agent should run -- security has
    real veto power, not just an advisory note."""
    downstream_was_called = {"value": False}

    def fake_low_risk_agent(state, task_packet):
        return {"status": "complete", "summary": "looks fine on the surface"}

    def fake_downstream_agent(state, task_packet):
        downstream_was_called["value"] = True
        return {"status": "complete", "summary": "should never run"}

    def fake_review_blocks(state, last_action, provider="anthropic"):
        return {"status": "complete", "severity": "block",
                "findings": [{"category": "test", "description": "simulated block",
                               "evidence": "test"}],
                "recommendation": "simulated post-execution veto"}

    state = _fresh_state()
    state["context"]["next_agents"] = ["fake_low_risk_agent", "fake_downstream_agent"]

    with patch.dict(graph_module.AGENT_REGISTRY, {
        "fake_low_risk_agent": fake_low_risk_agent,
        "fake_downstream_agent": fake_downstream_agent,
    }), patch.object(graph_module.security_analyzer, "review", fake_review_blocks):
        result = graph_module.dispatch_node(state)

    assert not downstream_was_called["value"], \
        "Downstream agent ran even though security_analyzer.review() blocked the prior action"
    assert len(result["pending_approvals"]) == 1


def test_pre_execution_security_block_prevents_agent_from_running():
    """Phase 5's actual new behavior: for a medium/high risk agent,
    security_analyzer.review_proposal() runs BEFORE the agent is invoked.
    If it blocks, the agent function itself must NEVER be called -- no API
    cost spent, no output produced at all."""
    agent_was_called = {"value": False}

    def fake_high_risk_agent(state, task_packet):
        agent_was_called["value"] = True
        return {"status": "complete", "summary": "should never run"}

    def fake_review_proposal_blocks(state, agent_name, task_packet, provider="anthropic"):
        return {"status": "complete", "severity": "block",
                "findings": [{"category": "test", "description": "simulated pre-execution block",
                               "evidence": "test"}],
                "recommendation": "simulated pre-execution veto"}

    state = _fresh_state()
    state["context"]["next_agents"] = ["fake_high_risk_agent"]

    with patch.dict(graph_module.AGENT_REGISTRY, {"fake_high_risk_agent": fake_high_risk_agent}), \
         patch.dict(graph_module.AGENT_RISK_TIER, {"fake_high_risk_agent": "high"}), \
         patch.object(graph_module.security_analyzer, "review_proposal", fake_review_proposal_blocks):
        result = graph_module.dispatch_node(state)

    assert not agent_was_called["value"], \
        "Agent function was called even though pre-execution review blocked it -- Phase 5 gate did not work"
    assert result["context"]["agent_results"] == {}, \
        "agent_results contains an entry for an agent that should never have run"
    assert len(result["pending_approvals"]) == 1
    assert "Pre-execution review blocked" in result["pending_approvals"][0]["reason"]


def test_agents_handling_untrusted_content_warn_against_injection():
    """Structural check standing in for a live prompt-injection test (we
    can't reliably inject content into real search results in an automated
    test). research_agent is the one agent whose input includes real
    external content (web search results) -- its system prompt must
    explicitly instruct the model to treat that content as data, not
    instructions, or a malicious search result could hijack its behavior."""
    from src.agents import research_agent

    prompt_lower = research_agent.SYSTEM_PROMPT.lower()
    has_injection_guard = (
        "treat" in prompt_lower and "data" in prompt_lower and
        ("instruction" in prompt_lower or "direct your behavior" in prompt_lower)
    )
    assert has_injection_guard, (
        "research_agent's SYSTEM_PROMPT no longer contains explicit guidance "
        "to treat fetched web content as data rather than instructions -- "
        "this is the only defense against a malicious search result "
        "hijacking the agent's behavior. Do not remove this without "
        "replacing it with an equivalent safeguard."
    )


if __name__ == "__main__":
    tests = [
        test_unauthorized_agent_never_executes,
        test_blocked_agent_halts_chain_immediately,
        test_post_execution_security_block_halts_chain,
        test_pre_execution_security_block_prevents_agent_from_running,
        test_agents_handling_untrusted_content_warn_against_injection,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
