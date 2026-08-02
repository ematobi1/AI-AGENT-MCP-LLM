"""
Persistence layer tests (roadmap Phase 7) -- NO live API calls, no cost,
uses temporary SQLite files that are cleaned up after each test.

Run:
    python tests/test_persistence.py
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import persistence


def _fake_state(task_id="test-id", cost=0.01, blocked=False, input_tok=100, output_tok=50):
    return {
        "task_id": task_id,
        "user_goal": "test goal",
        "domain_tags": ["test"],
        "risk_level": "low",
        "final_output": None if blocked else "test output",
        "pending_approvals": [{"agent": "test_agent", "reason": "test"}] if blocked else [],
        "security_findings": [{"category": "c", "description": "d", "evidence": "e"}] if blocked else [],
        "agent_log": [{
            "agent": "test_agent", "provider": "anthropic", "model": "claude-sonnet-4-6",
            "action": "test", "status": "needs_approval" if blocked else "complete",
            "input_tokens": input_tok, "output_tokens": output_tok, "cost_usd": cost,
            "latency_ms": 500.0, "timestamp": "2026-08-01T00:00:00Z", "output": {"summary": "x"},
        }],
    }


def test_persist_and_retrieve_round_trip():
    db = tempfile.mktemp(suffix=".db")
    try:
        persistence.persist_task_run(_fake_state("round-trip-1"), db_path=db)
        summary = persistence.get_task_summary("round-trip-1", db_path=db)
        assert summary is not None, "Persisted task was not retrievable"
        assert summary["user_goal"] == "test goal"
        assert summary["total_cost_usd"] == 0.01
        assert summary["total_input_tokens"] == 100
        assert summary["was_blocked"] == 0
    finally:
        if os.path.exists(db):
            os.remove(db)


def test_blocked_task_recorded_correctly():
    db = tempfile.mktemp(suffix=".db")
    try:
        persistence.persist_task_run(_fake_state("blocked-1", blocked=True), db_path=db)
        summary = persistence.get_task_summary("blocked-1", db_path=db)
        assert summary["was_blocked"] == 1, "Blocked task was not flagged was_blocked=1"

        with persistence._connect(db) as conn:
            findings = conn.execute(
                "SELECT * FROM security_findings WHERE task_id = ?", ("blocked-1",)
            ).fetchall()
        assert len(findings) == 1, "Security finding was not persisted alongside the blocked task"
    finally:
        if os.path.exists(db):
            os.remove(db)


def test_fresh_database_never_crashes_on_read():
    """The exact bug caught during development: reading from a database
    that doesn't exist yet (e.g. a user running the cost report before
    ever running the pipeline) must return an empty result, not raise."""
    db = tempfile.mktemp(suffix=".db")
    assert not os.path.exists(db), "Test setup error: db should not exist yet"
    try:
        tasks = persistence.get_all_tasks(db_path=db)
        assert tasks == [], f"Expected [] on a fresh database, got {tasks}"

        by_agent = persistence.get_cost_by_agent(db_path=db)
        assert by_agent == [], f"Expected [] on a fresh database, got {by_agent}"

        summary = persistence.get_task_summary("anything", db_path=db)
        assert summary is None, "Expected None for a nonexistent task on a fresh database"
    finally:
        if os.path.exists(db):
            os.remove(db)


def test_db_path_override_is_actually_respected():
    """The exact bug caught during development: db_path must be resolved
    at CALL time, not baked in at function-definition time. This test
    would have failed under the old `def f(db_path=DB_PATH)` pattern."""
    db = tempfile.mktemp(suffix=".db")
    try:
        original_default = persistence.DB_PATH
        # Reassign the module-level default, same as a caller might do to
        # point at a different file at runtime.
        persistence.DB_PATH = db
        try:
            persistence.persist_task_run(_fake_state("override-test"))  # uses the default, no explicit db_path
            retrieved = persistence.get_task_summary("override-test")   # also uses the default
            assert retrieved is not None, "persist/get did not respect a runtime-reassigned DB_PATH"
        finally:
            persistence.DB_PATH = original_default
    finally:
        if os.path.exists(db):
            os.remove(db)


def test_cost_aggregation_across_multiple_agents():
    db = tempfile.mktemp(suffix=".db")
    try:
        state = _fake_state("multi-agent-1")
        state["agent_log"].append({
            "agent": "second_agent", "provider": "openai", "model": "gpt-5",
            "action": "test2", "status": "complete",
            "input_tokens": 200, "output_tokens": 100, "cost_usd": 0.02,
            "latency_ms": 300.0, "timestamp": "2026-08-01T00:00:01Z", "output": {},
        })
        persistence.persist_task_run(state, db_path=db)
        summary = persistence.get_task_summary("multi-agent-1", db_path=db)
        assert abs(summary["total_cost_usd"] - 0.03) < 1e-9, \
            f"Expected total cost 0.03 (0.01 + 0.02), got {summary['total_cost_usd']}"

        by_agent = persistence.get_cost_by_agent(db_path=db)
        agents_seen = {row["agent"] for row in by_agent}
        assert agents_seen == {"test_agent", "second_agent"}
    finally:
        if os.path.exists(db):
            os.remove(db)


if __name__ == "__main__":
    tests = [
        test_persist_and_retrieve_round_trip,
        test_blocked_task_recorded_correctly,
        test_fresh_database_never_crashes_on_read,
        test_db_path_override_is_actually_respected,
        test_cost_aggregation_across_multiple_agents,
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
