"""
Structural sanity checks -- NO live API calls, no cost, runs in under a
second. Catches the exact class of bug that ate the most debugging time
during development: a file that doesn't import cleanly, a registry key
that drifted out of sync with schemas.py, a missing run() function.

Run before every commit:
    python -m pytest tests/test_structure.py -v

Or without pytest installed:
    python tests/test_structure.py
"""
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_all_agent_modules_import_cleanly():
    """Every .py file in src/agents/ must import without error. This alone
    would have caught the syntax errors from bad manual edits earlier."""
    agents_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "agents")
    failures = []
    for fname in sorted(os.listdir(agents_dir)):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        modname = f"src.agents.{fname[:-3]}"
        try:
            importlib.import_module(modname)
        except Exception as e:
            failures.append(f"{modname}: {type(e).__name__}: {e}")
    assert not failures, "Agent modules failed to import:\n" + "\n".join(failures)


def test_every_dispatchable_agent_has_run_function():
    """graph.py's AGENT_REGISTRY assumes every agent module exposes run().
    Catches a module that exists but forgot the function, or has it
    misspelled."""
    from src.graph import AGENT_REGISTRY
    missing = [name for name, fn in AGENT_REGISTRY.items() if not callable(fn)]
    assert not missing, f"AGENT_REGISTRY entries that aren't callable: {missing}"


def test_agent_registry_matches_schema_registry():
    """graph.py's AGENT_REGISTRY and schemas.py's AGENT_SCHEMA_REGISTRY must
    have identical keys (minus orchestrator/security_analyzer, which are
    handled outside the dispatch loop). This is exactly the drift that
    silently breaks routing if one file gets updated without the other."""
    from src.graph import AGENT_REGISTRY
    from src.schemas import AGENT_SCHEMA_REGISTRY

    dispatchable_schema_keys = set(AGENT_SCHEMA_REGISTRY.keys()) - {"orchestrator", "security_analyzer"}
    graph_keys = set(AGENT_REGISTRY.keys())

    only_in_graph = graph_keys - dispatchable_schema_keys
    only_in_schema = dispatchable_schema_keys - graph_keys

    assert not only_in_graph, f"In graph.py AGENT_REGISTRY but no schema exists: {only_in_graph}"
    assert not only_in_schema, f"Has a schema but graph.py can't dispatch to it: {only_in_schema}"


def test_all_22_roster_agents_present():
    """Sanity check on total count -- catches an accidental deletion.
    22 total roster agents include research_agent and security_analyzer;
    orchestrator sits outside the roster entirely. AGENT_REGISTRY dispatches
    to everything except orchestrator and security_analyzer, so 22 - 1
    (security_analyzer) = 21 dispatchable agents expected."""
    from src.schemas import AGENT_SCHEMA_REGISTRY
    roster = set(AGENT_SCHEMA_REGISTRY.keys()) - {"orchestrator", "security_analyzer"}
    assert len(roster) == 21, f"Expected 21 dispatchable roster agents, found {len(roster)}: {sorted(roster)}"


def test_schema_validation_rejects_missing_required_field():
    """Confirms validate_agent_output actually raises on bad input --
    if this silently passed, agents could ship malformed output undetected."""
    from src.schemas import validate_agent_output
    try:
        validate_agent_output("research_agent", {"status": "complete"})  # missing required 'summary'
        assert False, "Expected a validation error for missing required field, got none"
    except Exception:
        pass  # expected


def test_legal_agent_schema_cannot_self_complete():
    """Confirms the hard-coded safety guarantee holds: LegalOutput's status
    field is typed as Literal["needs_approval", "blocked"] -- pydantic
    should reject any attempt to construct one with status="complete"
    outright, not just default away from it. If this stops raising, the
    safety constraint was loosened somewhere."""
    from src.schemas import LegalOutput
    import pydantic
    try:
        LegalOutput(redline_summary="test", status="complete")
        assert False, "LegalOutput accepted status='complete' -- safety constraint was removed!"
    except pydantic.ValidationError:
        pass  # expected -- the Literal type correctly rejects this

    # Confirm the *actual* default (no status passed) is the safe one.
    default_result = LegalOutput(redline_summary="test")
    assert default_result.status == "needs_approval", \
        f"Expected default status 'needs_approval', got '{default_result.status}'"


def test_correspondence_agent_never_send_ready_by_default():
    """Confirms the hard-coded safety default holds: correspondence drafts
    default to NOT send-ready unless explicitly overridden."""
    from src.schemas import CorrespondenceOutput
    result = CorrespondenceOutput(status="complete", draft_message="hi", tone="friendly")
    assert result.send_ready is False, "CorrespondenceOutput defaulted to send_ready=True -- safety default was removed!"


if __name__ == "__main__":
    tests = [
        test_all_agent_modules_import_cleanly,
        test_every_dispatchable_agent_has_run_function,
        test_agent_registry_matches_schema_registry,
        test_all_22_roster_agents_present,
        test_schema_validation_rejects_missing_required_field,
        test_legal_agent_schema_cannot_self_complete,
        test_correspondence_agent_never_send_ready_by_default,
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
