# Operations Runbook

This document is the answer to: **"something's wrong, what do I do?"** and
**"how do I safely change this system?"** — written so a new team member (or
you, six months from now) doesn't have to ask.

---

## 1. Emergency stop (kill switch)

If the system is doing something wrong and you need it to stop **immediately**,
you have two options — either works, use whichever is faster in the moment:

**Option A — environment variable** (works anywhere, including containers):
```powershell
$env:AGENT_SYSTEM_DISABLED = "1"
```
Set this in your actual deployment's environment (not just your current
terminal session) if the system is running as a service.

**Option B — a file** (works for a local operator without touching env config):
```powershell
New-Item .KILL_SWITCH -ItemType File
```

**Effect:** every new task is refused at the very first step (`orchestrator_node`),
before any API call happens — zero cost, zero output, immediate halt. Tasks
already mid-execution when the switch is flipped will finish their current
step but won't be able to start a new one.

**To resume normal operation:**
```powershell
Remove-Item Env:AGENT_SYSTEM_DISABLED   # if you used the env var
Remove-Item .KILL_SWITCH                 # if you used the file
```

**Verify it's actually off:**
```powershell
python -c "from src import config; print('Kill switch active:', config.is_kill_switch_active())"
```
Should print `False`.

---

## 2. Disabling a single misbehaving agent

If one specific agent is producing bad output (not the whole system), you
don't need the kill switch — you can pull just that agent:

```powershell
$env:DISABLED_AGENTS = "legal_agent,coding_agent"
```

Comma-separated, no spaces required (whitespace is trimmed automatically).
Any agent name in this list is skipped by `dispatch_node` before it's even
looked up — it's recorded in `state["context"]["disabled_agents_skipped"]`
so you can see it happened, but the agent never runs and costs nothing.

The orchestrator will still *try* to route to a disabled agent if it thinks
the task needs it — that's expected and fine; the disable check happens at
dispatch time, not at routing time, so you'll see it clearly in the agent
log rather than the orchestrator silently avoiding it.

To re-enable:
```powershell
Remove-Item Env:DISABLED_AGENTS
```

---

## 3. Rollback procedure

Every phase of this project was built and merged as its own commit on
`master`. To roll back to a known-good state:

**See what's available:**
```powershell
git log --oneline
```

**Roll back to a specific commit** (creates a new commit that undoes the
changes — safer than `reset` because it doesn't rewrite history that may
already be pushed):
```powershell
git revert <bad-commit-hash>
git push
```

**If you need to go back multiple commits at once**, revert them in reverse
order (most recent first) or use:
```powershell
git revert --no-commit <oldest-bad-hash>^..<newest-bad-hash>
git commit -m "Revert phases X through Y"
git push
```

**Confirm the rollback actually works** before considering it done — this
isn't optional:
```powershell
python tests\test_structure.py
python tests\test_adversarial.py
python tests\test_persistence.py
python -m src.graph
```
All three test files should pass, and the live run should complete cleanly.

---

## 4. Secret rotation

`.env` is fine for local development but is explicitly **not** a production
secrets store — see the top of `src/config.py` for why, and for the exact
swap-in point when you're ready to move to a real secrets manager (AWS
Secrets Manager, Azure Key Vault, HashiCorp Vault, GCP Secret Manager all
work the same way from this codebase's perspective: they just need to set
the same environment variables before the process starts).

**If a key is compromised (leaked in chat, committed to git, etc.):**

1. Revoke it immediately on the provider's dashboard:
   - Anthropic: `https://console.anthropic.com/settings/keys`
   - OpenAI: `https://platform.openai.com/api-keys`
2. Generate a new key.
3. Update `.env` with the new key — never paste a key anywhere except `.env`.
4. Confirm it works:
   ```powershell
   python test_keys.py
   ```
5. If the old key was ever committed to git (check with
   `git log -p | Select-String "sk-"` — should return nothing), the key is
   compromised in git history even after rotation and revocation doesn't
   erase that history. Revocation is what actually neutralizes it; history
   scrubbing (`git filter-repo` or similar) is a separate, more invasive
   step only worth doing if the repo is public.

---

## 5. Adding a new agent safely

Every agent follows an identical pattern. To add one:

1. **Add a schema** in `src/schemas.py` — a `class YourAgentOutput(AgentOutputBase)`
   with the fields your agent's JSON output should have, and register it
   in `AGENT_SCHEMA_REGISTRY`.
2. **Set its risk tier** in `AGENT_RISK_TIER` (same file) — `"low"` if it's
   purely informational/draft-only, `"medium"` if a human might act on it
   with real consequences, `"high"` if it can affect legal standing, money,
   or reach a real external party. Medium/high automatically get
   pre-execution security review — you don't need to wire that up yourself,
   it's keyed off this dict (confirmed by
   `tests/test_adversarial.py::test_every_medium_high_risk_real_agent_gets_preexecution_review`).
3. **Write the agent file** in `src/agents/your_agent.py` — copy the shape
   of any existing simple agent (e.g. `marketing_agent.py`): a `SYSTEM_PROMPT`
   and a `run(state, task_packet, provider="anthropic") -> dict` function
   that calls `get_adapter()`, parses the response, validates against your
   schema, and calls `log_agent_action(..., response=resp)` — the
   `response=resp` part is required for cost tracking to work; a structural
   test (`test_all_agents_pass_response_for_cost_tracking`) will fail your
   build if you forget it.
4. **Register it** in `src/graph.py`'s `AGENT_REGISTRY` dict and the import
   list at the top of the file.
5. **Update the orchestrator's roster** in `src/agents/orchestrator.py`'s
   `SYSTEM_PROMPT` so it knows the new agent exists and can route to it.
6. **Run the full test suite** before committing:
   ```powershell
   python tests\test_structure.py
   python tests\test_adversarial.py
   python tests\test_persistence.py
   ```
7. **Do a live test** targeting your new agent specifically, same pattern
   as every other phase in this project's history — invoke the graph with
   a goal that should route to it, confirm the agent log shows it running
   and producing schema-valid output.
8. **Commit on a sandbox branch first**, confirm everything works, then
   merge to `master` — this project's entire history follows this pattern;
   don't break the streak.

---

## 6. Where to look when something's wrong

| Symptom | Check |
|---|---|
| A task silently did nothing | Kill switch (`config.is_kill_switch_active()`) or `DISABLED_AGENTS` |
| An agent's output looks malformed | `agent_log` entry's `status` — should be `blocked` with an `error` field if schema validation failed, not silently passed through |
| Unexpectedly high cost | `python scripts\cost_report.py` — breaks down spend per agent |
| A risky action went through without review | Check `AGENT_RISK_TIER` in `schemas.py` — if the agent isn't listed as `medium`/`high`, it only gets post-execution review, not pre-execution |
| Approval flow seems stuck | The process must stay alive between `interrupt()` and `Command(resume=...)` when using `InMemorySaver` (the default) — a restart loses the pending state. Swap to `SqliteSaver`/`PostgresSaver` if approvals need to survive a restart |
| Need full history of what happened | `agent_runs.db` (SQLite) — every run is persisted, query it directly or via `scripts/cost_report.py --task <id>` |
