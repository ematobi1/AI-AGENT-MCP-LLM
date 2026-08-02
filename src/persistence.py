"""
Observability & persistence layer (roadmap Phase 7).

Before this, the only record of a run was whatever printed to stdout --
gone the moment the terminal closed. This persists every task run to a
local SQLite database (agent_runs.db, created next to this file's project
root on first use) so cost, token usage, and full agent history survive
past a single process.

Also provides real-time alerting: alert_on_security_block() fires the
MOMENT security_analyzer flags severity="block" (called from graph.py's
dispatch_node directly, not after the fact) -- a loud, unmissable console
alert today; the same call site is where you'd wire in a real notification
channel (Slack webhook, email, PagerDuty) for an actual deployment.

NOTE ON db_path DEFAULTS: every function below takes db_path=None and
resolves it to DB_PATH INSIDE the function body, not as the parameter's
default value. This is deliberate -- `def f(db_path=DB_PATH)` bakes in
whatever DB_PATH was at import time, so reassigning persistence.DB_PATH
later (e.g. to point at a different file) would silently be ignored.
Resolving inside the function body means it always reads the current
value of DB_PATH.
"""
from __future__ import annotations
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_runs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS task_runs (
    task_id TEXT PRIMARY KEY,
    user_goal TEXT NOT NULL,
    domain_tags TEXT,
    risk_level TEXT,
    final_output TEXT,
    was_blocked INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL,
    total_input_tokens INTEGER,
    total_output_tokens INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_log_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    action TEXT,
    status TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    timestamp TEXT,
    output_json TEXT,
    FOREIGN KEY (task_id) REFERENCES task_runs(task_id)
);

CREATE TABLE IF NOT EXISTS security_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    agent TEXT,
    category TEXT,
    description TEXT,
    evidence TEXT,
    FOREIGN KEY (task_id) REFERENCES task_runs(task_id)
);
"""


def _resolve_db_path(db_path: Optional[str]) -> str:
    return db_path if db_path is not None else DB_PATH


@contextmanager
def _connect(db_path: Optional[str] = None):
    resolved = _resolve_db_path(db_path)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def alert_on_security_block(agent_name: str, verdict: dict, stage: str) -> None:
    """Fires the moment a block happens, from inside dispatch_node -- not
    after the run finishes. stage is "pre-execution" or "post-execution".
    This is intentionally loud (impossible to miss scrolling past it in a
    terminal). Swap the print() for a real notification call (Slack
    webhook, email, PagerDuty, etc.) here for a production deployment --
    this is the one place that needs to change."""
    print("\n" + "!" * 60)
    print(f"SECURITY ALERT -- {stage} block on '{agent_name}'")
    print("!" * 60)
    print(f"Recommendation: {verdict.get('recommendation', 'none provided')}")
    findings = verdict.get("findings", [])
    if findings:
        print(f"Findings ({len(findings)}):")
        for f in findings:
            print(f"  - [{f.get('category', '?')}] {f.get('description', '')}")
    print("!" * 60 + "\n")


def persist_task_run(state: dict, db_path: Optional[str] = None) -> None:
    """Call once, after a task run finishes (blocked or completed) --
    writes the full record: the task itself, every agent log entry with
    its token/cost data, and every security finding accumulated."""
    init_db(db_path)

    agent_log = state.get("agent_log", [])
    total_cost = sum(e.get("cost_usd") or 0 for e in agent_log) or None
    total_input = sum(e.get("input_tokens") or 0 for e in agent_log) or None
    total_output = sum(e.get("output_tokens") or 0 for e in agent_log) or None
    was_blocked = 1 if state.get("pending_approvals") else 0

    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO task_runs
               (task_id, user_goal, domain_tags, risk_level, final_output,
                was_blocked, total_cost_usd, total_input_tokens, total_output_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state["task_id"], state["user_goal"],
                json.dumps(state.get("domain_tags", [])), state.get("risk_level"),
                state.get("final_output"), was_blocked,
                total_cost, total_input, total_output,
            ),
        )

        for entry in agent_log:
            conn.execute(
                """INSERT INTO agent_log_entries
                   (task_id, agent, provider, model, action, status,
                    input_tokens, output_tokens, cost_usd, latency_ms,
                    timestamp, output_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    state["task_id"], entry.get("agent"), entry.get("provider"),
                    entry.get("model"), entry.get("action"), entry.get("status"),
                    entry.get("input_tokens"), entry.get("output_tokens"),
                    entry.get("cost_usd"), entry.get("latency_ms"),
                    entry.get("timestamp"), json.dumps(entry.get("output", {})),
                ),
            )

        for finding in state.get("security_findings", []):
            conn.execute(
                """INSERT INTO security_findings (task_id, category, description, evidence)
                   VALUES (?, ?, ?, ?)""",
                (
                    state["task_id"], finding.get("category"),
                    finding.get("description"), finding.get("evidence"),
                ),
            )


def get_task_summary(task_id: str, db_path: Optional[str] = None) -> dict[str, Any] | None:
    init_db(db_path)  # safe no-op if schema already exists; prevents a crash on a fresh DB
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None


def get_all_tasks(db_path: Optional[str] = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM task_runs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_cost_by_agent(db_path: Optional[str] = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT agent,
                      COUNT(*) as call_count,
                      SUM(input_tokens) as total_input_tokens,
                      SUM(output_tokens) as total_output_tokens,
                      SUM(cost_usd) as total_cost_usd,
                      AVG(latency_ms) as avg_latency_ms
               FROM agent_log_entries
               GROUP BY agent
               ORDER BY total_cost_usd DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
