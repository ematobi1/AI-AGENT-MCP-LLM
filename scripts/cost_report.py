"""
Cost & observability report (roadmap Phase 7).

Reads agent_runs.db (created automatically by persistence.py the first
time any task run completes) and prints a summary: total spend, spend
broken down per agent, and a list of recent task runs with their cost and
whether they ended up blocked.

Run:
    python scripts/cost_report.py
    python scripts/cost_report.py --task <task_id>     # detail for one task
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import persistence


def print_overview():
    all_tasks = persistence.get_all_tasks()
    if not all_tasks:
        print("No task runs recorded yet. Run the pipeline at least once "
              "(e.g. `python -m src.graph`) to populate agent_runs.db.")
        return

    total_cost = sum(t.get("total_cost_usd") or 0 for t in all_tasks)
    total_input = sum(t.get("total_input_tokens") or 0 for t in all_tasks)
    total_output = sum(t.get("total_output_tokens") or 0 for t in all_tasks)
    blocked_count = sum(1 for t in all_tasks if t.get("was_blocked"))

    print("=" * 70)
    print("COST & OBSERVABILITY REPORT")
    print("=" * 70)
    print(f"Total task runs:     {len(all_tasks)}")
    print(f"Blocked runs:        {blocked_count} ({blocked_count / len(all_tasks):.0%})")
    print(f"Total cost:          ${total_cost:.4f}")
    print(f"Total input tokens:  {total_input:,}")
    print(f"Total output tokens: {total_output:,}")

    print("\n" + "-" * 70)
    print("COST BY AGENT")
    print("-" * 70)
    by_agent = persistence.get_cost_by_agent()
    if by_agent:
        print(f"{'Agent':<28} {'Calls':>6} {'In Tok':>10} {'Out Tok':>10} {'Cost':>10} {'Avg ms':>9}")
        for row in by_agent:
            print(f"{row['agent']:<28} {row['call_count']:>6} "
                  f"{row['total_input_tokens'] or 0:>10,} {row['total_output_tokens'] or 0:>10,} "
                  f"${row['total_cost_usd'] or 0:>8.4f} {row['avg_latency_ms'] or 0:>8.0f}")
    else:
        print("No agent-level data yet.")

    print("\n" + "-" * 70)
    print("RECENT TASK RUNS (most recent first)")
    print("-" * 70)
    for t in all_tasks[:10]:
        status = "BLOCKED" if t.get("was_blocked") else "ok"
        cost = t.get("total_cost_usd") or 0
        goal = (t.get("user_goal") or "")[:50]
        print(f"[{status:>7}] ${cost:>7.4f}  {t['task_id'][:8]}...  {goal}")


def print_task_detail(task_id: str):
    task = persistence.get_task_summary(task_id)
    if not task:
        # allow partial task_id match
        matches = [t for t in persistence.get_all_tasks() if t["task_id"].startswith(task_id)]
        if len(matches) == 1:
            task = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous task_id prefix '{task_id}' matches {len(matches)} tasks. Be more specific.")
            return
        else:
            print(f"No task found matching '{task_id}'.")
            return

    print("=" * 70)
    print(f"TASK DETAIL: {task['task_id']}")
    print("=" * 70)
    for k, v in task.items():
        print(f"{k:<22} {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost & observability report for the agent pipeline")
    parser.add_argument("--task", help="Show detail for a specific task_id (or unique prefix)")
    args = parser.parse_args()

    if args.task:
        print_task_detail(args.task)
    else:
        print_overview()
