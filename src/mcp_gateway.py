"""
MCP gateway (architecture doc §11).
Agents never connect to MCP servers directly — they call through this
gateway, which enforces per-agent server allowlists, rate limits, and
logs every call into the shared task state's agent_log.

This uses the official MCP Python SDK (`pip install mcp`) to talk to
MCP servers over stdio or SSE. Swap the transport for your deployment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import time


# --- Per-agent allowlist (architecture doc §11.3) ---
AGENT_MCP_PERMISSIONS: dict[str, dict[str, str]] = {
    "research_agent":        {"web_search": "read_only"},
    "finance_agent":         {"spreadsheet": "propose_write"},
    "legal_agent":           {"document_store": "draft_write"},
    "coding_agent":          {"code_sandbox": "sandbox_write"},
    "scheduling_agent":      {"calendar": "propose_write"},
    "correspondence_agent":  {"email": "draft_write"},   # never "send"
    "security_analyzer":     {"*": "audit_read_only"},   # every server, read-only
}

RATE_LIMIT_PER_MINUTE = 30


@dataclass
class MCPGateway:
    servers: dict[str, "MCPServerHandle"] = field(default_factory=dict)
    _call_counts: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def register_server(self, name: str, handle: "MCPServerHandle") -> None:
        self.servers[name] = handle

    def _check_permission(self, agent: str, server: str) -> str:
        perms = AGENT_MCP_PERMISSIONS.get(agent, {})
        if "*" in perms:
            return perms["*"]
        if server not in perms:
            raise PermissionError(f"{agent} is not permitted to call MCP server '{server}'")
        return perms[server]

    def _check_rate_limit(self, agent: str) -> None:
        now = time.time()
        window = [t for t in self._call_counts[agent] if now - t < 60]
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            raise RuntimeError(f"{agent} exceeded {RATE_LIMIT_PER_MINUTE} MCP calls/minute")
        window.append(now)
        self._call_counts[agent] = window

    def call(self, agent: str, server: str, tool: str, arguments: dict, state) -> dict:
        """The single choke point every agent's tool call goes through."""
        access_level = self._check_permission(agent, server)
        self._check_rate_limit(agent)

        if server not in self.servers:
            raise ValueError(f"MCP server '{server}' is not registered with the gateway")

        result = self.servers[server].call_tool(tool, arguments)

        # Log every call into shared state (architecture doc §6, §11.4)
        from .state import log_agent_action
        log_agent_action(
            state, agent=agent, provider=f"mcp:{server}",
            action=f"{tool}({arguments})", status="complete",
            output={"access_level": access_level, "result_preview": str(result)[:500]},
        )
        return result


class MCPServerHandle:
    """Thin wrapper around a live MCP server connection.
    Replace the body of call_tool with the real MCP SDK client call."""

    def __init__(self, name: str, client):
        self.name = name
        self.client = client  # an mcp.ClientSession instance

    def call_tool(self, tool: str, arguments: dict):
        # Real implementation (async in practice):
        #   result = await self.client.call_tool(tool, arguments)
        #   return result.content
        raise NotImplementedError("Wire this up to your actual MCP ClientSession")
