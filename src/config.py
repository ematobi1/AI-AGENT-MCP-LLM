"""
Centralized config and startup validation (roadmap Phase 8).

SECRETS: this still reads from .env via python-dotenv, which is fine for
local development but is NOT a production secrets store -- .env files can
be accidentally committed, sit in plaintext on disk, and have no access
control or rotation audit trail. For an actual production deployment,
replace load_config()'s dotenv call with a fetch from a real secrets
manager (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault, GCP Secret
Manager) that sets the same environment variables before the process
starts -- the rest of the codebase only depends on the env vars being SET,
not on how they got there, so this is a localized swap, not a rewrite.

KILL SWITCH: set AGENT_SYSTEM_DISABLED=1 in the environment, OR create a
file named .KILL_SWITCH in the project root, to immediately halt all new
agent dispatch. Checked at the top of orchestrator_node in graph.py.
Two trigger mechanisms because they suit different situations: the env var
works in any deployment including containers where you can't easily touch
the filesystem; the file works for a local operator who just wants to
`New-Item .KILL_SWITCH` without worrying about env var propagation or
restarting anything.

DISABLE A SINGLE AGENT: set DISABLED_AGENTS="legal_agent,coding_agent"
(comma-separated) to remove specific agents from dispatch without a code
change or redeploy -- useful if one agent starts misbehaving and you need
to pull it immediately while you investigate, without taking the whole
system down.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KILL_SWITCH_FILE = os.path.join(_PROJECT_ROOT, ".KILL_SWITCH")


class ConfigError(Exception):
    """Raised at startup when required configuration is missing. Distinct
    from a plain RuntimeError so callers (e.g. api.py's startup hook) can
    catch it specifically and fail the process cleanly rather than letting
    a missing key surface as a confusing traceback three agent calls deep,
    or worse, an opaque 401 from a vendor API."""
    pass


def validate_startup_config() -> None:
    """Call this once at process startup -- graph.py's __main__ and api.py's
    startup event both do. Never logs the actual value of any secret, only
    which ones are missing by name."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise ConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Set them in .env (local dev) or your secrets manager "
            f"(production) before starting the system."
        )


def is_kill_switch_active() -> bool:
    return os.environ.get("AGENT_SYSTEM_DISABLED", "").strip() == "1" or os.path.exists(KILL_SWITCH_FILE)


def get_disabled_agents() -> set[str]:
    raw = os.environ.get("DISABLED_AGENTS", "")
    return {a.strip() for a in raw.split(",") if a.strip()}
