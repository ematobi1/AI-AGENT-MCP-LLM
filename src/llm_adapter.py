"""
Provider abstraction layer (see architecture doc §3).
Every agent calls a model through this interface instead of hitting a
vendor SDK directly, so you can swap models per agent without touching
agent logic, and get a consistent audit trail of which model produced
which output.

Fill in ANTHROPIC_API_KEY / OPENAI_API_KEY etc. in your .env file.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
import os
import time


@dataclass
class AgentResponse:
    provider: str
    model: str
    content: str
    raw: Any
    latency_ms: float


class LLMAdapter:
    """Base interface every provider adapter implements."""

    provider_name: str = "base"

    def complete(self, system_prompt: str, user_content: str,
                 max_tokens: int = 1024) -> AgentResponse:
        raise NotImplementedError


class AnthropicAdapter(LLMAdapter):
    provider_name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        from anthropic import Anthropic  # pip install anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system_prompt: str, user_content: str,
                 max_tokens: int = 1024) -> AgentResponse:
        start = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return AgentResponse(
            provider=self.provider_name, model=self.model, content=text,
            raw=resp, latency_ms=(time.time() - start) * 1000,
        )


class OpenAIAdapter(LLMAdapter):
    provider_name = "openai"

    def __init__(self, model: str = "gpt-5"):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, system_prompt: str, user_content: str,
                 max_tokens: int = 1024) -> AgentResponse:
        start = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = resp.choices[0].message.content or ""
        return AgentResponse(
            provider=self.provider_name, model=self.model, content=text,
            raw=resp, latency_ms=(time.time() - start) * 1000,
        )


# --- Registry: map each agent to a provider/model (architecture doc §3.1) ---
ADAPTER_REGISTRY: dict[str, type[LLMAdapter]] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
}


def get_adapter(provider: str, model: str | None = None) -> LLMAdapter:
    cls = ADAPTER_REGISTRY[provider]
    return cls(model=model) if model else cls()


def parse_json_response(text: str) -> dict:
    """Every agent is instructed to return JSON matching a schema.
    Strip markdown fences defensively before parsing."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"status": "blocked", "error": "unparseable_response", "raw": text}
