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
from typing import Any, Optional
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
                 max_tokens: int = 1024, tools: Optional[list[dict]] = None,
                 tool_choice: Optional[dict] = None) -> AgentResponse:
        raise NotImplementedError


class AnthropicAdapter(LLMAdapter):
    provider_name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        from anthropic import Anthropic  # pip install anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system_prompt: str, user_content: str,
                 max_tokens: int = 1024, tools: Optional[list[dict]] = None,
                 tool_choice: Optional[dict] = None) -> AgentResponse:
        """tools: e.g. [{"type": "web_search_20250305", "name": "web_search"}]
        to enable Anthropic's server-side web search. By default Claude decides
        for itself whether a search is worth doing, which means it can (and
        will) skip search entirely if it feels confident from training data --
        pass tool_choice={"type": "any"} to force at least one real tool call
        instead of trusting the model's judgment on when to search."""
        start = time.time()
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        resp = self.client.messages.create(**kwargs)
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
                 max_tokens: int = 1024, tools: Optional[list[dict]] = None,
                 tool_choice: Optional[dict] = None) -> AgentResponse:
        start = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
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
    Strips markdown fences (leading AND trailing -- models sometimes add a
    stray closing fence with no matching opening one) and, if that still
    doesn't parse, falls back to extracting the first balanced {...} block
    from the text rather than giving up on any surrounding junk."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1:] if first_newline != -1 else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return {"status": "blocked", "error": "unparseable_response", "raw": text}
