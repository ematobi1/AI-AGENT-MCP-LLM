"""
Provider abstraction layer (see architecture doc §3).
Every agent calls a model through this interface instead of hitting a
vendor SDK directly, so you can swap models per agent without touching
agent logic, and get a consistent audit trail of which model produced
which output.

Fill in ANTHROPIC_API_KEY / OPENAI_API_KEY etc. in your .env file.

Phase 7 addition: every AgentResponse now carries token counts and a
computed cost_usd, using MODEL_PRICING below. Pricing changes over time --
verify current rates at https://docs.claude.com (Anthropic) and
https://platform.openai.com/docs/pricing (OpenAI) periodically and update
MODEL_PRICING if a rate drifts. An unrecognized model logs token counts
but cost_usd stays None rather than silently guessing a wrong number.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import json
import os
import time


# USD per 1M tokens: (input_rate, output_rate). Confirmed against current
# vendor pricing pages as of Aug 2026 -- re-verify periodically, pricing
# is not static.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # "gpt-5" was OpenAI's original Aug 2025 rate ($0.625/$5.00 per MTok);
    # OpenAI has since delisted it from their main pricing page in favor of
    # the 5.4/5.5 family, but the model ID may still resolve on an existing
    # account at its original rate. Verify against your actual invoice/the
    # current pricing page if this stops matching what you're billed.
    "gpt-5": (0.625, 5.00),
}


def calculate_cost_usd(model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    if model not in MODEL_PRICING or input_tokens is None or output_tokens is None:
        return None
    input_rate, output_rate = MODEL_PRICING[model]
    return round((input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate, 6)


@dataclass
class AgentResponse:
    provider: str
    model: str
    content: str
    raw: Any
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


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

        input_tokens = getattr(resp.usage, "input_tokens", None) if hasattr(resp, "usage") else None
        output_tokens = getattr(resp.usage, "output_tokens", None) if hasattr(resp, "usage") else None

        return AgentResponse(
            provider=self.provider_name, model=self.model, content=text,
            raw=resp, latency_ms=(time.time() - start) * 1000,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost_usd(self.model, input_tokens, output_tokens),
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

        input_tokens = getattr(resp.usage, "prompt_tokens", None) if hasattr(resp, "usage") else None
        output_tokens = getattr(resp.usage, "completion_tokens", None) if hasattr(resp, "usage") else None

        return AgentResponse(
            provider=self.provider_name, model=self.model, content=text,
            raw=resp, latency_ms=(time.time() - start) * 1000,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost_usd(self.model, input_tokens, output_tokens),
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
