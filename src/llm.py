"""
Thin wrapper over the Anthropic SDK.

Two jobs, kept deliberately small:
  - `parse_structured` - get a validated pydantic object back, not prose.
  - `complete_text`    - get prose, for the one step that genuinely needs it.

Everything model-specific lives here so that swapping models (or providers)
later is a single-file change, and so the eval harness has one place to read
token usage from.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from src import config

T = TypeVar("T", bound=BaseModel)


class LLMRefusal(RuntimeError):
    """The model declined the request. Surfaced rather than retried blindly."""


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """One shared client. The SDK is thread-safe and pools connections."""
    return anthropic.Anthropic(api_key=config.require("ANTHROPIC_API_KEY"))


def _usage_dict(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _system_blocks(system: str, cache: bool) -> list[dict]:
    """
    Mark the system prompt as cacheable.

    Extraction resends the same long instruction block once per abstract, so
    caching it is the single biggest cost lever in the project. If
    `cache_read_input_tokens` stays at zero across a run, something upstream is
    varying the prompt text and the cache is silently doing nothing.
    """
    block: dict = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def parse_structured(
    system: str,
    user: str,
    schema: type[T],
    *,
    model: str | None = None,
    max_tokens: int = 8000,
    cache_system: bool = True,
) -> tuple[T, dict[str, int]]:
    """
    Ask for one validated instance of `schema`.

    Uses `messages.parse`, which constrains the response to the schema and
    validates it - so a malformed extraction fails here rather than becoming a
    subtly wrong finding later.

    Effort is left at the default here. `parse()` does accept `output_config`
    alongside `output_format`, so turning effort down for bulk extraction is
    possible - but it trades extraction quality, which is the one thing this
    project measures. That makes it a change to justify with eval numbers
    rather than a default to assume, and this is the single place to make it.
    """
    response = get_client().messages.parse(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=_system_blocks(system, cache_system),
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )

    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusal(f"model declined: {getattr(details, 'category', 'unknown')}")

    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            f"structured parse returned nothing (stop_reason="
            f"{getattr(response, 'stop_reason', None)})"
        )
    return parsed, _usage_dict(response)


def complete_text(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 4000,
    effort: str = "medium",
    cache_system: bool = True,
) -> tuple[str, dict[str, int]]:
    """Plain prose, for the final write-up step where the content is already fixed."""
    response = get_client().messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=_system_blocks(system, cache_system),
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
    )

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusal(f"model declined: {getattr(details, 'category', 'unknown')}")

    text = "".join(b.text for b in response.content if b.type == "text")
    return text.strip(), _usage_dict(response)
