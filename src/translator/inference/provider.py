"""Thin provider abstraction over OpenAI and Anthropic.

v3 deliberately splits work across providers so the translator and the
deviation-extractor don't share a blind spot. Two helpers, one per
provider, both returning a string completion. The CLI picks which to
use based on the model id prefix (``claude-*`` -> Anthropic, anything
else -> OpenAI), but callers can also pin explicitly.

Key design points:

* OpenAI calls go through the **Responses API** (``client.responses.create``).
  This is the recommended path for GPT-5.x reasoning models and exposes
  the ``reasoning.effort`` knob (``low`` / ``medium`` / ``high`` / ``xhigh``).
  Older models (e.g. ``chat-latest``) still work because Responses is a
  superset of Chat Completions.
* Anthropic calls go through the **Messages API**. Claude Opus 4.7
  rejects non-default ``temperature`` / ``top_p`` / ``top_k`` (returns
  HTTP 400), so for known-strict models we omit ``temperature``
  entirely. All other Anthropic models still accept ``temperature``.
* Network calls are intentionally NOT made during ``assemble_prompt`` or
  ``build_window``; only :func:`complete` reaches out, so dry-run flows
  work entirely offline and tests don't need to mock either client.
"""

from __future__ import annotations

import random
import time
from typing import Literal

from translator.config import require_anthropic_key, require_openai_key

Provider = Literal["openai", "anthropic"]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]

# Anthropic model ids (substrings) that reject non-default sampling params.
# Confirmed from the Anthropic May 2026 docs: Opus 4.7 returns HTTP 400 if
# ``temperature`` / ``top_p`` / ``top_k`` are explicitly set. Be conservative
# and skip ``temperature`` for the entire Opus 4.7 family.
_ANTHROPIC_STRICT_SAMPLING: tuple[str, ...] = ("opus-4-7",)


def detect_provider(model: str) -> Provider:
    """Pick a provider from a model id by prefix."""
    if model.lower().startswith(("claude", "anthropic")):
        return "anthropic"
    return "openai"


def complete(
    *,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 4096,
    reasoning_effort: ReasoningEffort | None = None,
    json_schema: dict | None = None,
    provider: Provider | None = None,
) -> str:
    """Send a single completion request and return the text content.

    ``reasoning_effort`` is passed through for OpenAI Responses-API calls
    and ignored for Anthropic. ``temperature`` is honored for Anthropic
    when the model accepts it; OpenAI reasoning models ignore it on the
    server side.

    ``json_schema`` (OpenAI only) enables structured outputs: the model
    is forced to emit JSON conforming to the schema and the SDK guarantees
    parseability. Pass a JSON-schema dict (with at minimum ``type`` /
    ``properties`` / ``required``); it's wrapped in the Responses API
    ``text.format`` envelope automatically. Anthropic doesn't expose an
    equivalent so the parameter is ignored when routed there.
    """
    provider = provider or detect_provider(model)
    if provider == "anthropic":
        return _complete_anthropic(model, prompt, system, temperature, max_tokens)
    return _complete_openai(model, prompt, system, max_tokens, reasoning_effort, json_schema)


def _complete_openai(
    model: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    reasoning_effort: ReasoningEffort | None,
    json_schema: dict | None = None,
) -> str:
    """Call OpenAI via the Responses API.

    The Responses API is a superset of Chat Completions; using it
    uniformly means GPT-5.5 (with reasoning.effort) and legacy
    chat-latest both go through the same code path.
    """
    from openai import OpenAI

    client = OpenAI(api_key=require_openai_key())

    kwargs: dict = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    if system:
        kwargs["instructions"] = system
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if json_schema is not None:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": json_schema.get("name", "response"),
                "schema": json_schema.get("schema", json_schema),
                "strict": json_schema.get("strict", True),
            }
        }

    resp = client.responses.create(**kwargs)
    # The SDK exposes a flattened convenience field that concatenates all
    # text output blocks; fall back to manual aggregation for older SDKs.
    text = getattr(resp, "output_text", None)
    if text is None:
        chunks: list[str] = []
        for item in getattr(resp, "output", []) or []:
            for block in getattr(item, "content", None) or []:
                t = getattr(block, "text", None)
                if t:
                    chunks.append(t)
        text = "".join(chunks)
    return text


_ANTHROPIC_MAX_RETRIES = 6
_ANTHROPIC_BASE_DELAY = 4.0
_ANTHROPIC_MAX_DELAY = 120.0


def _complete_anthropic(
    model: str,
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    import anthropic

    # The SDK's built-in retry caps at 2 attempts, which isn't enough
    # for the sustained overload windows we hit on long Opus 4.7 runs.
    # Use ``max_retries=0`` and drive the backoff loop ourselves so we
    # can log progress and apply our own ceiling.
    client = anthropic.Anthropic(
        api_key=require_anthropic_key(),
        max_retries=0,
    )
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not _anthropic_rejects_temperature(model):
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system

    last_exc: Exception | None = None
    for attempt in range(_ANTHROPIC_MAX_RETRIES + 1):
        try:
            resp = client.messages.create(**kwargs)
            parts: list[str] = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        except (
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ) as exc:
            last_exc = exc
            # Retry on 408/429/5xx (overload, rate limit, transient
            # server errors). Don't retry on 4xx that indicate a real
            # programming bug.
            status = getattr(exc, "status_code", None)
            transient = (
                isinstance(exc, anthropic.APIConnectionError | anthropic.APITimeoutError)
                or status in {408, 409, 425, 429}
                or (status is not None and 500 <= status < 600)
            )
            if not transient or attempt == _ANTHROPIC_MAX_RETRIES:
                raise
            # Exponential backoff with jitter, capped to keep wall
            # time bounded even when the API is hard down.
            delay = min(_ANTHROPIC_BASE_DELAY * (2**attempt), _ANTHROPIC_MAX_DELAY)
            delay = delay * (0.75 + 0.5 * random.random())
            print(
                f"note: anthropic {type(exc).__name__} "
                f"(status={status}); retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{_ANTHROPIC_MAX_RETRIES})"
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _anthropic_rejects_temperature(model: str) -> bool:
    lower = model.lower()
    return any(strict in lower for strict in _ANTHROPIC_STRICT_SAMPLING)


__all__ = ["Provider", "ReasoningEffort", "complete", "detect_provider"]
