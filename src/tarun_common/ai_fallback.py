"""Free AI providers with fallback: Groq → OpenRouter → Gemini, one prompt in, text out.

IRA built this chain and Orange copied it; this is the one copy. Each provider gets up to
three attempts on a 429 (2 s, then 4 s) before the next is tried; Gemini sits last because
its free tier rate-limits hardest. Every failure returns None rather than raising, because
the callers all have something useful to show without AI.

What stays in each app: where keys and models come from (.env, an admin setting), any
provider tried before these (Orange's per-user Claude), and its call ledger, which it
plugs in through `on_call`.

A provider that answers 4xx other than 429 is misconfigured, not busy, so it is logged as
an ERROR naming the setting to fix. Groq retired `llama-3.3-70b-versatile` in 2026 and both
apps fell through to OpenRouter for weeks with nothing louder than a warning.

All three speak plain HTTP (httpx, imported when a call is made). Groq and OpenRouter are
OpenAI-compatible and share a request shape. Nothing here logs a key, a prompt or an answer.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

ORDER = ("groq", "openrouter", "gemini")
DEFAULT_MODELS: dict[str, str] = {
    # Checked against Groq's model list 2026-10-18: valid JSON for short tasks in <1 s.
    "groq": "openai/gpt-oss-120b",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "gemini": "gemini-2.0-flash",
}
_SETTING = {"groq": "GROQ", "openrouter": "OPENROUTER", "gemini": "GEMINI"}

_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}
ATTEMPTS = 3


@dataclass(frozen=True)
class Provider:
    name: str  # "groq" | "openrouter" | "gemini"
    api_key: str
    model: str
    headers: Mapping[str, str] = field(default_factory=dict)  # e.g. OpenRouter attribution


@dataclass(frozen=True)
class CallRecord:
    """One HTTP attempt, for an app's call ledger. No prompt, no answer."""

    provider: str
    model: str
    status_code: int | None  # None: no response (network error, timeout)
    latency_ms: int
    ok: bool
    error: str | None


OnCall = Callable[[CallRecord], Awaitable[None] | None]


def providers_from(
    *,
    groq_key: str = "",
    groq_model: str | None = None,
    openrouter_key: str = "",
    openrouter_model: str | None = None,
    gemini_key: str = "",
    gemini_model: str | None = None,
    app_name: str | None = None,
    app_url: str | None = None,
) -> list[Provider]:
    """The providers that have a key, in fallback order. `app_name` / `app_url` are sent to
    OpenRouter as its attribution headers (X-Title, HTTP-Referer)."""
    attribution: dict[str, str] = {}
    if app_url:
        attribution["HTTP-Referer"] = app_url
    if app_name:
        attribution["X-Title"] = app_name
    given = {
        "groq": (groq_key, groq_model, {}),
        "openrouter": (openrouter_key, openrouter_model, attribution),
        "gemini": (gemini_key, gemini_model, {}),
    }
    return [
        Provider(name, key, model or DEFAULT_MODELS[name], headers)
        for name in ORDER
        for key, model, headers in [given[name]]
        if key
    ]


def _make_client(timeout: float) -> Any:
    import httpx

    return httpx.AsyncClient(timeout=timeout)


def _request(
    p: Provider, prompt: str, max_tokens: int, temperature: float
) -> tuple[str, dict[str, str], dict[str, Any]]:
    if p.name == "gemini":
        url = _URLS["gemini"].format(model=p.model) + f"?key={p.api_key}"
        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        return url, {"Content-Type": "application/json"}, body
    headers = {"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"}
    headers.update(p.headers)
    body = {
        "model": p.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if p.name == "groq" and p.model.startswith("openai/gpt-oss"):
        # These reason before answering, inside max_tokens; low effort halves the
        # tokens a short task spends and keeps small caps from cutting the answer off.
        body["reasoning_effort"] = "low"
    return _URLS[p.name], headers, body


def _text(p: Provider, data: Any) -> str | None:
    if p.name == "gemini":
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        text = data["choices"][0]["message"]["content"]
    return text.strip() if isinstance(text, str) and text.strip() else None


async def _report(on_call: OnCall | None, record: CallRecord) -> None:
    if on_call is None:
        return
    try:
        result = on_call(record)
        if inspect.isawaitable(result):
            await result
    except Exception:  # a ledger failure must never fail the call
        log.debug("on_call hook failed", exc_info=True)


async def complete(
    provider: Provider,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: float = 30.0,
    on_call: OnCall | None = None,
) -> str | None:
    """One provider's answer, or None. Retries a 429 up to ATTEMPTS times."""
    url, headers, body = _request(provider, prompt, max_tokens, temperature)
    setting = _SETTING[provider.name]
    for attempt in range(ATTEMPTS):
        start = time.monotonic()
        status: int | None = None
        try:
            async with _make_client(timeout) as client:
                r = await client.post(url, headers=headers, json=body)
            status = r.status_code
            ms = int((time.monotonic() - start) * 1000)
            if status == 429:
                await _report(
                    on_call,
                    CallRecord(provider.name, provider.model, status, ms, False, "rate limited"),
                )
                if attempt < ATTEMPTS - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                log.warning("%s: rate limit exhausted", provider.name)
                return None
            if status >= 400:
                await _report(
                    on_call,
                    CallRecord(provider.name, provider.model, status, ms, False, f"HTTP {status}"),
                )
                if status < 500:
                    log.error(
                        "%s answered %s for model %r: misconfigured, not busy. Check "
                        "%s_API_KEY and %s_MODEL",
                        provider.name,
                        status,
                        provider.model,
                        setting,
                        setting,
                    )
                else:
                    log.warning("%s answered %s", provider.name, status)
                return None
            text = _text(provider, r.json())
            await _report(
                on_call,
                CallRecord(
                    provider.name,
                    provider.model,
                    status,
                    ms,
                    text is not None,
                    None if text else "empty answer",
                ),
            )
            if text is None:
                log.warning(
                    "%s returned no text (max_tokens=%d too small?)", provider.name, max_tokens
                )
            return text
        except Exception as e:
            ms = int((time.monotonic() - start) * 1000)
            await _report(
                on_call,
                CallRecord(provider.name, provider.model, status, ms, False, type(e).__name__),
            )
            log.warning("%s failed: %s", provider.name, type(e).__name__)
            return None
    return None


async def generate(
    providers: Iterable[Provider],
    prompt: str,
    *,
    only: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: float = 30.0,
    on_call: OnCall | None = None,
) -> str | None:
    """The first answer down the chain, or None when every provider failed.

    `only` restricts the chain to one provider (an app's "AI provider" setting); None or
    "auto" uses them all, in order.
    """
    chain = [p for p in providers if not only or only == "auto" or p.name == only]
    if only and only != "auto" and not chain:
        log.warning("AI provider %r has no key configured", only)
    for i, p in enumerate(chain):
        text = await complete(
            p,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            on_call=on_call,
        )
        if text:
            if i:
                log.info("AI fallback: succeeded with %s", p.name)
            return text
    if chain:
        log.error("All AI providers failed (%s)", ", ".join(p.name for p in chain))
    return None
