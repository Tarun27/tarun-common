"""Claude: which model a family runs on, checking a key, and the request options and errors
every app handles the same way.

Before this module Orange, Mango, SDE Prep, health-tracker and Resume Maker each carried
their own copy of the same logic. What stays in each app is what differs between them:
where the key is stored, which families it offers and how it describes them, its prompts,
its call ledger, and which HTTP status it turns an error into.

The rules, shared by all of them:

- **The user picks a family, never a version.** Each call runs on the family's newest
  model, found with the Models API (bills no tokens) and cached for 6 hours per key, so a
  new release needs no code change. Model access differs between Anthropic orgs, which is
  why the cache is per key (and per base URL).
- **The Models API failing never blocks a call.** The family's known model stands in and
  the call itself reports a bad key.
- **Refusal fallback** (`fallbacks: "default"`, a beta) re-runs a declined request on
  another model inside the same call. Haiku has none and rejects the parameter.
- `anthropic` is imported only when a function that needs it runs, so an app with AI
  switched off never loads a network client (health-tracker's Phase 6 rule).

Nothing here logs or stores a key, a prompt or an answer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
RESOLVE_TTL_SECONDS = 6 * 3600
# Models known to reject `output_config.effort`, for when the Models API hasn't said.
_NO_EFFORT_PREFIXES = ("claude-haiku-4-5",)


@dataclass(frozen=True)
class Family:
    id: str  # what apps store, e.g. "haiku"
    label: str  # "Claude Haiku"
    known_newest: str  # used only when the Models API can't be asked
    refusal_fallback: bool  # accepts `fallbacks: "default"`


FAMILIES: tuple[Family, ...] = (
    Family("haiku", "Claude Haiku", "claude-haiku-5-5", refusal_fallback=False),
    Family("sonnet", "Claude Sonnet", "claude-sonnet-5-5", refusal_fallback=True),
    Family("opus", "Claude Opus", "claude-opus-5-5", refusal_fallback=True),
    Family("fable", "Claude Fable", "claude-fable-5-1", refusal_fallback=True),
)
FAMILIES_BY_ID: dict[str, Family] = {f.id: f for f in FAMILIES}


@dataclass(frozen=True)
class NewestModel:
    """A family's newest model as the Models API describes it."""

    id: str
    display_name: str  # "Claude Haiku 5.5"
    family: str
    created_at: datetime | None
    structured_outputs: bool
    # Effort levels it accepts; None when the API didn't report capabilities.
    efforts: tuple[str, ...] | None


class ClaudeError(Exception):
    """An Anthropic failure in words a person can act on.

    `kind` says what went wrong (auth, permission, not_found, rate_limit, overloaded,
    connection, status, no_model); `status_code` is a sensible HTTP status for it, which
    an app is free to map differently.
    """

    def __init__(self, message: str, status_code: int, kind: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.kind = kind


# (sha256 of key|base_url) -> (monotonic expiry, family -> newest model)
_cache: dict[str, tuple[float, dict[str, NewestModel]]] = {}
# model id -> what the last lookup said about it (display name, capabilities)
_seen: dict[str, NewestModel] = {}


def family_of(stored: str | None, allowed: Iterable[str] = FAMILIES_BY_ID) -> str | None:
    """The family a stored setting names, or None.

    Rows saved before apps stored families hold a full model id (`claude-haiku-4-5`);
    those count as their family, so they move up to its newest model too.
    """
    allowed = set(allowed)
    if stored in allowed:
        return stored
    match = re.match(r"claude-([a-z]+)-", stored or "")
    if match and match.group(1) in allowed:
        return match.group(1)
    return None


def _cache_key(api_key: str, base_url: str | None) -> str:
    return hashlib.sha256(f"{api_key}|{base_url or ''}".encode()).hexdigest()


def _supported(model: Any, *path: str) -> bool:
    node = getattr(model, "capabilities", None)
    for name in path:
        node = getattr(node, name, None)
    return bool(getattr(node, "supported", False))


def _pick(models: Iterable[Any]) -> dict[str, NewestModel]:
    """The newest model (by release date) of each known family."""
    newest: dict[str, Any] = {}
    for m in models:
        family = family_of(m.id)
        if family and (family not in newest or m.created_at > newest[family].created_at):
            newest[family] = m
    out: dict[str, NewestModel] = {}
    for family, m in newest.items():
        has_caps = getattr(m, "capabilities", None) is not None
        efforts: tuple[str, ...] | None = None
        if has_caps:
            efforts = (
                tuple(lvl for lvl in EFFORT_LEVELS if _supported(m, "effort", lvl))
                if _supported(m, "effort")
                else ()
            )
        out[family] = NewestModel(
            id=m.id,
            display_name=m.display_name,
            family=family,
            created_at=m.created_at,
            structured_outputs=_supported(m, "structured_outputs"),
            efforts=efforts,
        )
    return out


def _store(api_key: str, base_url: str | None, newest: dict[str, NewestModel]) -> None:
    _cache[_cache_key(api_key, base_url)] = (time.monotonic() + RESOLVE_TTL_SECONDS, newest)
    for m in newest.values():
        _seen[m.id] = m


def _cached(api_key: str, base_url: str | None) -> dict[str, NewestModel] | None:
    hit = _cache.get(_cache_key(api_key, base_url))
    return hit[1] if hit and hit[0] > time.monotonic() else None


def _client_options(api_key: str, base_url: str | None) -> dict[str, Any]:
    options: dict[str, Any] = {"max_retries": 1, "timeout": 20}
    if api_key:  # empty: the SDK reads ANTHROPIC_API_KEY itself
        options["api_key"] = api_key
    if base_url:
        options["base_url"] = base_url
    return options


# ── which model a family runs on ─────────────────────────────────────────────

# The only two places that talk to Anthropic's Models API. `tarun_common.claude_testing`
# replaces them in app tests, so caching and refresh still run for real.


async def _list_async(api_key: str, base_url: str | None) -> list[Any]:
    import anthropic

    client = anthropic.AsyncAnthropic(**_client_options(api_key, base_url))
    try:
        return [m async for m in client.models.list(limit=1000)]
    finally:
        await client.close()


def _list_sync(api_key: str, base_url: str | None) -> list[Any]:
    import anthropic

    client = anthropic.Anthropic(**_client_options(api_key, base_url))
    try:
        return list(client.models.list(limit=1000))
    finally:
        client.close()


async def newest_by_family(
    api_key: str, *, base_url: str | None = None, refresh: bool = False
) -> dict[str, NewestModel]:
    """Each family's newest model this key can see, cached for 6 hours; `refresh` asks
    Anthropic even when cached. Raises the SDK's errors."""
    if not refresh and (hit := _cached(api_key, base_url)) is not None:
        return hit
    newest = _pick(await _list_async(api_key, base_url))
    _store(api_key, base_url, newest)
    return newest


def newest_by_family_sync(
    api_key: str, *, base_url: str | None = None, refresh: bool = False
) -> dict[str, NewestModel]:
    """`newest_by_family` for apps on the synchronous SDK."""
    if not refresh and (hit := _cached(api_key, base_url)) is not None:
        return hit
    newest = _pick(_list_sync(api_key, base_url))
    _store(api_key, base_url, newest)
    return newest


def _known(family: str) -> str:
    return FAMILIES_BY_ID[family].known_newest


async def newest_model(api_key: str, family: str, *, base_url: str | None = None) -> str:
    """The model to call for `family`: its newest, cached for 6 hours. A failed lookup
    falls back to the family's known model; the call itself then reports a key problem."""
    import anthropic

    try:
        newest = await newest_by_family(api_key, base_url=base_url)
    except anthropic.APIError as e:
        log.warning(
            "Claude model lookup for %s failed (%s); using %s",
            family,
            type(e).__name__,
            _known(family),
        )
        return _known(family)
    return newest[family].id if family in newest else _known(family)


def newest_model_sync(api_key: str, family: str, *, base_url: str | None = None) -> str:
    """`newest_model` for apps on the synchronous SDK."""
    import anthropic

    try:
        newest = newest_by_family_sync(api_key, base_url=base_url)
    except anthropic.APIError as e:
        log.warning(
            "Claude model lookup for %s failed (%s); using %s",
            family,
            type(e).__name__,
            _known(family),
        )
        return _known(family)
    return newest[family].id if family in newest else _known(family)


def cached_model(api_key: str, family: str, *, base_url: str | None = None) -> str | None:
    """What `family` last resolved to for this key, without any API call."""
    hit = _cached(api_key, base_url)
    return hit[family].id if hit and family in hit else None


def seen(model_id: str) -> NewestModel | None:
    """What the last lookup reported for this model (display name, capabilities)."""
    return _seen.get(model_id)


def supports_effort(model_id: str) -> bool:
    """Whether a request to this model may set `output_config.effort`."""
    info = _seen.get(model_id)
    if info is not None and info.efforts is not None:
        return bool(info.efforts)
    return not model_id.startswith(_NO_EFFORT_PREFIXES)


# ── checking a key ───────────────────────────────────────────────────────────


def _no_model(family: str) -> ClaudeError:
    label = FAMILIES_BY_ID[family].label if family in FAMILIES_BY_ID else family
    return ClaudeError(f"This API key can't use any {label} model", 422, "no_model")


async def verify(api_key: str, family: str, *, base_url: str | None = None) -> NewestModel:
    """Check, without billing tokens, that the key works and can use a model in `family`.
    Always asks Anthropic (never the cache), so it is also the live "which version am I
    on" lookup. Raises ClaudeError."""
    try:
        newest = await newest_by_family(api_key, base_url=base_url, refresh=True)
    except Exception as e:
        raise friendly_error(e) from e
    if family not in newest:
        raise _no_model(family)
    return newest[family]


def verify_sync(api_key: str, family: str, *, base_url: str | None = None) -> NewestModel:
    """`verify` for apps on the synchronous SDK."""
    try:
        newest = newest_by_family_sync(api_key, base_url=base_url, refresh=True)
    except Exception as e:
        raise friendly_error(e) from e
    if family not in newest:
        raise _no_model(family)
    return newest[family]


# ── requests and errors ──────────────────────────────────────────────────────


def fallback_options(model_or_family: str) -> dict[str, Any]:
    """Extra arguments opting a request into server-side refusal fallback, or {} for a
    family without it (Haiku). They need `client.beta.messages.create`."""
    family = family_of(model_or_family)
    if family and FAMILIES_BY_ID[family].refusal_fallback:
        return {"betas": [REFUSAL_FALLBACK_BETA], "fallbacks": "default"}
    return {}


def friendly_error(exc: BaseException) -> ClaudeError:
    """An Anthropic SDK error as a ClaudeError a person can act on. Most specific first:
    a bad key, a missing model and a rate limit are different problems, and telling them
    apart is the difference between "try again" and "fix your settings"."""
    if isinstance(exc, ClaudeError):
        return exc
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return ClaudeError("Anthropic rejected this API key", 422, "auth")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return ClaudeError(
            "This API key isn't allowed to do that (check its workspace)", 403, "permission"
        )
    if isinstance(exc, anthropic.NotFoundError):
        return ClaudeError("That model isn't available to this API key", 422, "not_found")
    if isinstance(exc, anthropic.RateLimitError):
        return ClaudeError("Anthropic rate limit hit; try again in a minute", 429, "rate_limit")
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529:
        return ClaudeError("Anthropic is overloaded; try again shortly", 503, "overloaded")
    if isinstance(exc, anthropic.APIStatusError):
        return ClaudeError(f"Claude API error ({exc.status_code})", 502, "status")
    if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
        return ClaudeError("Couldn't reach the Claude API; check the network", 502, "connection")
    return ClaudeError(f"Claude call failed ({type(exc).__name__})", 502, "status")


def text_of(response: Any) -> str:
    """The text blocks of a Messages response, joined. The content can also carry
    thinking and fallback blocks, which are skipped."""
    return "".join(b.text for b in response.content if b.type == "text").strip()
