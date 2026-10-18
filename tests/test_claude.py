"""tarun_common.claude with the Anthropic client faked: no network, no key."""

from datetime import UTC, datetime
from types import SimpleNamespace

import anthropic

try:  # anthropic >= 1.15 ships its HTTP layer as httpx2
    import httpx2 as httpx
except ImportError:  # older SDKs (the apps pin their own)
    import httpx
import pytest

from tarun_common import claude


def model(mid: str, day: int, *, caps: bool = True, efforts=("low", "high"), structured=True):
    capabilities = None
    if caps:
        effort = SimpleNamespace(
            supported=bool(efforts),
            **{lvl: SimpleNamespace(supported=lvl in efforts) for lvl in claude.EFFORT_LEVELS},
        )
        capabilities = SimpleNamespace(
            effort=effort, structured_outputs=SimpleNamespace(supported=structured)
        )
    name = "Claude " + mid.removeprefix("claude-").replace("-", " ").title()
    return SimpleNamespace(
        id=mid,
        display_name=name,
        created_at=datetime(2026, 1, day, tzinfo=UTC),
        capabilities=capabilities,
    )


LISTING = [
    model("claude-haiku-4-5", 1, efforts=()),
    model("claude-haiku-5-5", 5),
    model("claude-sonnet-5-5", 4, structured=False),
    model("claude-opus-5", 2),
    model("claude-opus-5-5", 6),
    model("claude-3-old-model", 9),  # no family: ignored
]


def status_error(cls, code: int):
    request = httpx.Request("GET", "https://api.anthropic.com/v1/models")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


class FakeList:
    """Async and sync iteration over a listing, or raising on the first step."""

    def __init__(self, items, error=None):
        self.items, self.error = items, error

    def __aiter__(self):
        async def gen():
            if self.error:
                raise self.error
            for m in self.items:
                yield m

        return gen()

    def __iter__(self):
        if self.error:
            raise self.error
        return iter(self.items)


@pytest.fixture
def api(monkeypatch):
    """Fake both SDK clients. `api['listing']` / `api['error']` steer the next lookup."""
    state = {"listing": LISTING, "error": None, "calls": 0, "options": None, "closed": 0}

    def models_list(limit):
        state["calls"] += 1
        return FakeList(state["listing"], state["error"])

    class Client:
        def __init__(self, **options):
            state["options"] = options
            self.models = SimpleNamespace(list=models_list)

        def close(self):
            state["closed"] += 1

    class AsyncClient(Client):
        async def close(self):
            state["closed"] += 1

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", AsyncClient)
    claude._cache.clear()
    claude._seen.clear()
    return state


# ── family_of ────────────────────────────────────────────────────────────────


def test_family_of_reads_families_and_legacy_ids():
    assert claude.family_of("opus") == "opus"
    assert claude.family_of("claude-haiku-4-5-20251001") == "haiku"
    assert claude.family_of("gpt-4o") is None
    assert claude.family_of(None) is None
    assert claude.family_of("fable", allowed=("haiku", "opus")) is None


# ── newest per family ────────────────────────────────────────────────────────


async def test_newest_by_family_picks_by_release_date(api):
    newest = await claude.newest_by_family("sk-ant-1")
    assert {f: m.id for f, m in newest.items()} == {
        "haiku": "claude-haiku-5-5",
        "sonnet": "claude-sonnet-5-5",
        "opus": "claude-opus-5-5",
    }
    assert newest["sonnet"].structured_outputs is False
    assert newest["opus"].efforts == ("low", "high")
    assert newest["opus"].display_name == "Claude Opus 5 5"
    assert api["closed"] == 1


async def test_lookups_are_cached_per_key_until_refreshed(api):
    await claude.newest_by_family("sk-ant-1")
    await claude.newest_by_family("sk-ant-1")
    assert api["calls"] == 1
    await claude.newest_by_family("sk-ant-2")  # another org may see other models
    assert api["calls"] == 2
    await claude.newest_by_family("sk-ant-1", refresh=True)
    assert api["calls"] == 3
    await claude.newest_by_family("sk-ant-1", base_url="http://proxy")
    assert api["calls"] == 4
    assert api["options"]["base_url"] == "http://proxy"


async def test_cache_expires(api, monkeypatch):
    await claude.newest_by_family("sk-ant-1")
    now = claude.time.monotonic()
    monkeypatch.setattr(claude.time, "monotonic", lambda: now + claude.RESOLVE_TTL_SECONDS + 1)
    await claude.newest_by_family("sk-ant-1")
    assert api["calls"] == 2


def test_sync_variant_matches(api):
    newest = claude.newest_by_family_sync("sk-ant-1")
    assert newest["haiku"].id == "claude-haiku-5-5"
    assert claude.newest_model_sync("sk-ant-1", "opus") == "claude-opus-5-5"
    assert api["calls"] == 1


async def test_newest_model_falls_back_when_the_lookup_fails(api):
    api["error"] = anthropic.APIConnectionError(
        request=httpx.Request("GET", "https://api.anthropic.com/v1/models")
    )
    assert await claude.newest_model("sk-ant-1", "sonnet") == "claude-sonnet-5-5"
    assert claude.newest_model_sync("sk-ant-1", "haiku") == "claude-haiku-5-5"


async def test_newest_model_for_a_family_the_key_cant_see(api):
    assert await claude.newest_model("sk-ant-1", "fable") == "claude-fable-5-1"


async def test_cached_model_never_calls_the_api(api):
    assert claude.cached_model("sk-ant-1", "opus") is None
    await claude.newest_by_family("sk-ant-1")
    assert claude.cached_model("sk-ant-1", "opus") == "claude-opus-5-5"
    assert claude.cached_model("sk-ant-1", "fable") is None
    assert api["calls"] == 1


# ── capabilities ─────────────────────────────────────────────────────────────


async def test_supports_effort_uses_what_the_api_said(api):
    api["listing"] = [model("claude-haiku-5-5", 1, efforts=()), model("claude-opus-5-5", 1)]
    await claude.newest_by_family("sk-ant-1")
    assert claude.supports_effort("claude-haiku-5-5") is False
    assert claude.supports_effort("claude-opus-5-5") is True
    assert claude.seen("claude-opus-5-5").display_name == "Claude Opus 5 5"


def test_supports_effort_without_capability_info():
    claude._seen.clear()
    assert claude.supports_effort("claude-haiku-4-5-20251001") is False
    assert claude.supports_effort("claude-sonnet-5-5") is True


async def test_missing_capabilities_mean_unknown_not_unsupported(api):
    api["listing"] = [model("claude-sonnet-5-5", 1, caps=False)]
    newest = await claude.newest_by_family("sk-ant-1")
    assert newest["sonnet"].efforts is None
    assert claude.supports_effort("claude-sonnet-5-5") is True


# ── verify ───────────────────────────────────────────────────────────────────


async def test_verify_always_asks_and_returns_the_newest(api):
    await claude.newest_by_family("sk-ant-1")
    m = await claude.verify("sk-ant-1", "haiku")
    assert m.id == "claude-haiku-5-5"
    assert api["calls"] == 2


@pytest.mark.parametrize(
    ("error", "kind", "status"),
    [
        (status_error(anthropic.AuthenticationError, 401), "auth", 422),
        (status_error(anthropic.PermissionDeniedError, 403), "permission", 403),
        (status_error(anthropic.RateLimitError, 429), "rate_limit", 429),
        (status_error(anthropic.InternalServerError, 500), "status", 502),
        (
            anthropic.APIConnectionError(request=httpx.Request("GET", "https://x")),
            "connection",
            502,
        ),
    ],
)
async def test_verify_turns_sdk_errors_into_claude_errors(api, error, kind, status):
    api["error"] = error
    with pytest.raises(claude.ClaudeError) as e:
        await claude.verify("sk-ant-1", "haiku")
    assert (e.value.kind, e.value.status_code) == (kind, status)
    with pytest.raises(claude.ClaudeError):
        claude.verify_sync("sk-ant-1", "haiku")


async def test_verify_a_family_the_key_cant_use(api):
    with pytest.raises(claude.ClaudeError, match="can't use any Claude Fable model") as e:
        await claude.verify("sk-ant-1", "fable")
    assert e.value.kind == "no_model"


# ── requests and errors ──────────────────────────────────────────────────────


def test_fallback_options_only_where_supported():
    assert claude.fallback_options("haiku") == {}
    assert claude.fallback_options("claude-haiku-5-5") == {}
    assert claude.fallback_options("claude-opus-5-5") == {
        "betas": [claude.REFUSAL_FALLBACK_BETA],
        "fallbacks": "default",
    }
    assert claude.fallback_options("sonnet")["fallbacks"] == "default"
    assert claude.fallback_options("gpt-4o") == {}


def test_friendly_error_specifics():
    assert claude.friendly_error(status_error(anthropic.NotFoundError, 404)).kind == "not_found"
    overloaded = claude.friendly_error(status_error(anthropic.APIStatusError, 529))
    assert (overloaded.kind, overloaded.status_code) == ("overloaded", 503)
    assert claude.friendly_error(ValueError("x")).message == "Claude call failed (ValueError)"
    same = claude.ClaudeError("m", 400, "auth")
    assert claude.friendly_error(same) is same


def test_text_of_skips_non_text_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text=" Hello "),
            SimpleNamespace(type="text", text="world. "),
        ]
    )
    assert claude.text_of(response) == "Hello world."


def test_importing_the_module_does_not_import_anthropic():
    import subprocess
    import sys

    code = "import sys, tarun_common.claude; print('anthropic' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
