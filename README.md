# tarun-common

Small helpers shared by my apps, so each piece of logic lives in one place instead of a
copy per project. Each module stands alone.

| module | what | used by |
|---|---|---|
| `tarun_common.claude` | Claude model families → newest model, key checks, refusal fallback, friendly errors | Orange, Mango, SDE Prep, health-tracker, Resume Maker |

**Library, not a service.** These are thin wrappers; as a service they'd add a network
hop, a port and a process that can be down. (Heavy shared things are services instead,
e.g. [transcriber](https://github.com/Tarun27/transcriber).)

**Nothing secret lives here, and nothing is stored.** Keys stay in each app, encrypted
where that app keeps them; functions take a key and return. Nothing logs a key, a prompt
or an answer.

## Install

Pin a tag, so a change here never reaches an app until it opts in:

```toml
# pyproject.toml
dependencies = ["tarun-common @ git+https://github.com/Tarun27/tarun-common@v0.2.1"]
```

```text
# requirements.txt
tarun-common @ git+https://github.com/Tarun27/tarun-common@v0.2.1
```

Python 3.11+. No required dependencies: `anthropic` is the app's own dependency and is
imported only when a function needs it.

## `tarun_common.claude`

The user picks a **family** (`haiku`, `sonnet`, `opus`, `fable`), never a version. Each
call runs on the family's newest model, looked up with the Models API (bills no tokens)
and cached for 6 hours per key, so a new release needs no code change.

```python
from tarun_common import claude

family = claude.family_of(stored_setting) or "haiku"  # legacy full ids map to a family
model = await claude.newest_model(api_key, family)  # cached; never raises on lookup
extra = claude.fallback_options(model)  # refusal fallback; {} on Haiku
create = client.beta.messages.create if extra else client.messages.create
response = await create(model=model, max_tokens=16000, messages=[...], **extra)
text = claude.text_of(response)

info = await claude.verify(api_key, family)  # Save in Settings: live, no tokens
info.id, info.display_name, info.efforts, info.structured_outputs
```

| function | |
|---|---|
| `family_of(stored, allowed=…)` | family a stored value names (a family, or a legacy `claude-<family>-…` id); `None` if neither |
| `newest_by_family(key, refresh=False, base_url=None)` | `{family: NewestModel}` for every family the key can see; raises SDK errors |
| `newest_model(key, family)` | model id to call; falls back to the family's known model if the lookup fails |
| `cached_model(key, family)` | last resolved id, no API call; `None` if not cached |
| `verify(key, family)` | always asks; returns `NewestModel` or raises `ClaudeError` |
| `seen(model_id)` / `supports_effort(model_id)` | what the last lookup said about a model |
| `fallback_options(model_or_family)` | `{"betas": […], "fallbacks": "default"}` or `{}`; needs `client.beta.messages.create` |
| `friendly_error(exc)` | SDK exception → `ClaudeError(message, status_code, kind)` |
| `text_of(response)` | the response's text blocks, joined |

Every lookup function has a `_sync` twin (`newest_by_family_sync`, `newest_model_sync`,
`verify_sync`) for apps on the synchronous SDK.

`ClaudeError.kind` is one of `auth`, `permission`, `not_found`, `rate_limit`,
`overloaded`, `connection`, `status`, `no_model`; `status_code` is a sensible default that
an app may map to its own.

### Testing an app that uses it

`tarun_common.claude_testing.fake_models(monkeypatch, [(id, released, display_name), …])`
replaces only the Models API listing; the cache, `refresh`, the known-model fallback and
`verify` run for real, and the cache starts empty each test. It returns a `FakeModels`
whose `calls` counts lookups and whose `models` / `error` can change mid-test.

```python
@pytest.fixture(autouse=True)
def models(monkeypatch):
    return fake_models(monkeypatch, [("claude-haiku-5-5", 2.0, "Claude Haiku 5.5")])
```

What stays in each app: where the key is stored, which families it offers and how it
describes them, prompts, the call ledger, and per-feature switches.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Release: bump `version` in `pyproject.toml` and `__init__.py`, commit, tag `vX.Y.Z`, push
the tag; then move each app's pin when it's ready.
