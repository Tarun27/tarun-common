# tarun-common

Small helpers shared by my apps, so each piece of logic lives in one place instead of a
copy per project. Each module stands alone.

| module | what | used by |
|---|---|---|
| `tarun_common.claude` | Claude model families → newest model, key checks, refusal fallback, friendly errors | Orange, Mango, SDE Prep, health-tracker, Resume Maker |
| `tarun_common.secrets` | Fernet secrets at rest: key checks, clear errors, Resume Maker's key derivation | Orange, Mango, SDE Prep, Resume Maker |
| `tarun_common.ai_fallback` | Free providers with fallback: Groq → OpenRouter → Gemini | Orange, IRA |
| `tarun_common.backup` | Admin backups: download a dump, load one (or a nightly snapshot) into a validation DB read-only, switch back | Orange, SDE Prep, IRA |

Which apps use which module, and why each piece is shared, is tracked in
[shared-stack](https://github.com/Tarun27/shared-stack) (private) — the map across all apps.

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
dependencies = ["tarun-common @ git+https://github.com/Tarun27/tarun-common@v0.5.0"]
```

```text
# requirements.txt
tarun-common @ git+https://github.com/Tarun27/tarun-common@v0.5.0
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

## `tarun_common.secrets`

Plain Fernet, byte for byte what the apps already stored, so existing ciphertexts keep
decrypting. Each app keeps its own policy (which env var, and whether a missing key
disables a feature, is generated, or is an error).

```python
from tarun_common.secrets import SecretBox, SecretKeyError   # not `import secrets`: stdlib

box = SecretBox(settings.credential_key, name="CREDENTIAL_KEY")  # SecretKeyError if missing/bad
token = box.encrypt("sk-ant-…")
box.decrypt(token)            # SecretKeyError if the key changed or the value is corrupt
box.decrypt_or_none(token)    # None instead, for callers that fall back
box.encrypt_bytes(b"…") / box.decrypt_bytes(b"…")
```

`generate_key()` makes a new key; `derive_key(secret, purpose)` is Resume Maker's
`urlsafe_b64(sha256("<purpose>:<secret>"))` scheme, kept for its stored values. Needs
`cryptography` (every app already has it; extra `secrets`), imported only when a box is
made.

## `tarun_common.ai_fallback`

```python
from tarun_common import ai_fallback

providers = ai_fallback.providers_from(
    groq_key=s.groq_api_key, groq_model=s.groq_model,          # models default to DEFAULT_MODELS
    openrouter_key=s.openrouter_api_key, openrouter_model=s.openrouter_model,
    gemini_key=s.gemini_api_key, gemini_model=s.gemini_model,
    app_name="Orange", app_url="https://orange.local",          # OpenRouter attribution
)
text = await ai_fallback.generate(providers, prompt, max_tokens=512, only=None, on_call=ledger)
```

Providers without a key are skipped. A 429 is retried (2 s, 4 s) before the next provider;
any other failure moves on at once. A 4xx other than 429 means misconfigured, so it is
logged as an ERROR naming `<PROVIDER>_API_KEY` / `<PROVIDER>_MODEL`. Returns None when all
fail, never raises. `only="gemini"` uses one provider (`None`/`"auto"`: all).
`on_call(CallRecord)` (sync or async) sees every HTTP attempt — provider, model, status,
latency, ok — for an app's ledger; its own failures are swallowed. Groq's `gpt-oss`
models are sent `reasoning_effort: "low"`. Needs `httpx` (extra `ai`), imported per call.

## `tarun_common.backup`

The admin page's backup flow, which Orange, SDE Prep and IRA each had a copy of: download
a dump of the live database, load a dump into a separate **validation** database and
browse the app read-only against it, switch back. A dump is never loaded over the live
database from the app — that is disaster recovery, done outside it.

```python
from tarun_common import backup

validation = backup.Validation(
    settings.database_url,
    settings.backup_test_url,             # empty → validation mode is off
    on_switch=db_manager.use,             # app hook: point sessions at "primary"/"validation"
)
app.add_middleware(backup.ReadOnlyMiddleware, validation=validation,
                   allow=("/api/admin/backup/", "/api/auth/"))
app.include_router(
    backup.router(validation, filename_prefix="orange",
                  snapshots_root="~/Backups/local-apps",   # optional: nightly snapshots
                  dependencies=[Depends(require_admin)]),
    prefix="/api/admin/backup",
)
```

| endpoint | |
|---|---|
| `GET /status` | mode, read-only, what is loaded, whether snapshots are configured |
| `GET /export` | streamed plain-SQL `pg_dump` of the live database (no owners/grants) |
| `GET /snapshots` | nightly dumps of this database, newest first |
| `POST /restore-snapshot` `{"stamp"}` | load a nightly snapshot into validation |
| `POST /restore-upload` (multipart `file`) | load an uploaded dump (plain SQL or custom format) |
| `POST /restore-path` `{"path"}` | load a dump on the server's disk; `restore_roots=[…]` confines it |
| `POST /switch-primary` | back to the live database |

Loading runs `on_switch("primary")` first, replaces the validation database's `public`
schema (plain SQL through psql, custom format through pg_restore, both stopping at the
first error, owners and grants skipped), then `on_switch("validation")`. A failed load
leaves the app on primary. `snapshots_root` is a folder laid out as
`<root>/db/<YYYY-MM-DD_HHMMSS>/<server>/<database>.dump`.

| function | |
|---|---|
| `pg_target(url)` | SQLAlchemy/libpq URL → `PgTarget(dbname, args, env)`; password only in `env` |
| `dump_chunks(url)` | streaming plain-SQL dump; raises `BackupError` at the end if pg_dump failed |
| `restore(target_url, source)` | replace `public` from a path, bytes or file object |
| `snapshots(root, dbname)` | `[Snapshot(stamp, path, bytes, created_at)]`, newest first |
| `Validation(...)` / `ReadOnlyMiddleware` / `router(...)` | as above |

Needs `pg_dump`, `pg_restore` and `psql` on the PATH, and `fastapi` + `python-multipart`
for the router (`tarun-common[backup]`). Limitation: replacing `public` drops extensions
installed there, and a non-superuser role can't recreate a superuser-only one (pgvector),
so an app whose schema needs one (Mango) can't use validation loads as they are.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Release: bump `version` in `pyproject.toml` and `__init__.py`, commit, tag `vX.Y.Z`, push
the tag; then move each app's pin when it's ready.
