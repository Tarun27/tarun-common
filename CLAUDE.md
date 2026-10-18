# tarun-common — Claude Notes

Shared Python helpers for the apps in `../` (Orange, Mango, SDE Prep, health-tracker,
Resume Maker…). Read [README.md](./README.md): it is the contract the apps code against.

- **Apps pin a tag** (`@v0.1.0`). Never move or delete a pushed tag; release a new one.
  A breaking change needs a new minor version and every app moved on purpose.
- **Public repo.** Nothing secret, nothing personal, no app-specific behaviour. Keys,
  storage, prompts, ledgers and per-app labels stay in the apps.
- **No required dependencies, lazy imports.** `anthropic` is imported inside the
  functions that use it (health-tracker must not load a network client with AI off; a
  test enforces this). Keep it that way for anything added.
- **Library, not a service.** Heavy shared things (models, GPU) are services, like
  `../transcriber`.
- Python 3.11+ (Resume Maker runs 3.11). `uv run pytest`, `uv run ruff check .`.
