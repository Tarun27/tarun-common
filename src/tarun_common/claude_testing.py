"""Test helper for apps using `tarun_common.claude`: fake the Models API listing.

    from tarun_common.claude_testing import fake_models

    @pytest.fixture(autouse=True)
    def models(monkeypatch):
        return fake_models(monkeypatch, [("claude-haiku-5-5", 2.0, "Claude Haiku 5.5")])

Only the listing is replaced; the 6-hour cache, `refresh`, the known-model fallback and
`verify` run for real, and the cache starts empty for each test. The returned `FakeModels`
counts lookups, and its `models` / `error` can be changed mid-test (a list to return, or
an exception to raise, e.g. an `anthropic.AuthenticationError`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from tarun_common import claude

# (id, released — a datetime or a POSIX timestamp, display name)
Listed = tuple[str, "float | datetime", str]


def model(model_id: str, released: float | datetime, display_name: str) -> Any:
    """One Models API entry, without capability info (as some SDK versions return)."""
    if not isinstance(released, datetime):
        released = datetime.fromtimestamp(released, UTC)
    return SimpleNamespace(
        id=model_id, display_name=display_name, created_at=released, capabilities=None
    )


class FakeModels:
    def __init__(self, models: Sequence[Listed | Any]) -> None:
        self.models: list[Any] = list(models)
        self.error: BaseException | None = None
        self.calls = 0

    def _listing(self) -> list[Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [model(*m) if isinstance(m, tuple) else m for m in self.models]


def fake_models(monkeypatch: Any, models: Sequence[Listed | Any]) -> FakeModels:
    """Replace the Models API for the rest of the test (pytest's `monkeypatch`)."""
    fake = FakeModels(models)

    async def listing_async(api_key: str, base_url: str | None) -> list[Any]:
        return fake._listing()

    def listing_sync(api_key: str, base_url: str | None) -> list[Any]:
        return fake._listing()

    monkeypatch.setattr(claude, "_list_async", listing_async)
    monkeypatch.setattr(claude, "_list_sync", listing_sync)
    monkeypatch.setattr(claude, "_cache", {})
    monkeypatch.setattr(claude, "_seen", {})
    return fake
