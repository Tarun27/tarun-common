"""The helper apps use in their own tests."""

import pytest

from tarun_common import claude
from tarun_common.claude_testing import fake_models


@pytest.fixture
def fake(monkeypatch):
    return fake_models(
        monkeypatch,
        [
            ("claude-haiku-5-5", 2.0, "Claude Haiku 5.5"),
            ("claude-haiku-4-5", 1.0, "Claude Haiku 4.5"),
        ],
    )


async def test_lookups_run_through_the_real_cache(fake):
    assert await claude.newest_model("k", "haiku") == "claude-haiku-5-5"
    assert await claude.newest_model("k", "haiku") == "claude-haiku-5-5"
    assert fake.calls == 1
    assert (await claude.verify("k", "haiku")).display_name == "Claude Haiku 5.5"
    assert fake.calls == 2  # verify always asks
    assert claude.newest_model_sync("other", "opus") == "claude-opus-5-5"  # known fallback


async def test_errors_and_listing_can_change_mid_test(fake):
    import anthropic

    fake.error = anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]
    assert await claude.newest_model("k", "haiku") == "claude-haiku-5-5"  # known model
    with pytest.raises(claude.ClaudeError):
        await claude.verify("k", "haiku")
    fake.error = None
    fake.models = [("claude-haiku-6", 3.0, "Claude Haiku 6")]
    assert (await claude.verify("k", "haiku")).id == "claude-haiku-6"


def test_cache_starts_empty_each_test(fake):
    assert claude.cached_model("k", "haiku") is None
