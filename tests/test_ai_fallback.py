"""tarun_common.ai_fallback with the providers faked through httpx's MockTransport."""

import json
import logging
import subprocess
import sys

import httpx
import pytest

from tarun_common import ai_fallback as ai


@pytest.fixture
def server(monkeypatch):
    """Every request goes to `state['handler'](request)`; requests are collected."""
    state = {"requests": [], "handler": None, "sleeps": []}

    def dispatch(request):
        state["requests"].append(request)
        return state["handler"](request)

    monkeypatch.setattr(
        ai,
        "_make_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(dispatch), timeout=timeout),
    )

    async def no_sleep(s):
        state["sleeps"].append(s)

    monkeypatch.setattr(ai.asyncio, "sleep", no_sleep)
    return state


def chat(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def gemini_answer(text):
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


ALL = ai.providers_from(
    groq_key="g",
    openrouter_key="o",
    gemini_key="m",
    app_name="Orange",
    app_url="https://orange.local",
)


def test_providers_from_keeps_order_and_skips_missing_keys():
    assert [p.name for p in ALL] == ["groq", "openrouter", "gemini"]
    assert ALL[0].model == "openai/gpt-oss-120b"
    assert ALL[1].headers == {"HTTP-Referer": "https://orange.local", "X-Title": "Orange"}
    only = ai.providers_from(openrouter_key="o", openrouter_model="x/y")
    assert [(p.name, p.model) for p in only] == [("openrouter", "x/y")]


async def test_groq_request_shape(server):
    server["handler"] = lambda r: chat("  ready ")
    assert await ai.complete(ALL[0], "hi", max_tokens=100, temperature=0.1) == "ready"
    req = server["requests"][0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer g"
    assert body == {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "temperature": 0.1,
        "reasoning_effort": "low",
    }


async def test_openrouter_gets_attribution_and_no_reasoning_flag(server):
    server["handler"] = lambda r: chat("ok")
    await ai.complete(ALL[1], "hi")
    req = server["requests"][0]
    assert req.headers["x-title"] == "Orange"
    assert "reasoning_effort" not in json.loads(req.content)


async def test_gemini_request_shape(server):
    server["handler"] = lambda r: gemini_answer("ok")
    assert await ai.complete(ALL[2], "hi", max_tokens=50) == "ok"
    req = server["requests"][0]
    assert req.url.path.endswith("/models/gemini-2.0-flash:generateContent")
    assert req.url.params["key"] == "m"
    assert json.loads(req.content)["generationConfig"]["maxOutputTokens"] == 50


async def test_rate_limits_are_retried_with_backoff(server):
    answers = iter([httpx.Response(429), httpx.Response(429), chat("third time")])
    server["handler"] = lambda r: next(answers)
    assert await ai.complete(ALL[0], "hi") == "third time"
    assert server["sleeps"] == [2, 4]


async def test_rate_limit_exhausted_falls_through(server):
    def handler(r):
        return httpx.Response(429) if r.url.host == "api.groq.com" else chat("from openrouter")

    server["handler"] = handler
    assert await ai.generate(ALL, "hi") == "from openrouter"
    assert [r.url.host for r in server["requests"]] == ["api.groq.com"] * 3 + ["openrouter.ai"]


async def test_a_misconfigured_model_is_an_error_and_not_retried(server, caplog):
    def handler(r):
        return httpx.Response(404) if r.url.host == "api.groq.com" else chat("ok")

    server["handler"] = handler
    with caplog.at_level(logging.INFO):
        assert await ai.generate(ALL, "hi") == "ok"
    assert sum(1 for r in server["requests"] if r.url.host == "api.groq.com") == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("GROQ_MODEL" in m and "404" in m for m in errors)
    assert any("succeeded with openrouter" in r.getMessage() for r in caplog.records)


async def test_everything_failing_returns_none(server, caplog):
    def handler(r):
        raise httpx.ConnectError("down", request=r)

    server["handler"] = handler
    assert await ai.generate(ALL, "hi") is None
    assert any("All AI providers failed" in r.getMessage() for r in caplog.records)


async def test_only_restricts_the_chain(server):
    server["handler"] = lambda r: gemini_answer("gemini only")
    assert await ai.generate(ALL, "hi", only="gemini") == "gemini only"
    assert len(server["requests"]) == 1
    assert await ai.generate(ALL, "hi", only="auto") is not None


async def test_an_empty_answer_moves_on(server):
    answers = iter([chat(""), chat("second")])
    server["handler"] = lambda r: next(answers)
    assert await ai.generate(ALL, "hi") == "second"


async def test_on_call_sees_every_attempt_and_its_failures_are_ignored(server):
    answers = iter([httpx.Response(429), chat("ok")])
    server["handler"] = lambda r: next(answers)
    seen = []

    async def ledger(rec):
        seen.append((rec.provider, rec.status_code, rec.ok))
        raise RuntimeError("ledger down")

    assert await ai.complete(ALL[0], "hi", on_call=ledger) == "ok"
    assert seen == [("groq", 429, False), ("groq", 200, True)]


def test_importing_the_module_does_not_import_httpx():
    code = "import sys, tarun_common.ai_fallback; print('httpx' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
