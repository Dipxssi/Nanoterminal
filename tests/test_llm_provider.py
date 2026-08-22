import llm
import pytest


def test_get_llm_provider_defaults_gemini(monkeypatch):
    monkeypatch.delenv("NANOTERMINAL_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.get_llm_provider() == "gemini"


def test_get_llm_provider_explicit_groq(monkeypatch):
    monkeypatch.setenv("NANOTERMINAL_LLM_PROVIDER", "groq")
    assert llm.get_llm_provider() == "groq"


def test_get_llm_provider_auto_groq_from_key(monkeypatch):
    monkeypatch.delenv("NANOTERMINAL_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert llm.get_llm_provider() == "groq"


def test_ask_text_raw_routes_to_groq(monkeypatch):
    monkeypatch.setenv("NANOTERMINAL_LLM_PROVIDER", "groq")
    calls: list[str] = []
    monkeypatch.setattr(
        llm, "ask_groq_raw", lambda p, max_tokens=512: calls.append(p) or "ok"
    )
    assert llm.ask_text_raw("hello") == "ok"
    assert calls == ["hello"]


def test_parse_retry_seconds():
    body = (
        '{"error":{"message":"Rate limit reached. Please try again in 15.3525s."}}'
    )
    assert llm._parse_retry_seconds(body) == pytest.approx(15.8525, rel=0.01)


def test_parse_retry_seconds_milliseconds():
    body = '{"error":{"message":"Please try again in 352.5ms"}}'
    assert llm._parse_retry_seconds(body) == pytest.approx(0.5525, rel=0.01)


def test_http_api_headers_include_user_agent():
    headers = llm._http_api_headers("test-key")
    assert headers["Authorization"] == "Bearer test-key"
    assert "User-Agent" in headers
    assert "Mozilla" in headers["User-Agent"]


def test_ask_text_raw_routes_to_grok(monkeypatch):
    monkeypatch.setenv("NANOTERMINAL_LLM_PROVIDER", "grok")
    calls: list[str] = []
    monkeypatch.setattr(llm, "ask_grok_raw", lambda p: calls.append(p) or "ok")
    assert llm.ask_text_raw("hello") == "ok"
    assert calls == ["hello"]
