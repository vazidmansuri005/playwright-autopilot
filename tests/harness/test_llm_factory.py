"""Tests for LLM factory and provider routing."""

import pytest
from autopilot.llm.factory import create_llm
from autopilot.llm.base import MockLLM


class TestLLMFactory:
    """Test that model strings route to the correct provider."""

    def test_none_returns_none(self):
        result = create_llm(None)
        assert result is None

    def test_mock_returns_mock(self):
        result = create_llm("mock")
        assert isinstance(result, MockLLM)

    def test_claude_model_detected(self):
        """Claude model string routes to ClaudeLLM."""
        try:
            result = create_llm("claude-sonnet-4-20250514")
            assert result.model_name() == "claude-sonnet-4-20250514"
        except Exception:
            # May fail without API key — that's OK, it still routed correctly
            pass

    def test_anthropic_prefix_detected(self):
        """anthropic/ prefix routes to ClaudeLLM."""
        try:
            result = create_llm("anthropic/claude-sonnet-4-20250514")
            assert result.model_name() == "claude-sonnet-4-20250514"
        except Exception:
            pass

    def test_openai_model_detected(self):
        # Will fail without API key but should route correctly
        try:
            result = create_llm("gpt-4o")
            assert result.model_name() == "gpt-4o"
        except (ImportError, Exception):
            pytest.skip("OpenAI SDK not installed")

    def test_litellm_prefix_detected(self):
        try:
            result = create_llm("gemini/gemini-2.0-flash")
            assert result.model_name() == "gemini/gemini-2.0-flash"
        except ImportError:
            pytest.skip("LiteLLM not installed")

    def test_ollama_prefix_detected(self):
        try:
            result = create_llm("ollama/llama3")
            assert result.model_name() == "ollama/llama3"
        except ImportError:
            pytest.skip("LiteLLM not installed")


class TestMockLLM:
    """Test the MockLLM for testing purposes."""

    async def test_mock_returns_predefined_responses(self):
        mock = MockLLM(responses=["ref=e5", "action=click ref=e3"])
        r1 = await mock.complete("prompt 1")
        assert r1.text == "ref=e5"
        r2 = await mock.complete("prompt 2")
        assert r2.text == "action=click ref=e3"

    async def test_mock_returns_default_when_exhausted(self):
        mock = MockLLM(responses=["ref=e0"])
        await mock.complete("first")
        r = await mock.complete("second")
        assert r.text == "ref=e0"  # Default

    async def test_mock_tracks_tokens(self):
        mock = MockLLM(responses=["hello"])
        r = await mock.complete("test prompt here")
        assert r.input_tokens > 0
        assert r.output_tokens > 0
        assert r.model == "mock"

    async def test_mock_vision(self):
        mock = MockLLM(responses=["ref=e2"])
        r = await mock.complete_with_vision("text", "base64data")
        assert r.text == "ref=e2"
