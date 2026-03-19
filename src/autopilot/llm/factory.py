"""LLM factory — create the right provider from a model string.

Usage:
    llm = create_llm("claude-sonnet-4-20250514")     # → ClaudeLLM
    llm = create_llm("gpt-4o")                        # → OpenAILLM
    llm = create_llm("gemini/gemini-2.0-flash")       # → LiteLLMLLM
    llm = create_llm("ollama/llama3")                  # → LiteLLMLLM
    llm = create_llm("mock")                           # → MockLLM
    llm = create_llm(None)                             # → None (Tier 0-1 only)
"""

from __future__ import annotations

import logging

from autopilot.llm.base import BaseLLM, MockLLM

logger = logging.getLogger(__name__)

# Prefixes/patterns that route to specific providers
_CLAUDE_PREFIXES = ("claude-", "anthropic/")
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "openai/")
_LITELLM_PREFIXES = (
    "gemini/", "ollama/", "groq/", "together/", "deepseek/",
    "mistral/", "cohere/", "bedrock/", "vertex_ai/", "azure/",
    "huggingface/", "replicate/", "anyscale/", "perplexity/",
)


def create_llm(
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    **kwargs,
) -> BaseLLM | None:
    """Create an LLM provider from a model identifier string.

    Args:
        model: Model identifier. None returns None (Tier 0-1 only mode).
        api_key: Optional API key override.
        max_tokens: Maximum tokens for completions.

    Returns:
        BaseLLM instance or None.
    """
    if model is None:
        logger.info("No LLM configured — running in Tier 0-1 only mode")
        return None

    if model == "mock":
        return MockLLM(**kwargs)

    model_lower = model.lower()

    # Claude / Anthropic
    if any(model_lower.startswith(p) for p in _CLAUDE_PREFIXES):
        from autopilot.llm.claude import ClaudeLLM
        # Strip "anthropic/" prefix if present
        clean_model = model.replace("anthropic/", "")
        return ClaudeLLM(model=clean_model, api_key=api_key, max_tokens=max_tokens)

    # OpenAI
    if any(model_lower.startswith(p) for p in _OPENAI_PREFIXES):
        from autopilot.llm.openai import OpenAILLM
        clean_model = model.replace("openai/", "")
        return OpenAILLM(model=clean_model, api_key=api_key, max_tokens=max_tokens)

    # LiteLLM (catch-all for provider/model format)
    if any(model_lower.startswith(p) for p in _LITELLM_PREFIXES) or "/" in model:
        from autopilot.llm.litellm_provider import LiteLLMLLM
        return LiteLLMLLM(model=model, api_key=api_key, max_tokens=max_tokens)

    # Default: try LiteLLM as it handles the most providers
    try:
        from autopilot.llm.litellm_provider import LiteLLMLLM
        return LiteLLMLLM(model=model, api_key=api_key, max_tokens=max_tokens)
    except ImportError:
        pass

    # Fallback: try OpenAI format (many compatible APIs)
    try:
        from autopilot.llm.openai import OpenAILLM
        return OpenAILLM(model=model, api_key=api_key, max_tokens=max_tokens)
    except ImportError:
        pass

    raise ValueError(
        f"Cannot create LLM for model '{model}'. "
        f"Install a provider: pip install playwright-autopilot[claude|openai|litellm]"
    )
