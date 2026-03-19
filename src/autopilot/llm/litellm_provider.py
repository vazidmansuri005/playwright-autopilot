"""LiteLLM provider — unified interface to 100+ LLM providers.

Supports: Claude, GPT, Gemini, Mistral, Ollama, Groq, Together, DeepSeek,
and any other provider supported by LiteLLM.

Usage:
    llm = LiteLLMLLM("anthropic/claude-sonnet-4-20250514")
    llm = LiteLLMLLM("gpt-4o")
    llm = LiteLLMLLM("gemini/gemini-2.0-flash")
    llm = LiteLLMLLM("ollama/llama3")
    llm = LiteLLMLLM("groq/llama-3.1-70b-versatile")
"""

from __future__ import annotations

import logging

from autopilot.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class LiteLLMLLM(BaseLLM):
    """LiteLLM provider for 100+ model providers."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm package required. Install with: pip install playwright-autopilot[litellm]"
            )
        self._model = model
        self._max_tokens = max_tokens

        # LiteLLM handles API keys via env vars, but allow override
        if api_key:
            self._litellm.api_key = api_key
        if api_base:
            self._litellm.api_base = api_base

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._litellm.acompletion(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            model=self._model,
            raw=response,
        )

    async def complete_with_vision(
        self,
        text: str,
        image_b64: str,
        media_type: str = "image/png",
        system: str = "",
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # LiteLLM normalizes vision format across providers
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                },
                {"type": "text", "text": text},
            ],
        })

        response = await self._litellm.acompletion(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )

        response_text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=response_text,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            model=self._model,
            raw=response,
        )

    def model_name(self) -> str:
        return self._model
