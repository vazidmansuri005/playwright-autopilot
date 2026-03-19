"""OpenAI GPT LLM provider."""

from __future__ import annotations

import logging
import os

from autopilot.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI GPT provider using the official SDK."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 1024,
        base_url: str | None = None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install playwright-autopilot[openai]"
            )
        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
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

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )

        response_text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=response_text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self._model,
            raw=response,
        )

    def model_name(self) -> str:
        return self._model
