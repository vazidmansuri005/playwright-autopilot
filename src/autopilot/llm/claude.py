"""Claude (Anthropic) LLM provider."""

from __future__ import annotations

import logging
import os

from autopilot.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class ClaudeLLM(BaseLLM):
    """Anthropic Claude provider using the official SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install playwright-autopilot[claude]"
            )
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)

        text = response.content[0].text if response.content else ""
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)

        response_text = response.content[0].text if response.content else ""
        return LLMResponse(
            text=response_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._model,
            raw=response,
        )

    def model_name(self) -> str:
        return self._model
