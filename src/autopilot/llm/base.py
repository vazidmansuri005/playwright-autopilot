"""Abstract LLM interface — provider-agnostic."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: Any = None  # Provider-specific raw response

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMUsageTracker:
    """Track cumulative token usage across all LLM calls."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    calls: list[dict] = field(default_factory=list)

    def record(self, response: LLMResponse, tier: int) -> None:
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.call_count += 1
        self.calls.append({
            "tier": tier,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "model": response.model,
        })

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "by_tier": self._by_tier(),
        }

    def _by_tier(self) -> dict[int, int]:
        tiers: dict[int, int] = {}
        for call in self.calls:
            t = call["tier"]
            tiers[t] = tiers.get(t, 0) + call["input_tokens"] + call["output_tokens"]
        return tiers


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        """Send a text prompt and get a response."""
        ...

    @abstractmethod
    async def complete_with_vision(
        self,
        text: str,
        image_b64: str,
        media_type: str = "image/png",
        system: str = "",
    ) -> LLMResponse:
        """Send text + image and get a response (for Tier 4)."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...


class MockLLM(BaseLLM):
    """Mock LLM for testing — returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_index = 0

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        if self._call_index < len(self._responses):
            text = self._responses[self._call_index]
            self._call_index += 1
        else:
            text = "ref=e0"
        return LLMResponse(text=text, input_tokens=len(prompt.split()), output_tokens=5, model="mock")

    async def complete_with_vision(
        self, text: str, image_b64: str, media_type: str = "image/png", system: str = "",
    ) -> LLMResponse:
        return await self.complete(text, system)

    def model_name(self) -> str:
        return "mock"
