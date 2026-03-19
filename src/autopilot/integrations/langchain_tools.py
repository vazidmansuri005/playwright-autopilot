"""LangChain integration — use autopilot tools with any LangChain agent.

Usage:
    from autopilot.integrations.langchain_tools import AutopilotToolkit

    toolkit = AutopilotToolkit(llm_model="claude-sonnet-4-20250514")
    tools = toolkit.get_tools()

    # Use with any LangChain agent
    from langchain.agents import create_tool_calling_agent
    agent = create_tool_calling_agent(llm, tools, prompt)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_langchain():
    try:
        from langchain_core.tools import StructuredTool
        return StructuredTool
    except ImportError:
        raise ImportError(
            "langchain-core required. Install with: pip install playwright-autopilot[langchain]"
        )


class AutopilotToolkit:
    """Provides LangChain-compatible tools for browser automation."""

    def __init__(
        self,
        llm_model: str | None = None,
        headless: bool = True,
        max_tier: int = 4,
    ):
        self._llm_model = llm_model
        self._headless = headless
        self._max_tier = max_tier
        self._claude_tools = None

    def _ensure_tools(self):
        if self._claude_tools is None:
            from autopilot.integrations.claude_api import AutopilotTools
            self._claude_tools = AutopilotTools(
                headless=self._headless,
                llm_model=self._llm_model,
                max_tier=self._max_tier,
            )
        return self._claude_tools

    def get_tools(self) -> list:
        """Return LangChain StructuredTool instances."""
        StructuredTool = _ensure_langchain()
        tools = self._ensure_tools()

        async def run_playbook(playbook_path: str, variables: dict | None = None, max_tier: int = 4) -> str:
            """Execute a browser automation playbook with tiered escalation."""
            return await tools.handle_tool_call("autopilot_run", {
                "playbook_path": playbook_path,
                "variables": variables or {},
                "max_tier": max_tier,
            })

        async def execute_step(intent: str, action: str = "click", selector: str = "", value: str = "") -> str:
            """Execute a single browser step with automatic healing."""
            return await tools.handle_tool_call("autopilot_step", {
                "intent": intent, "action": action, "selector": selector, "value": value,
            })

        async def navigate(url: str) -> str:
            """Navigate the browser to a URL."""
            return await tools.handle_tool_call("autopilot_navigate", {"url": url})

        async def snapshot(format: str = "compact") -> str:
            """Get current page state (compact ~100 tokens, full ~5K tokens, screenshot)."""
            return await tools.handle_tool_call("autopilot_snapshot", {"format": format})

        return [
            StructuredTool.from_function(coroutine=run_playbook, name="autopilot_run",
                description="Execute a browser automation playbook with tiered escalation."),
            StructuredTool.from_function(coroutine=execute_step, name="autopilot_step",
                description="Execute a single browser step with automatic healing."),
            StructuredTool.from_function(coroutine=navigate, name="autopilot_navigate",
                description="Navigate the browser to a URL."),
            StructuredTool.from_function(coroutine=snapshot, name="autopilot_snapshot",
                description="Get current page state in compact, full, or screenshot format."),
        ]

    async def close(self):
        if self._claude_tools:
            await self._claude_tools.close()
