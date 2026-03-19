"""Claude API integration — use autopilot tools directly with the Anthropic SDK.

Usage:
    import anthropic
    from autopilot.integrations.claude_api import AutopilotTools

    client = anthropic.Anthropic()
    tools = AutopilotTools()

    # Get tool schemas for Claude API
    tool_defs = tools.get_tool_definitions()

    # In your agentic loop, handle tool calls:
    result = await tools.handle_tool_call(tool_name, tool_input)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.digest import extract_digest
from autopilot.core.healer import heal
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner
from autopilot.core.snapshot import extract_snapshot
from autopilot.core.vision import capture_for_vision
from autopilot.llm.factory import create_llm

logger = logging.getLogger(__name__)


TOOL_DEFINITIONS = [
    {
        "name": "autopilot_run",
        "description": (
            "Execute a browser automation playbook with tiered escalation. "
            "Tries cached selectors first (0 tokens), escalates to AI only when needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "playbook_path": {"type": "string", "description": "Path to playbook JSON file"},
                "variables": {
                    "type": "object",
                    "description": "Template variable substitutions",
                    "additionalProperties": {"type": "string"},
                },
                "max_tier": {
                    "type": "integer",
                    "description": "Max escalation tier (0=replay, 1=+heal, 2=+compact AI, 3=+full AI, 4=+vision)",
                    "default": 4,
                },
            },
            "required": ["playbook_path"],
        },
    },
    {
        "name": "autopilot_step",
        "description": (
            "Execute a single browser step. Provide intent and optional selector. "
            "Falls back through healing tiers automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "description": "What this step does (e.g., 'click login button')"},
                "action": {
                    "type": "string",
                    "description": "Action type",
                    "enum": ["click", "fill", "select", "press", "check", "uncheck", "hover", "clear"],
                    "default": "click",
                },
                "selector": {"type": "string", "description": "CSS selector to try first"},
                "value": {"type": "string", "description": "Value for fill/select/press actions"},
            },
            "required": ["intent"],
        },
    },
    {
        "name": "autopilot_navigate",
        "description": "Navigate the browser to a URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "autopilot_snapshot",
        "description": (
            "Get current page state. Use 'compact' for ~100 tokens (interactive elements only), "
            "'full' for ~5K tokens (accessibility tree), or 'screenshot' for base64 PNG."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Output format",
                    "enum": ["compact", "full", "screenshot"],
                    "default": "compact",
                },
            },
        },
    },
]


class AutopilotTools:
    """Provides Claude API tool definitions and handles tool calls."""

    def __init__(
        self,
        headless: bool = True,
        llm_model: str | None = None,
        max_tier: int = 4,
    ):
        self._headless = headless
        self._llm_model = llm_model
        self._max_tier = max_tier
        self._browser: Browser | None = None
        self._runner: Runner | None = None

    def get_tool_definitions(self) -> list[dict]:
        """Return tool schemas for Claude API messages.create(tools=...)."""
        return TOOL_DEFINITIONS

    async def handle_tool_call(self, name: str, input: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string."""
        browser = await self._ensure_browser()
        page = browser.page

        if name == "autopilot_run":
            return await self._handle_run(input)
        elif name == "autopilot_step":
            return await self._handle_step(input, page)
        elif name == "autopilot_navigate":
            await page.goto(input["url"], wait_until="domcontentloaded")
            title = await page.title()
            return json.dumps({"url": input["url"], "title": title})
        elif name == "autopilot_snapshot":
            return await self._handle_snapshot(input, page)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None

    async def _ensure_browser(self) -> Browser:
        if self._browser is None:
            config = BrowserConfig(headless=self._headless)
            self._browser = Browser(config)
            await self._browser.start()
            llm = create_llm(self._llm_model) if self._llm_model else None
            self._runner = Runner(browser=self._browser, llm=llm, max_tier=self._max_tier)
        return self._browser

    async def _handle_run(self, input: dict) -> str:
        playbook = Playbook.load(input["playbook_path"])
        result = await self._runner.run(
            playbook,
            variables=input.get("variables", {}),
        )
        return json.dumps(result.summary, indent=2)

    async def _handle_step(self, input: dict, page) -> str:
        step = PlaybookStep(
            intent=input["intent"],
            selector=input.get("selector", ""),
            action=input.get("action", "click"),
            value=input.get("value"),
        )
        result = await self._runner._execute_step(page, step, 0, {})
        return json.dumps({
            "success": result.success,
            "tier": result.tier,
            "selector_used": result.selector_used,
            "tokens_used": result.tokens_used,
        }, indent=2)

    async def _handle_snapshot(self, input: dict, page) -> str:
        fmt = input.get("format", "compact")
        if fmt == "compact":
            digest = await extract_digest(page)
            return digest.to_prompt()
        elif fmt == "full":
            snapshot = await extract_snapshot(page, interactive_only=True)
            return f"Page: {snapshot.title} ({snapshot.url})\n\n{snapshot.tree}"
        elif fmt == "screenshot":
            capture = await capture_for_vision(page)
            return json.dumps({
                "screenshot_b64": capture.screenshot_b64,
                "accessibility_summary": capture.accessibility_summary,
            })
        return json.dumps({"error": f"Unknown format: {fmt}"})
