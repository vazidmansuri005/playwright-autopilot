"""AI Explorer — give it a URL and a goal, it figures out the steps.

This is the "learning mode" that makes autopilot a real AI tool.
It uses compact page digests (~500 tokens per step) to decide what
to do next, records every action as a PlaybookStep, and saves the
playbook for free replay on subsequent runs.

Usage:
    explorer = Explorer(browser, llm)
    playbook = await explorer.explore(
        url="https://example.com",
        goal="Log in with test@test.com and go to settings",
    )
    # Playbook now has recorded steps — replay for free next time
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

from autopilot.core.browser import Browser
from autopilot.core.digest import extract_digest
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import _build_selector, _find_by_snapshot_ref, _parse_action_response
from autopilot.llm.base import BaseLLM, LLMUsageTracker

logger = logging.getLogger(__name__)


PLANNER_SYSTEM = """\
You are a browser automation agent. You navigate web pages to accomplish goals.

You see a compact list of interactive elements on the current page.
Decide the NEXT SINGLE action to take toward the goal.

Respond in EXACTLY this format (one line):
action=ACTION ref=REF [value=VALUE] | intent=DESCRIPTION

Where:
- ACTION: click, fill, select, press, check, hover, wait, done
- REF: element reference (e.g., e5)
- VALUE: text to type (required for fill/select/press)
- DESCRIPTION: short human-readable intent for this step

Examples:
  action=fill ref=e2 value=user@test.com | intent=fill email field
  action=click ref=e5 | intent=click login button
  action=select ref=e8 value=PST | intent=select pacific timezone
  action=done | intent=goal complete

Rules:
- Take ONE action at a time
- Use "action=done" when the goal is achieved
- Use "action=wait" if the page is loading and you need to wait
- If you need to type into a field, ALWAYS use action=fill
- Do not repeat the same action twice in a row
"""


@dataclass
class ExploreResult:
    """Result of an AI exploration session."""
    success: bool
    playbook: Playbook
    steps_taken: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0
    reason: str = ""  # Why exploration ended

    @property
    def summary(self) -> dict:
        return {
            "success": self.success,
            "steps_taken": self.steps_taken,
            "total_tokens": self.total_tokens,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "playbook_name": self.playbook.name,
            "reason": self.reason,
        }


class Explorer:
    """AI-driven browser explorer that records playbooks.

    Give it a URL and a goal. It uses compact page digests to decide
    what to do, executes actions, and records everything as a Playbook.
    """

    def __init__(
        self,
        browser: Browser,
        llm: BaseLLM,
        max_steps: int = 30,
        step_timeout: int = 10_000,
    ):
        if llm is None:
            raise ValueError("Explorer requires an LLM. Pass llm='claude-sonnet-4-20250514' or similar.")
        self.browser = browser
        self.llm = llm
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.usage = LLMUsageTracker()

    async def explore(
        self,
        url: str,
        goal: str,
        name: str | None = None,
        variables: dict[str, str] | None = None,
    ) -> ExploreResult:
        """Explore a website to accomplish a goal, recording a playbook.

        Args:
            url: Starting URL
            goal: Natural language goal description
            name: Playbook name (auto-generated if not provided)
            variables: Known variables to templatize (e.g., {"email": "user@test.com"})

        Returns:
            ExploreResult with the recorded playbook
        """
        variables = variables or {}
        page = self.browser.page
        start_time = time.monotonic()

        # Generate playbook name from goal
        if not name:
            name = _slugify(goal)

        playbook = Playbook(name=name, url=url, variables=list(variables.keys()))

        # Navigate to starting URL
        logger.info("Explorer: navigating to %s", url)
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(1)  # Let page settle

        completed_intents: list[str] = []
        last_action = ""
        consecutive_failures = 0

        for step_num in range(self.max_steps):
            # Get compact page digest
            digest = await extract_digest(page, max_elements=50, visible_only=True)

            # Build prompt with context
            prompt = self._build_prompt(goal, digest.to_prompt(), completed_intents, last_action)

            # Ask LLM for next action
            response = await self.llm.complete(prompt, system=PLANNER_SYSTEM)
            self.usage.record(response, tier=2)

            # Parse the response
            parsed = _parse_explore_response(response.text)
            if not parsed:
                consecutive_failures += 1
                logger.warning("Explorer: failed to parse LLM response: %s", response.text[:200])
                if consecutive_failures >= 3:
                    return ExploreResult(
                        success=False, playbook=playbook,
                        steps_taken=step_num, total_tokens=self.usage.total_tokens,
                        total_duration_ms=(time.monotonic() - start_time) * 1000,
                        reason="Failed to parse LLM responses",
                    )
                continue

            consecutive_failures = 0
            action = parsed["action"]
            intent = parsed.get("intent", f"step {step_num + 1}")

            # Check if done
            if action == "done":
                logger.info("Explorer: goal complete after %d steps", step_num)
                return ExploreResult(
                    success=True, playbook=playbook,
                    steps_taken=step_num, total_tokens=self.usage.total_tokens,
                    total_duration_ms=(time.monotonic() - start_time) * 1000,
                    reason="Goal achieved",
                )

            # Handle wait
            if action == "wait":
                await asyncio.sleep(2)
                last_action = "wait"
                continue

            # Execute the action
            ref_idx = parsed.get("ref")
            value = parsed.get("value", "")

            # Templatize values if they match known variables
            value_template = None
            if value:
                for var_name, var_val in variables.items():
                    if var_val and var_val in value:
                        value_template = value.replace(var_val, f"${{{var_name}}}")

            # Find the element using the digest data (NOT _find_by_snapshot_ref which re-queries DOM)
            selector = ""
            alternatives = []
            success = False

            # Look up element by ref string (e.g., "e13"), NOT by list index
            element_data = None
            if ref_idx is not None:
                ref_str = f"e{ref_idx}"
                for el in digest.elements:
                    if el.get("ref") == ref_str:
                        element_data = el
                        break

            if element_data is not None:
                selector = _build_selector(element_data)
                alternatives = _generate_alternatives(element_data)

                # Try the built selector first, then alternatives, then fallback
                from autopilot.core.healer import try_selector
                locator = None

                # For fill actions, also try Playwright's dedicated input locators
                if action in ("fill", "type") and element_data.get("placeholder"):
                    try:
                        pl = page.get_by_placeholder(element_data["placeholder"])
                        if await pl.count() > 0 and await pl.first.is_visible():
                            locator = pl.first
                            selector = f'[placeholder="{element_data["placeholder"]}"]'
                    except Exception:
                        pass

                if not locator and action in ("fill", "type") and element_data.get("name"):
                    try:
                        ll = page.get_by_label(element_data["name"], exact=False)
                        if await ll.count() > 0 and await ll.first.is_visible():
                            locator = ll.first
                            selector = f'label={element_data["name"]}'
                    except Exception:
                        pass

                # Try built selectors
                if not locator:
                    for sel in [selector] + alternatives:
                        locator = await try_selector(page, sel, timeout=3000)
                        if locator:
                            selector = sel
                            break

                if locator:
                    try:
                        success = await _perform_explore_action(locator, action, value)
                    except Exception as e:
                        logger.warning("Explorer: action failed: %s", e)

            if success:
                # Record the step
                step = PlaybookStep(
                    intent=intent,
                    selector=selector,
                    selector_alternatives=alternatives if ref_idx is not None else [],
                    action=action,
                    value=value if not value_template else None,
                    value_template=value_template,
                    confidence=0.9,
                    tier_resolved=2,
                    wait_after_ms=500,  # Small wait between steps
                )
                playbook.add_step(step)
                completed_intents.append(intent)
                last_action = f"{action} {intent}"

                logger.info(
                    "Explorer step %d: %s %s → %s",
                    step_num + 1, action, intent, selector,
                )

                # Wait for page to settle after action
                await asyncio.sleep(0.5)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
            else:
                last_action = f"FAILED: {action} {intent}"
                logger.warning("Explorer: failed to execute step %d: %s", step_num + 1, intent)

        # Hit max steps
        return ExploreResult(
            success=len(playbook.steps) > 0,
            playbook=playbook,
            steps_taken=self.max_steps,
            total_tokens=self.usage.total_tokens,
            total_duration_ms=(time.monotonic() - start_time) * 1000,
            reason=f"Max steps ({self.max_steps}) reached",
        )

    async def extract(
        self,
        url: str,
        goal: str,
        schema: dict[str, str] | None = None,
    ) -> list[dict] | str:
        """Extract structured data from a page using AI.

        Args:
            url: Page URL
            goal: What data to extract (e.g., "top 5 stories with title, points, and URL")
            schema: Optional schema hint (e.g., {"title": "str", "points": "int"})

        Returns:
            Extracted data as a list of dicts (if schema provided) or text.
        """
        page = self.browser.page
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(1)

        # Get page content via digest + text
        digest = await extract_digest(page, max_elements=80, visible_only=True)

        # Also get visible text content (truncated)
        text_content = await page.evaluate("""
        () => {
            const main = document.querySelector('main') || document.body;
            return main.innerText.substring(0, 3000);
        }
        """)

        schema_hint = ""
        if schema:
            schema_hint = f"\n\nExpected output schema: {schema}\nReturn as JSON array."

        prompt = (
            f"Goal: {goal}{schema_hint}\n\n"
            f"Page: {digest.title} ({digest.url})\n\n"
            f"Page text (truncated):\n{text_content[:2000]}\n\n"
            f"Interactive elements:\n{digest.to_prompt()}\n\n"
            f"Extract the requested data. Return ONLY valid JSON."
        )

        system = (
            "You are a data extraction assistant. Extract structured data from web pages. "
            "Return ONLY valid JSON — no markdown, no explanation."
        )

        response = await self.llm.complete(prompt, system=system)
        self.usage.record(response, tier=2)

        # Try to parse as JSON
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        try:
            import json
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    def _build_prompt(
        self,
        goal: str,
        page_digest: str,
        completed: list[str],
        last_action: str,
    ) -> str:
        """Build the exploration prompt."""
        parts = [f"GOAL: {goal}"]

        if completed:
            parts.append(f"\nCompleted steps ({len(completed)}):")
            for i, intent in enumerate(completed[-5:], 1):  # Last 5 steps
                parts.append(f"  {i}. {intent}")

        if last_action:
            parts.append(f"\nLast action: {last_action}")

        parts.append(f"\nCurrent page:\n{page_digest}")
        parts.append("\nWhat is the NEXT action? (one line)")

        return "\n".join(parts)


def _parse_explore_response(text: str) -> dict[str, Any] | None:
    """Parse explorer LLM response — lenient to handle varied LLM output.

    Expected format: action=fill ref=e2 value=user@test.com | intent=fill email field
    Also handles: action=CLICK ref=e13 (no intent, uppercase)
    Also handles: click e13 (shorthand)
    Also handles: I'll click on the search box (ref=e13) (natural language)
    """
    if not text or not text.strip():
        return None

    text = text.strip()
    # Take first meaningful line (skip empty lines, explanations)
    for line in text.split("\n"):
        line = line.strip()
        if line and ("action=" in line.lower() or "ref=" in line.lower() or re.match(r"^\w+=", line)):
            text = line
            break

    result: dict[str, Any] = {}

    # Parse action
    action_match = re.search(r"action\s*=\s*(\w+)", text, re.IGNORECASE)
    if action_match:
        result["action"] = action_match.group(1).lower()
    else:
        # Try shorthand: "click e13" or "fill e2 value=..."
        shorthand = re.match(r"^(click|fill|select|press|check|hover|wait|done|navigate)\s", text, re.IGNORECASE)
        if shorthand:
            result["action"] = shorthand.group(1).lower()
        else:
            return None

    # Parse ref
    ref_match = re.search(r"ref\s*=\s*e?(\d+)", text, re.IGNORECASE)
    if ref_match:
        result["ref"] = int(ref_match.group(1))
    else:
        # Try bare eN pattern
        bare_ref = re.search(r"\be(\d+)\b", text)
        if bare_ref:
            result["ref"] = int(bare_ref.group(1))

    # Parse value — handle quoted and unquoted
    value_match = re.search(r'value\s*=\s*["\'](.+?)["\']', text, re.IGNORECASE)
    if not value_match:
        value_match = re.search(r"value\s*=\s*(.+?)(?:\s*\||\s+ref=|\s+action=|\s+intent=|$)", text, re.IGNORECASE)
    if value_match:
        result["value"] = value_match.group(1).strip().strip("'\"")

    # Parse intent
    intent_match = re.search(r"intent\s*=\s*(.+?)$", text, re.IGNORECASE)
    if intent_match:
        result["intent"] = intent_match.group(1).strip()
    elif "|" in text:
        # Try to get intent after pipe
        parts = text.split("|", 1)
        if len(parts) > 1:
            intent_text = parts[1].strip()
            if intent_text:
                result["intent"] = intent_text

    # For fill/type actions without explicit value, check if the goal context has it
    if result.get("action") in ("fill", "type") and "value" not in result:
        # LLM sometimes puts the value after ref without value= prefix
        after_ref = re.search(r"ref\s*=\s*e?\d+\s+(.+?)(?:\s*\||$)", text, re.IGNORECASE)
        if after_ref:
            potential_val = after_ref.group(1).strip()
            if potential_val and not potential_val.startswith("intent") and not potential_val.startswith("action"):
                result["value"] = potential_val.strip("'\"")

    return result


def _generate_alternatives(element_data: dict) -> list[str]:
    """Generate alternative selectors from element data for the playbook."""
    alternatives = []
    tag = element_data.get("tag", "")
    name = element_data.get("name", "")
    el_type = element_data.get("type", "")
    placeholder = element_data.get("placeholder", "")
    role = element_data.get("role", "")
    href = element_data.get("href", "")

    if role and name:
        alternatives.append(f"role:{role}[name='{name[:30]}']")
    if tag == "input" and el_type:
        alternatives.append(f"input[type='{el_type}']")
    if placeholder:
        alternatives.append(f"[placeholder='{placeholder[:40]}']")
    if tag == "a" and href:
        alternatives.append(f"a[href='{href[:60]}']")
    if name and len(name) < 40:
        alternatives.append(f"{tag}:has-text('{name[:30]}')")
    if tag == "button" and name:
        alternatives.append(f"role:button[name='{name[:30]}']")

    return alternatives


async def _perform_explore_action(locator, action: str, value: str = "") -> bool:
    """Perform an action during exploration."""
    try:
        match action:
            case "click":
                await locator.click(timeout=5000)
            case "fill" | "type":
                await locator.fill(value, timeout=5000)
            case "select":
                await locator.select_option(value, timeout=5000)
            case "press":
                await locator.press(value)
            case "check":
                await locator.check(timeout=5000)
            case "uncheck":
                await locator.uncheck(timeout=5000)
            case "hover":
                await locator.hover(timeout=5000)
            case _:
                return False
        return True
    except Exception as e:
        logger.warning("Explore action '%s' failed: %s", action, e)
        return False


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a URL-safe slug for playbook naming."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len]
