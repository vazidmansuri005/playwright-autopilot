"""The Tiered Escalation Runner — the heart of playwright-autopilot.

Executes playbook steps through 5 tiers:
  Tier 0: Deterministic replay (0 tokens)
  Tier 1: Heuristic self-heal (0 tokens)
  Tier 2: Compact AI with page digest (~500 tokens)
  Tier 3: Full AI with accessibility tree (~5K tokens)
  Tier 4: Vision fallback with screenshot (~10K tokens)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Locator, Page

from autopilot.core.browser import Browser
from autopilot.core.digest import extract_digest, extract_focused_digest
from autopilot.core.healer import HealResult, heal, try_selector, try_role_selector
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.snapshot import extract_snapshot
from autopilot.core.vision import capture_for_vision, format_vision_prompt
from autopilot.llm.base import BaseLLM, LLMUsageTracker

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a single playbook step."""
    success: bool
    tier: int                    # Which tier resolved this step (-1 = all failed)
    step_index: int
    intent: str
    selector_used: str = ""
    strategy: str = ""           # How it was resolved
    duration_ms: float = 0
    tokens_used: int = 0
    error: str = ""


@dataclass
class RunResult:
    """Result of a full playbook run."""
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0
    tier_counts: dict[int, int] = field(default_factory=dict)
    playbook_updated: bool = False
    visual_diffs: list = field(default_factory=list)
    assertion_results: list[dict] = field(default_factory=list)
    performance: dict = field(default_factory=dict)
    audit_path: str = ""

    @property
    def summary(self) -> dict:
        result = {
            "success": self.success,
            "steps_total": len(self.steps),
            "steps_passed": sum(1 for s in self.steps if s.success),
            "steps_skipped": sum(1 for s in self.steps if s.strategy == "skipped"),
            "total_tokens": self.total_tokens,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "tier_distribution": self.tier_counts,
        }
        if self.assertion_results:
            result["assertions_passed"] = sum(1 for a in self.assertion_results if a.get("passed"))
            result["assertions_failed"] = sum(1 for a in self.assertion_results if not a.get("passed"))
        if self.visual_diffs:
            result["visual_changes"] = sum(1 for d in self.visual_diffs if d.changed)
        if self.performance:
            result["performance"] = self.performance
        return result


class Runner:
    """Tiered escalation engine for browser automation.

    Executes playbook steps starting from the cheapest tier (deterministic
    replay) and escalating only when needed. Caches successful resolutions
    back to the playbook for future runs.
    """

    # System prompts for LLM tiers
    TIER2_SYSTEM = (
        "You are a browser automation assistant. Given a compact list of interactive "
        "page elements, identify which element matches the user's intent. "
        "Respond with ONLY the element reference, e.g.: ref=e5"
    )

    TIER3_SYSTEM = (
        "You are a browser automation assistant. Given an accessibility tree of a web page, "
        "determine what action to take to accomplish the user's intent. "
        "Respond in the format: action=click ref=e5 OR action=fill ref=e3 value=hello"
    )

    TIER4_SYSTEM = (
        "You are a browser automation assistant analyzing a screenshot. "
        "Identify the element to interact with and the action to take. "
        "Respond in the format: action=click ref=e5 OR action=fill ref=e3 value=hello"
    )

    def __init__(
        self,
        browser: Browser,
        llm: BaseLLM | None = None,
        max_tier: int = 4,
        tier0_timeout: int = 3000,
        tier1_timeout: int = 5000,
        save_on_heal: bool = True,
        visual_diff: bool = False,
        audit: bool = False,
        profile: bool = False,
    ):
        self.browser = browser
        self.llm = llm
        self.max_tier = max_tier
        self.tier0_timeout = tier0_timeout
        self.tier1_timeout = tier1_timeout
        self.save_on_heal = save_on_heal
        self.usage = LLMUsageTracker()

        # Optional features
        self._visual_diff = None
        if visual_diff:
            from autopilot.core.visual_diff import VisualDiffTracker
            self._visual_diff = VisualDiffTracker()

        self._audit = None
        if audit:
            from autopilot.core.audit import AuditTrail
            self._audit = AuditTrail()

        self._profile = profile
        self._perf_data: list[dict] = []

    async def run(
        self,
        playbook: Playbook,
        variables: dict[str, str] | None = None,
    ) -> RunResult:
        """Execute a full playbook with tiered escalation.

        Args:
            playbook: The playbook to execute
            variables: Variable substitutions for templated values

        Returns:
            RunResult with per-step details and token usage
        """
        variables = variables or {}
        page = self.browser.page
        result = RunResult(success=True)
        run_start = time.monotonic()

        # Navigate to starting URL
        start_url = playbook.url
        for var_name, var_val in variables.items():
            start_url = start_url.replace(f"${{{var_name}}}", var_val)

        logger.info("Starting playbook '%s' at %s (%d steps)", playbook.name, start_url, len(playbook.steps))
        await page.goto(start_url, wait_until="domcontentloaded")

        for i, step in enumerate(playbook.steps):
            step_start = time.monotonic()

            # --- Conditional execution ---
            if step.condition:
                should_run = await self._evaluate_condition(page, step)
                if not should_run:
                    skip_result = StepResult(
                        success=True, tier=0, step_index=i,
                        intent=step.intent, strategy="skipped",
                    )
                    result.steps.append(skip_result)
                    logger.info("Step %d skipped (condition: %s)", i + 1, step.condition)
                    continue

            # --- Network mocking ---
            if step.network_mocks:
                await self._setup_network_mocks(page, step.network_mocks)

            # --- Execute step ---
            step_result = await self._execute_step(page, step, i, variables)
            step_result.duration_ms = (time.monotonic() - step_start) * 1000

            # Track per-step stats
            step.run_count += 1

            result.steps.append(step_result)
            tier = step_result.tier
            result.tier_counts[tier] = result.tier_counts.get(tier, 0) + 1
            result.total_tokens += step_result.tokens_used

            # --- Audit trail ---
            if self._audit:
                self._audit.record(
                    step_index=i, intent=step.intent, action=step.action,
                    selector=step_result.selector_used, tier=tier,
                    strategy=step_result.strategy, success=step_result.success,
                    duration_ms=step_result.duration_ms, tokens_used=step_result.tokens_used,
                    value=step.resolve_value(variables), url=page.url,
                    error=step_result.error,
                )

            # --- Performance profiling ---
            if self._profile:
                perf = await self._capture_performance(page, i, step.intent)
                self._perf_data.append(perf)

            if not step_result.success:
                if step.skip_on_fail:
                    step.fail_count += 1
                    logger.warning("Step %d failed but skip_on_fail=True: %s", i, step_result.error)
                    step_result.success = True
                    step_result.strategy = "skipped_on_fail"
                else:
                    step.fail_count += 1
                    result.success = False
                    logger.error("Step %d failed: %s", i, step_result.error)
                    break

            # --- Post-step assertion ---
            if step.assert_after and step_result.success:
                from autopilot.core.assertions import evaluate_assertion
                passed, msg = await evaluate_assertion(page, step.assert_after, self.llm)
                result.assertion_results.append({
                    "step": i, "assertion": step.assert_after, "passed": passed, "message": msg,
                })
                if not passed:
                    logger.warning("Assertion failed at step %d: %s", i, msg)

            # --- Visual diff ---
            if self._visual_diff and step_result.success:
                diff = await self._visual_diff.capture_and_compare(
                    page, playbook.name, i, step.intent,
                )
                result.visual_diffs.append(diff)

            # Post-step wait
            if step.wait_after_ms > 0:
                await asyncio.sleep(step.wait_after_ms / 1000)

            # Update playbook if healed at tier > 0
            if tier > 0 and step_result.selector_used and self.save_on_heal:
                step.heal_count += 1
                self._update_playbook_step(playbook, i, step_result)
                result.playbook_updated = True

            logger.info(
                "Step %d/%d [Tier %d] '%s' → %s (%.0fms, %d tokens)",
                i + 1, len(playbook.steps), tier, step.intent,
                "OK" if step_result.success else "FAIL",
                step_result.duration_ms, step_result.tokens_used,
            )

        result.total_duration_ms = (time.monotonic() - run_start) * 1000
        playbook.record_run(result.success)

        # Save audit trail
        if self._audit:
            audit_path = self._audit.save(playbook.name)
            result.audit_path = str(audit_path)

        # Performance summary
        if self._profile and self._perf_data:
            durations = [p["duration_ms"] for p in self._perf_data]
            result.performance = {
                "avg_step_ms": round(sum(durations) / len(durations), 1),
                "slowest_step": max(self._perf_data, key=lambda p: p["duration_ms"]),
                "fastest_step": min(self._perf_data, key=lambda p: p["duration_ms"]),
                "total_ms": round(sum(durations), 1),
            }

        logger.info(
            "Playbook '%s' %s: %d/%d steps, %d tokens, %.1fs",
            playbook.name,
            "PASSED" if result.success else "FAILED",
            sum(1 for s in result.steps if s.success),
            len(playbook.steps),
            result.total_tokens,
            result.total_duration_ms / 1000,
        )
        return result

    async def _execute_step(
        self,
        page: Page,
        step: PlaybookStep,
        index: int,
        variables: dict[str, str],
    ) -> StepResult:
        """Execute a single step through tiered escalation."""

        value = step.resolve_value(variables)

        # --- TIER 0: Deterministic replay ---
        if self.max_tier >= 0:
            for selector in step.all_selectors():
                if selector.startswith("role:"):
                    locator = await try_role_selector(page, selector, timeout=self.tier0_timeout)
                else:
                    locator = await try_selector(page, selector, timeout=self.tier0_timeout)
                if locator:
                    success = await self._perform_action(locator, step.action, value)
                    if success:
                        return StepResult(
                            success=True, tier=0, step_index=index,
                            intent=step.intent, selector_used=selector,
                            strategy="replay",
                        )

        # --- TIER 1: Heuristic self-heal ---
        if self.max_tier >= 1:
            heal_result = await heal(page, step)
            if heal_result.success and heal_result.element:
                success = await self._perform_action(heal_result.element, step.action, value)
                if success:
                    return StepResult(
                        success=True, tier=1, step_index=index,
                        intent=step.intent, selector_used=heal_result.new_selector,
                        strategy=f"heal:{heal_result.strategy}",
                    )

        # --- TIER 2: Compact AI ---
        if self.max_tier >= 2 and self.llm:
            result = await self._tier2_compact_ai(page, step, index, value)
            if result and result.success:
                return result

        # --- TIER 3: Full AI ---
        if self.max_tier >= 3 and self.llm:
            result = await self._tier3_full_ai(page, step, index, value)
            if result and result.success:
                return result

        # --- TIER 4: Vision fallback ---
        if self.max_tier >= 4 and self.llm:
            result = await self._tier4_vision(page, step, index, value)
            if result and result.success:
                return result

        # All tiers exhausted
        return StepResult(
            success=False, tier=-1, step_index=index,
            intent=step.intent,
            error=f"All tiers exhausted for step: {step.intent}",
        )

    async def _tier2_compact_ai(
        self, page: Page, step: PlaybookStep, index: int, value: str | None,
    ) -> StepResult | None:
        """Tier 2: Send compact element digest to LLM (~500 tokens)."""
        try:
            digest = await extract_focused_digest(page, step.intent, max_elements=30)
            prompt = (
                f'I need to: "{step.intent}"\n'
                f'Original selector "{step.selector}" not found.\n\n'
                f'{digest.to_prompt()}\n\n'
                f'Which element matches? Reply with ONLY: ref=eN'
            )

            response = await self.llm.complete(prompt, system=self.TIER2_SYSTEM)
            self.usage.record(response, tier=2)

            ref = _parse_ref(response.text)
            if ref is not None:
                # Look up by ref string, not list index (filtered lists may have gaps)
                ref_str = f"e{ref}"
                element_data = next((el for el in digest.elements if el.get("ref") == ref_str), None)
            else:
                element_data = None
            if element_data is not None:
                # Build a selector from the element data
                selector = _build_selector(element_data)
                locator = await try_selector(page, selector, timeout=5000)
                if locator:
                    success = await self._perform_action(locator, step.action, value)
                    if success:
                        return StepResult(
                            success=True, tier=2, step_index=index,
                            intent=step.intent, selector_used=selector,
                            strategy="compact_ai",
                            tokens_used=response.total_tokens,
                        )
        except Exception as e:
            logger.warning("Tier 2 failed: %s", e)
        return None

    async def _tier3_full_ai(
        self, page: Page, step: PlaybookStep, index: int, value: str | None,
    ) -> StepResult | None:
        """Tier 3: Send filtered accessibility tree to LLM (~5K tokens)."""
        try:
            snapshot = await extract_snapshot(page, interactive_only=True)
            prompt = (
                f'I need to: "{step.intent}"\n'
                f'Action type: {step.action}\n'
                f'Page: {snapshot.title} ({snapshot.url})\n\n'
                f'Accessibility tree:\n{snapshot.tree}\n\n'
                f'What should I do? Reply: action=ACTION ref=eN [value=VALUE]'
            )

            response = await self.llm.complete(prompt, system=self.TIER3_SYSTEM)
            self.usage.record(response, tier=3)

            parsed = _parse_action_response(response.text)
            if parsed:
                ref_idx = parsed.get("ref")
                action = parsed.get("action", step.action)
                action_value = parsed.get("value", value)

                # Try to find element by ref in snapshot
                locator = await _find_by_snapshot_ref(page, ref_idx)
                if locator:
                    success = await self._perform_action(locator, action, action_value)
                    if success:
                        return StepResult(
                            success=True, tier=3, step_index=index,
                            intent=step.intent, selector_used=f"a11y:e{ref_idx}",
                            strategy="full_ai",
                            tokens_used=response.total_tokens,
                        )
        except Exception as e:
            logger.warning("Tier 3 failed: %s", e)
        return None

    async def _tier4_vision(
        self, page: Page, step: PlaybookStep, index: int, value: str | None,
    ) -> StepResult | None:
        """Tier 4: Send screenshot to vision LLM (~10K tokens)."""
        try:
            capture = await capture_for_vision(page)
            vision_prompt = format_vision_prompt(capture, step.intent)

            response = await self.llm.complete_with_vision(
                text=vision_prompt["text"],
                image_b64=vision_prompt["image_b64"],
                media_type=vision_prompt["media_type"],
                system=self.TIER4_SYSTEM,
            )
            self.usage.record(response, tier=4)

            parsed = _parse_action_response(response.text)
            if parsed:
                ref_idx = parsed.get("ref")
                action = parsed.get("action", step.action)
                action_value = parsed.get("value", value)

                locator = await _find_by_snapshot_ref(page, ref_idx)
                if locator:
                    success = await self._perform_action(locator, action, action_value)
                    if success:
                        return StepResult(
                            success=True, tier=4, step_index=index,
                            intent=step.intent, selector_used=f"vision:e{ref_idx}",
                            strategy="vision",
                            tokens_used=response.total_tokens,
                        )
        except Exception as e:
            logger.warning("Tier 4 failed: %s", e)
        return None

    async def _perform_action(
        self, locator: Locator, action: str, value: str | None = None,
    ) -> bool:
        """Perform a browser action on a locator."""
        try:
            match action:
                case "click":
                    await locator.click()
                case "fill" | "type":
                    if value is not None:
                        await locator.fill(value)
                    else:
                        logger.warning("Fill action with no value")
                        return False
                case "select":
                    if value is not None:
                        await locator.select_option(value)
                    else:
                        return False
                case "press":
                    if value is not None:
                        await locator.press(value)
                    else:
                        return False
                case "check":
                    await locator.check()
                case "uncheck":
                    await locator.uncheck()
                case "hover":
                    await locator.hover()
                case "clear":
                    await locator.clear()
                case _:
                    logger.warning("Unknown action: %s", action)
                    return False
            return True
        except Exception as e:
            logger.warning("Action '%s' failed: %s", action, e)
            return False

    async def _evaluate_condition(self, page: Page, step: PlaybookStep) -> bool:
        """Evaluate a step's condition to decide if it should run."""
        condition = step.condition.lower().strip()

        if condition == "if_visible":
            for sel in step.all_selectors():
                try:
                    locator = page.locator(sel)
                    if await locator.count() > 0 and await locator.first.is_visible():
                        return True
                except Exception:
                    continue
            return False

        elif condition == "if_not_visible":
            for sel in step.all_selectors():
                try:
                    locator = page.locator(sel)
                    if await locator.count() > 0 and await locator.first.is_visible():
                        return False
                except Exception:
                    continue
            return True

        elif condition == "if_url_contains":
            return step.condition_value in page.url

        elif condition == "if_url_not_contains":
            return step.condition_value not in page.url

        elif condition == "if_text_visible":
            try:
                locator = page.get_by_text(step.condition_value, exact=False)
                return await locator.count() > 0
            except Exception:
                return False

        return True  # Unknown condition — run by default

    async def _setup_network_mocks(self, page: Page, mocks: list[dict]) -> None:
        """Set up network route mocking for a step."""
        for mock in mocks:
            url_pattern = mock.get("url", "")
            response_body = mock.get("response", {})
            status = mock.get("status", 200)

            async def handle_route(route, body=response_body, code=status):
                await route.fulfill(
                    status=code,
                    content_type="application/json",
                    body=json.dumps(body) if isinstance(body, dict) else str(body),
                )

            await page.route(url_pattern, handle_route)
            logger.info("Network mock set: %s → %s", url_pattern, status)

    async def _capture_performance(self, page: Page, step_index: int, intent: str) -> dict:
        """Capture performance metrics after a step."""
        try:
            metrics = await page.evaluate("""
            () => {
                const perf = performance.getEntriesByType('navigation')[0] || {};
                return {
                    dom_content_loaded: perf.domContentLoadedEventEnd || 0,
                    load_complete: perf.loadEventEnd || 0,
                    dom_interactive: perf.domInteractive || 0,
                };
            }
            """)
        except Exception:
            metrics = {}

        return {
            "step_index": step_index,
            "intent": intent,
            "duration_ms": 0,  # Filled by caller
            "url": page.url,
            **metrics,
        }

    def _update_playbook_step(
        self, playbook: Playbook, index: int, result: StepResult,
    ) -> None:
        """Update a playbook step with a healed selector."""
        step = playbook.steps[index]
        old_selector = step.selector

        # Promote the working selector to primary
        if result.selector_used and result.selector_used != old_selector:
            # Move old primary to alternatives
            if old_selector not in step.selector_alternatives:
                step.selector_alternatives.insert(0, old_selector)
            step.selector = result.selector_used
            step.tier_resolved = result.tier
            playbook.update_step(index, selector=result.selector_used)
            logger.info(
                "Playbook updated: step %d selector '%s' → '%s'",
                index, old_selector, result.selector_used,
            )


# --- Response parsing helpers ---

def _parse_ref(text: str) -> int | None:
    """Parse 'ref=eN' from LLM response."""
    match = re.search(r"ref\s*=\s*e?(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: just look for eN pattern
    match = re.search(r"\be(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _parse_action_response(text: str) -> dict[str, Any] | None:
    """Parse 'action=click ref=e5 value=hello' from LLM response."""
    result: dict[str, Any] = {}

    ref_match = re.search(r"ref\s*=\s*e?(\d+)", text, re.IGNORECASE)
    if ref_match:
        result["ref"] = int(ref_match.group(1))

    action_match = re.search(r"action\s*=\s*(\w+)", text, re.IGNORECASE)
    if action_match:
        result["action"] = action_match.group(1).lower()

    value_match = re.search(r"value\s*=\s*(.+?)(?:\s+\w+=|$)", text, re.IGNORECASE)
    if value_match:
        result["value"] = value_match.group(1).strip().strip("'\"")

    return result if "ref" in result else None


def _build_selector(element_data: dict) -> str:
    """Build a CSS selector from digest element data."""
    tag = element_data.get("tag", "")
    el_type = element_data.get("type", "")
    name = element_data.get("name", "")
    placeholder = element_data.get("placeholder", "")
    role = element_data.get("role", "")

    # Try specific selectors first
    if role:
        if name:
            return f'[role="{role}"][aria-label="{name}"]'
        return f'[role="{role}"]'

    if tag == "input" and el_type:
        if placeholder:
            return f'input[type="{el_type}"][placeholder="{placeholder}"]'
        return f'input[type="{el_type}"]'

    if tag == "button" and name:
        return f'button:has-text("{name[:30]}")'

    if tag == "a" and name:
        return f'a:has-text("{name[:30]}")'

    if name:
        return f'{tag}:has-text("{name[:30]}")'

    return tag


async def _find_by_snapshot_ref(page: Page, ref_idx: int | None) -> Locator | None:
    """Find an element by its snapshot reference index."""
    if ref_idx is None:
        return None

    try:
        # Get all interactive elements in order (matching digest/snapshot extraction)
        elements = await page.evaluate(f"""
        () => {{
            const INTERACTIVE = [
                'a[href]', 'button', 'input', 'select', 'textarea',
                '[role="button"]', '[role="link"]', '[role="textbox"]',
                '[role="checkbox"]', '[role="radio"]', '[role="combobox"]',
                '[role="menuitem"]', '[role="tab"]', '[role="switch"]',
                '[tabindex]:not([tabindex="-1"])',
                '[contenteditable="true"]',
            ].join(', ');
            const all = document.querySelectorAll(INTERACTIVE);
            const visible = Array.from(all).filter(el => {{
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            }});
            if ({ref_idx} < visible.length) {{
                const el = visible[{ref_idx}];
                // Generate a unique selector
                if (el.id) return '#' + el.id;
                if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                // nth-child fallback
                const parent = el.parentElement;
                if (parent) {{
                    const siblings = Array.from(parent.children);
                    const idx = siblings.indexOf(el) + 1;
                    return el.tagName.toLowerCase() + ':nth-child(' + idx + ')';
                }}
                return el.tagName.toLowerCase();
            }}
            return null;
        }}
        """)
        if elements:
            return await try_selector(page, elements, timeout=3000)
    except Exception as e:
        logger.warning("Failed to find element by ref e%s: %s", ref_idx, e)

    return None
