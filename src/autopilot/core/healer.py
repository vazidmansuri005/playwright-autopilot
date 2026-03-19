"""Tier 1: Heuristic self-healing — zero LLM tokens."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from playwright.async_api import Locator, Page

from autopilot.core.playbook import PlaybookStep

logger = logging.getLogger(__name__)


@dataclass
class HealResult:
    """Result of a healing attempt."""
    success: bool
    element: Locator | None = None
    new_selector: str = ""
    strategy: str = ""
    score: float = 0.0


async def try_selector(page: Page, selector: str, timeout: int = 3000) -> Locator | None:
    """Try a selector, return locator if found within timeout."""
    try:
        locator = page.locator(selector)
        await locator.first.wait_for(state="visible", timeout=timeout)
        if await locator.first.is_visible():
            return locator.first
    except Exception:
        pass
    return None


async def try_role_selector(page: Page, selector: str, timeout: int = 3000) -> Locator | None:
    """Try a role-based selector like 'role:button[name=Submit]'."""
    match = re.match(r"role:(\w+)\[name=['\"]?(.+?)['\"]?\]", selector)
    if not match:
        return None
    role, name = match.groups()
    try:
        locator = page.get_by_role(role, name=name)
        await locator.first.wait_for(state="visible", timeout=timeout)
        if await locator.first.is_visible():
            return locator.first
    except Exception:
        pass
    return None


async def heal_with_alternatives(page: Page, step: PlaybookStep) -> HealResult:
    """Strategy 1: Try all alternative selectors from the playbook."""
    for alt in step.selector_alternatives:
        # Handle role-based selectors
        if alt.startswith("role:"):
            locator = await try_role_selector(page, alt)
        else:
            locator = await try_selector(page, alt)

        if locator:
            logger.info("Healed via alternative selector: %s", alt)
            return HealResult(success=True, element=locator, new_selector=alt, strategy="alternative", score=0.9)

    return HealResult(success=False)


async def heal_with_text_match(page: Page, step: PlaybookStep) -> HealResult:
    """Strategy 2: Match by text content (for clickable elements)."""
    if step.action not in ("click", "navigate"):
        return HealResult(success=False)

    # Extract meaningful words from intent
    words = _extract_intent_keywords(step.intent)
    if not words:
        return HealResult(success=False)

    for word in words:
        try:
            locator = page.get_by_text(word, exact=False)
            count = await locator.count()
            if count == 1:
                if await locator.first.is_visible():
                    new_sel = f"text={word}"
                    logger.info("Healed via text match: %s", new_sel)
                    return HealResult(success=True, element=locator.first, new_selector=new_sel, strategy="text_match", score=0.7)
        except Exception:
            continue

    return HealResult(success=False)


async def heal_with_role(page: Page, step: PlaybookStep) -> HealResult:
    """Strategy 3: Match by ARIA role + approximate name."""
    # Infer role from action and selector
    role = _infer_role(step)
    if not role:
        return HealResult(success=False)

    keywords = _extract_intent_keywords(step.intent)

    try:
        for keyword in keywords:
            locator = page.get_by_role(role, name=re.compile(keyword, re.IGNORECASE))
            count = await locator.count()
            if count == 1 and await locator.first.is_visible():
                new_sel = f"role:{role}[name='{keyword}']"
                logger.info("Healed via role match: %s", new_sel)
                return HealResult(success=True, element=locator.first, new_selector=new_sel, strategy="role_match", score=0.75)
    except Exception:
        pass

    return HealResult(success=False)


async def heal_with_label(page: Page, step: PlaybookStep) -> HealResult:
    """Strategy 4: Match by associated label (for form fields)."""
    if step.action not in ("fill", "select", "type"):
        return HealResult(success=False)

    keywords = _extract_intent_keywords(step.intent)
    for keyword in keywords:
        try:
            locator = page.get_by_label(re.compile(keyword, re.IGNORECASE))
            count = await locator.count()
            if count == 1 and await locator.first.is_visible():
                new_sel = f"label={keyword}"
                logger.info("Healed via label match: %s", new_sel)
                return HealResult(success=True, element=locator.first, new_selector=new_sel, strategy="label_match", score=0.8)
        except Exception:
            continue

    return HealResult(success=False)


async def heal_with_placeholder(page: Page, step: PlaybookStep) -> HealResult:
    """Strategy 5: Match by placeholder text (for inputs)."""
    if step.action not in ("fill", "type"):
        return HealResult(success=False)

    keywords = _extract_intent_keywords(step.intent)
    for keyword in keywords:
        try:
            locator = page.get_by_placeholder(re.compile(keyword, re.IGNORECASE))
            count = await locator.count()
            if count == 1 and await locator.first.is_visible():
                new_sel = f"placeholder={keyword}"
                logger.info("Healed via placeholder match: %s", new_sel)
                return HealResult(success=True, element=locator.first, new_selector=new_sel, strategy="placeholder_match", score=0.8)
        except Exception:
            continue

    return HealResult(success=False)


async def heal_with_locator_engine(page: Page, step: PlaybookStep) -> HealResult:
    """Strategy 6: Use Playwright's built-in locator engine for resilient matching.

    Playwright's locator engine combines multiple signals (role, text, test-id,
    CSS) to find elements resiliently. This is what Playwright MCP's
    browser_generate_locator uses internally.
    """
    # Try get_by_role with name from intent keywords
    role = _infer_role(step)
    keywords = _extract_intent_keywords(step.intent)

    # Strategy 6a: Playwright's get_by_test_id (if available)
    for keyword in keywords:
        try:
            locator = page.get_by_test_id(keyword)
            count = await locator.count()
            if count == 1 and await locator.first.is_visible():
                new_sel = f"data-testid={keyword}"
                logger.info("Healed via test-id: %s", new_sel)
                return HealResult(success=True, element=locator.first, new_selector=new_sel, strategy="test_id", score=0.95)
        except Exception:
            continue

    # Strategy 6b: Playwright's get_by_alt_text (for images/buttons with alt)
    if step.action == "click":
        for keyword in keywords:
            try:
                locator = page.get_by_alt_text(re.compile(keyword, re.IGNORECASE))
                count = await locator.count()
                if count == 1 and await locator.first.is_visible():
                    new_sel = f"alt={keyword}"
                    logger.info("Healed via alt text: %s", new_sel)
                    return HealResult(success=True, element=locator.first, new_selector=new_sel, strategy="alt_text", score=0.8)
            except Exception:
                continue

    # Strategy 6c: Playwright's get_by_title
    for keyword in keywords:
        try:
            locator = page.get_by_title(re.compile(keyword, re.IGNORECASE))
            count = await locator.count()
            if count == 1 and await locator.first.is_visible():
                new_sel = f"title={keyword}"
                logger.info("Healed via title: %s", new_sel)
                return HealResult(success=True, element=locator.first, new_selector=new_sel, strategy="title", score=0.75)
        except Exception:
            continue

    # Strategy 6d: Composite locator — chain role + text for precision
    if role and keywords:
        for keyword in keywords:
            try:
                role_locator = page.get_by_role(role)
                filtered = role_locator.filter(has_text=re.compile(keyword, re.IGNORECASE))
                count = await filtered.count()
                if count == 1 and await filtered.first.is_visible():
                    new_sel = f"role:{role}+text:{keyword}"
                    logger.info("Healed via composite role+text: %s", new_sel)
                    return HealResult(success=True, element=filtered.first, new_selector=new_sel, strategy="composite_locator", score=0.85)
            except Exception:
                continue

    # Strategy 6e: Nearest locator by DOM proximity — find visible elements
    # of the expected type and score by attribute overlap with old selector
    old_attrs = _extract_selector_attrs(step.selector)
    if old_attrs:
        try:
            candidates = await _find_candidates_by_attrs(page, step, old_attrs)
            if candidates:
                best = candidates[0]
                logger.info("Healed via attribute proximity: %s (score=%.2f)", best.new_selector, best.score)
                return best
        except Exception:
            pass

    return HealResult(success=False)


async def heal(page: Page, step: PlaybookStep) -> HealResult:
    """Run all heuristic healing strategies in order.

    Returns the first successful result. All strategies are zero-token (no LLM).
    Order: alternatives → label → placeholder → role → text → locator engine
    (most reliable first, Playwright locator engine as comprehensive fallback).
    """
    strategies = [
        ("alternatives", heal_with_alternatives),
        ("label", heal_with_label),
        ("placeholder", heal_with_placeholder),
        ("role", heal_with_role),
        ("text", heal_with_text_match),
        ("locator_engine", heal_with_locator_engine),
    ]

    for name, strategy_fn in strategies:
        result = await strategy_fn(page, step)
        if result.success:
            logger.info("Tier 1 healed step '%s' using strategy '%s'", step.intent, name)
            return result

    logger.info("Tier 1 exhausted all strategies for step '%s'", step.intent)
    return HealResult(success=False)


# --- Helper functions ---

_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "click", "fill", "type", "enter", "select", "press", "tap",
    "input", "field", "button", "link", "form", "page",
})


def _extract_intent_keywords(intent: str) -> list[str]:
    """Extract meaningful keywords from an intent string."""
    words = re.findall(r"\b[a-zA-Z]{2,}\b", intent.lower())
    keywords = [w for w in words if w not in _STOP_WORDS]
    # Return longest words first (more specific)
    return sorted(keywords, key=len, reverse=True)


def _extract_selector_attrs(selector: str) -> dict[str, str]:
    """Extract attribute hints from a CSS selector for proximity matching."""
    attrs: dict[str, str] = {}

    # Extract id: #foo
    id_match = re.search(r"#([\w-]+)", selector)
    if id_match:
        attrs["id_hint"] = id_match.group(1)

    # Extract name: [name='foo'] or input[name="foo"]
    name_match = re.search(r"\[name=['\"]?(\w+)['\"]?\]", selector)
    if name_match:
        attrs["name"] = name_match.group(1)

    # Extract type: [type='email'] or input[type="text"]
    type_match = re.search(r"\[type=['\"]?(\w+)['\"]?\]", selector)
    if type_match:
        attrs["type"] = type_match.group(1)

    # Extract tag
    tag_match = re.match(r"^(\w+)", selector)
    if tag_match and tag_match.group(1) not in ("role", "text"):
        attrs["tag"] = tag_match.group(1)

    # Extract class hints: .foo-bar
    class_match = re.findall(r"\.([\w-]+)", selector)
    if class_match:
        attrs["class_hints"] = " ".join(class_match[:3])

    return attrs


async def _find_candidates_by_attrs(
    page: Page,
    step: PlaybookStep,
    old_attrs: dict[str, str],
) -> list[HealResult]:
    """Find candidate elements by matching attributes from the old selector."""
    tag = old_attrs.get("tag", "*")
    if tag in ("div", "span", "*"):
        # Too generic — narrow by action
        if step.action in ("fill", "type"):
            tag = "input,textarea"
        elif step.action == "click":
            tag = "button,a,input[type='submit']"
        elif step.action == "select":
            tag = "select"

    js = f"""
    () => {{
        const candidates = document.querySelectorAll('{tag}');
        return Array.from(candidates).slice(0, 30).map((el, i) => {{
            const rect = el.getBoundingClientRect();
            const visible = rect.width > 0 && rect.height > 0;
            return {{
                index: i,
                visible: visible,
                id: el.id || '',
                name: el.name || '',
                type: el.type || '',
                tag: el.tagName.toLowerCase(),
                classes: el.className || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                placeholder: el.getAttribute('placeholder') || '',
                text: (el.textContent || '').trim().slice(0, 50),
            }};
        }}).filter(e => e.visible);
    }}
    """

    try:
        elements = await page.evaluate(js)
    except Exception:
        return []

    # Score each candidate against old attributes
    results: list[tuple[float, dict]] = []
    for el in elements:
        score = 0.0

        # ID similarity (partial match — old id may have been e.g. "email", new is "user-email")
        if old_attrs.get("id_hint"):
            hint = old_attrs["id_hint"].lower()
            if hint in el.get("id", "").lower():
                score += 3.0
            elif hint in el.get("name", "").lower():
                score += 2.0

        # Name match
        if old_attrs.get("name") and old_attrs["name"].lower() in el.get("name", "").lower():
            score += 2.5

        # Type match
        if old_attrs.get("type") and old_attrs["type"] == el.get("type", ""):
            score += 1.5

        # Class hint overlap
        if old_attrs.get("class_hints"):
            old_classes = set(old_attrs["class_hints"].split())
            new_classes = set(el.get("classes", "").split())
            overlap = old_classes & new_classes
            score += len(overlap) * 1.0

        if score > 1.0:
            results.append((score, el))

    results.sort(key=lambda x: x[0], reverse=True)

    heal_results = []
    for score, el in results[:3]:
        # Build a selector for this candidate
        if el.get("id"):
            sel = f"#{el['id']}"
        elif el.get("name"):
            sel = f"{el['tag']}[name='{el['name']}']"
        elif el.get("placeholder"):
            sel = f"{el['tag']}[placeholder='{el['placeholder']}']"
        else:
            continue

        locator = await try_selector(page, sel, timeout=2000)
        if locator:
            heal_results.append(HealResult(
                success=True,
                element=locator,
                new_selector=sel,
                strategy="attr_proximity",
                score=min(score / 5.0, 1.0),  # Normalize to 0-1
            ))

    return heal_results


def _infer_role(step: PlaybookStep) -> str | None:
    """Infer ARIA role from step action and selector."""
    sel_lower = step.selector.lower()

    if step.action == "click":
        if "button" in sel_lower or "btn" in sel_lower or "submit" in sel_lower:
            return "button"
        if "a[" in sel_lower or "href" in sel_lower or "link" in sel_lower:
            return "link"
        if "checkbox" in sel_lower:
            return "checkbox"
        return "button"  # Default for clicks

    if step.action in ("fill", "type"):
        if "textarea" in sel_lower:
            return "textbox"
        return "textbox"

    if step.action == "select":
        return "combobox"

    return None
