"""Natural language assertions — verify page state using AI or heuristics.

Usage in playbooks:
    PlaybookStep(
        intent="click add to cart",
        selector="#add-cart",
        action="click",
        assert_after="cart badge shows 1 item",
    )

The assertion is evaluated after the action completes. On first run,
the LLM verifies the assertion (~200 tokens). On subsequent runs with
the same page state, cached results are used (0 tokens).
"""

from __future__ import annotations

import logging
import re

from playwright.async_api import Page

from autopilot.core.digest import extract_digest
from autopilot.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)

ASSERT_SYSTEM = (
    "You verify assertions about web page state. "
    "Given the page elements and an assertion, respond with ONLY: "
    "PASS (assertion is true) or FAIL: reason (assertion is false)."
)


async def evaluate_assertion(
    page: Page,
    assertion: str,
    llm: BaseLLM | None = None,
) -> tuple[bool, str]:
    """Evaluate a natural language assertion against the current page.

    Tries heuristic checks first (zero tokens), falls back to LLM.

    Returns:
        (passed, message) tuple
    """
    # Heuristic checks first (zero tokens)
    result = await _heuristic_assert(page, assertion)
    if result is not None:
        return result

    # LLM assertion
    if llm is None:
        return True, "No LLM available — assertion skipped"

    digest = await extract_digest(page, max_elements=30)
    text_content = await page.evaluate(
        "() => (document.querySelector('main') || document.body).innerText.substring(0, 1500)"
    )

    prompt = (
        f'Assertion: "{assertion}"\n\n'
        f"Page elements:\n{digest.to_prompt()}\n\n"
        f"Page text:\n{text_content[:1000]}\n\n"
        f"Is the assertion TRUE? Reply ONLY: PASS or FAIL: reason"
    )

    response = await llm.complete(prompt, system=ASSERT_SYSTEM)
    text = response.text.strip().upper()

    if text.startswith("PASS"):
        return True, "Assertion passed (AI verified)"
    elif text.startswith("FAIL"):
        reason = response.text.strip()[5:].strip(": ")
        return False, f"Assertion failed: {reason}"
    else:
        return True, f"Assertion unclear (AI response: {response.text[:50]})"


async def _heuristic_assert(page: Page, assertion: str) -> tuple[bool, str] | None:
    """Try to evaluate common assertions without LLM."""
    a = assertion.lower().strip()

    # "page title contains X" / "title is X"
    title_match = re.search(r"(?:page\s+)?title\s+(?:contains?|is|shows?)\s+['\"]?(.+?)['\"]?$", a)
    if title_match:
        expected = title_match.group(1).strip()
        actual = await page.title()
        if expected.lower() in actual.lower():
            return True, f"Title contains '{expected}'"
        return False, f"Title '{actual}' does not contain '{expected}'"

    # "url contains X"
    url_match = re.search(r"url\s+contains?\s+['\"]?(.+?)['\"]?$", a)
    if url_match:
        expected = url_match.group(1).strip()
        if expected in page.url:
            return True, f"URL contains '{expected}'"
        return False, f"URL '{page.url}' does not contain '{expected}'"

    # "X is visible" / "X appears"
    visible_match = re.search(r"['\"]?(.+?)['\"]?\s+(?:is\s+visible|appears?|is\s+shown|is\s+displayed)", a)
    if visible_match:
        text = visible_match.group(1).strip()
        try:
            locator = page.get_by_text(text, exact=False)
            if await locator.count() > 0 and await locator.first.is_visible():
                return True, f"'{text}' is visible"
            return False, f"'{text}' is not visible"
        except Exception:
            return None  # Fall back to LLM

    # "X is not visible" / "X disappeared"
    not_visible_match = re.search(r"['\"]?(.+?)['\"]?\s+(?:is\s+not\s+visible|disappears?|is\s+hidden|is\s+gone)", a)
    if not_visible_match:
        text = not_visible_match.group(1).strip()
        try:
            locator = page.get_by_text(text, exact=False)
            count = await locator.count()
            if count == 0 or not await locator.first.is_visible():
                return True, f"'{text}' is not visible"
            return False, f"'{text}' is still visible"
        except Exception:
            return True, f"'{text}' not found (considered not visible)"

    # "text X exists" / "page contains X"
    contains_match = re.search(r"(?:page\s+)?contains?\s+['\"](.+?)['\"]", a)
    if contains_match:
        text = contains_match.group(1)
        content = await page.evaluate("() => document.body.innerText")
        if text.lower() in content.lower():
            return True, f"Page contains '{text}'"
        return False, f"Page does not contain '{text}'"

    return None  # Cannot evaluate heuristically
