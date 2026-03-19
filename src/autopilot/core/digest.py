"""Tier 2: Compact page representation — ~500 tokens for LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class PageDigest:
    """Compact representation of interactive elements on a page."""
    url: str
    title: str
    elements: list[dict]
    token_estimate: int

    def to_prompt(self) -> str:
        """Format as a compact table for LLM consumption."""
        lines = [
            f"Page: {self.title} ({self.url})",
            "",
            "ref|tag|role|name|type|placeholder|visible",
            "---|---|---|---|---|---|---",
        ]
        for el in self.elements:
            lines.append(
                f"{el['ref']}|{el['tag']}|{el.get('role', '')}|"
                f"{el.get('name', '')}|{el.get('type', '')}|"
                f"{el.get('placeholder', '')}|{el.get('visible', True)}"
            )
        return "\n".join(lines)


# JavaScript to extract interactive elements from page
_EXTRACT_JS = """
() => {
    const INTERACTIVE = [
        'a[href]', 'button', 'input', 'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="textbox"]',
        '[role="checkbox"]', '[role="radio"]', '[role="combobox"]',
        '[role="menuitem"]', '[role="tab"]', '[role="switch"]',
        '[tabindex]:not([tabindex="-1"])',
        '[contenteditable="true"]',
    ].join(', ');

    const elements = document.querySelectorAll(INTERACTIVE);
    const results = [];

    for (let i = 0; i < Math.min(elements.length, MAX_ELEMENTS); i++) {
        const el = elements[i];
        const rect = el.getBoundingClientRect();
        const visible = rect.width > 0 && rect.height > 0 &&
                       window.getComputedStyle(el).visibility !== 'hidden' &&
                       window.getComputedStyle(el).display !== 'none';

        // Get accessible name: aria-label > label[for] > textContent > placeholder
        let name = el.getAttribute('aria-label') || '';
        if (!name) {
            const id = el.getAttribute('id');
            if (id) {
                const label = document.querySelector(`label[for="${id}"]`);
                if (label) name = label.textContent.trim();
            }
        }
        if (!name) {
            name = el.textContent?.trim() || '';
        }
        // Truncate long names
        if (name.length > 50) name = name.substring(0, 47) + '...';

        results.push({
            ref: `e${i}`,
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute('role') || '',
            name: name,
            type: el.getAttribute('type') || '',
            placeholder: el.getAttribute('placeholder') || '',
            visible: visible,
            href: el.tagName === 'A' ? (el.getAttribute('href') || '').substring(0, 80) : '',
            disabled: el.disabled || el.getAttribute('aria-disabled') === 'true',
        });
    }

    return results;
}
"""


async def extract_digest(
    page: Page,
    max_elements: int = 50,
    visible_only: bool = True,
) -> PageDigest:
    """Extract a compact digest of interactive page elements.

    Returns ~500 tokens worth of structured data about the page's
    interactive elements. This is the Tier 2 representation — enough
    for an LLM to identify the right element without seeing the full DOM.

    Args:
        page: Playwright page object
        max_elements: Maximum number of elements to extract
        visible_only: If True, filter to only visible elements

    Returns:
        PageDigest with compact element data
    """
    js = _EXTRACT_JS.replace("MAX_ELEMENTS", str(max_elements))

    try:
        elements = await page.evaluate(js)
    except Exception as e:
        logger.warning("Failed to extract digest: %s", e)
        elements = []

    if visible_only:
        elements = [e for e in elements if e.get("visible", True)]

    # Remove disabled elements (not actionable)
    elements = [e for e in elements if not e.get("disabled", False)]

    # Estimate tokens: ~12 tokens per element row + header
    token_estimate = len(elements) * 12 + 30

    url = page.url
    title = await page.title()

    digest = PageDigest(
        url=url,
        title=title,
        elements=elements,
        token_estimate=token_estimate,
    )

    logger.info(
        "Digest extracted: %d elements, ~%d tokens",
        len(elements), token_estimate,
    )
    return digest


async def extract_focused_digest(
    page: Page,
    intent: str,
    max_elements: int = 20,
) -> PageDigest:
    """Extract elements most relevant to a specific intent.

    Further reduces tokens by scoring elements against the intent
    and returning only the top matches.
    """
    full_digest = await extract_digest(page, max_elements=100, visible_only=True)

    intent_lower = intent.lower()
    intent_words = set(intent_lower.split())

    def relevance_score(el: dict) -> float:
        score = 0.0
        name_lower = el.get("name", "").lower()
        role = el.get("role", "").lower()
        tag = el.get("tag", "").lower()
        placeholder = el.get("placeholder", "").lower()

        # Name match
        for word in intent_words:
            if word in name_lower:
                score += 2.0
            if word in placeholder:
                score += 1.5
            if word in role:
                score += 1.0

        # Action-tag alignment
        if any(w in intent_lower for w in ("click", "press", "tap", "submit")):
            if tag in ("button", "a") or role in ("button", "link"):
                score += 1.0
        if any(w in intent_lower for w in ("fill", "type", "enter", "input")):
            if tag in ("input", "textarea") or role == "textbox":
                score += 1.0
        if "select" in intent_lower or "choose" in intent_lower:
            if tag == "select" or role == "combobox":
                score += 1.0

        return score

    scored = [(relevance_score(el), el) for el in full_digest.elements]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_elements = [el for _, el in scored[:max_elements]]

    token_estimate = len(top_elements) * 12 + 30

    return PageDigest(
        url=full_digest.url,
        title=full_digest.title,
        elements=top_elements,
        token_estimate=token_estimate,
    )
