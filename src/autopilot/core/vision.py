"""Tier 4: Screenshot + vision fallback — last resort."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class VisionCapture:
    """Screenshot data for vision LLM analysis."""
    url: str
    title: str
    screenshot_b64: str
    viewport_width: int
    viewport_height: int
    accessibility_summary: str  # Compact text summary alongside image
    token_estimate: int


async def capture_for_vision(
    page: Page,
    include_accessibility: bool = True,
    full_page: bool = False,
) -> VisionCapture:
    """Capture screenshot + minimal accessibility context for vision LLM.

    This is Tier 4 — the most expensive tier. Only used when Tiers 0-3
    all fail. Combines a screenshot with a compact text summary so the
    vision LLM has both pixel and semantic information.

    Args:
        page: Playwright page object
        include_accessibility: Include compact accessibility summary
        full_page: Capture full scrollable page (vs viewport only)
    """
    screenshot_bytes = await page.screenshot(full_page=full_page, type="png")
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

    viewport = page.viewport_size or {"width": 1280, "height": 720}

    # Build a compact accessibility summary (just interactive elements)
    a11y_summary = ""
    if include_accessibility:
        a11y_summary = await _compact_accessibility_summary(page)

    # Token estimate: ~1K for the image reference + ~500 for text
    token_estimate = 1500 + int(len(a11y_summary.split()) * 1.3)

    result = VisionCapture(
        url=page.url,
        title=await page.title(),
        screenshot_b64=screenshot_b64,
        viewport_width=viewport["width"],
        viewport_height=viewport["height"],
        accessibility_summary=a11y_summary,
        token_estimate=token_estimate,
    )

    logger.info("Vision capture: %dx%d, ~%d tokens", viewport["width"], viewport["height"], token_estimate)
    return result


async def _compact_accessibility_summary(page: Page) -> str:
    """Generate a very compact text summary of interactive elements."""
    try:
        elements = await page.evaluate("""
        () => {
            const els = document.querySelectorAll(
                'a[href], button, input, select, textarea, [role="button"], [role="link"]'
            );
            return Array.from(els).slice(0, 30).map((el, i) => {
                const rect = el.getBoundingClientRect();
                const name = el.getAttribute('aria-label') || el.textContent?.trim().slice(0, 30) || '';
                return `e${i}: ${el.tagName.toLowerCase()} "${name}" at (${Math.round(rect.x)},${Math.round(rect.y)})`;
            });
        }
        """)
        return "\n".join(elements)
    except Exception:
        return ""


def format_vision_prompt(capture: VisionCapture, intent: str) -> dict:
    """Format a vision capture as a prompt for vision-capable LLMs.

    Returns a dict with 'text' and 'image' keys suitable for
    Claude's vision API or OpenAI's image_url format.
    """
    text = (
        f"I'm trying to: \"{intent}\"\n"
        f"Page: {capture.title} ({capture.url})\n"
        f"Viewport: {capture.viewport_width}x{capture.viewport_height}\n"
    )
    if capture.accessibility_summary:
        text += f"\nInteractive elements:\n{capture.accessibility_summary}\n"
    text += (
        "\nLook at the screenshot. What element should I interact with? "
        "Respond with the element reference (e.g., e5) and the action to take."
    )

    return {
        "text": text,
        "image_b64": capture.screenshot_b64,
        "media_type": "image/png",
    }
