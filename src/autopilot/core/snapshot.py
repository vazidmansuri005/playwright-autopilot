"""Tier 3: Filtered accessibility tree — ~5K tokens for LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class PageSnapshot:
    """Accessibility tree representation of the page."""
    url: str
    title: str
    tree: str
    token_estimate: int


async def extract_snapshot(
    page: Page,
    interactive_only: bool = False,
    max_depth: int = 10,
) -> PageSnapshot:
    """Extract the accessibility tree from the page.

    This is Tier 3 — more detailed than a digest (~5K tokens) but
    far cheaper than a full DOM dump. Uses Playwright's built-in
    accessibility tree API.

    Args:
        page: Playwright page object
        interactive_only: If True, only include interactive elements
        max_depth: Maximum tree depth to traverse
    """
    try:
        # Playwright >= 1.49 uses page.accessibility.snapshot()
        # but some versions may expose it differently
        if hasattr(page, 'accessibility') and hasattr(page.accessibility, 'snapshot'):
            snapshot = await page.accessibility.snapshot()
        else:
            # Fallback: use aria snapshot or evaluate
            snapshot = await _build_snapshot_from_dom(page)
    except Exception as e:
        logger.warning("Failed to get accessibility snapshot: %s, falling back to DOM", e)
        snapshot = await _build_snapshot_from_dom(page)

    if not snapshot:
        return PageSnapshot(
            url=page.url,
            title=await page.title(),
            tree="[Empty accessibility tree]",
            token_estimate=10,
        )

    lines: list[str] = []
    _walk_tree(snapshot, lines, depth=0, max_depth=max_depth, interactive_only=interactive_only)
    tree_text = "\n".join(lines)

    # Rough token estimate: ~1.3 tokens per word
    token_estimate = int(len(tree_text.split()) * 1.3)

    result = PageSnapshot(
        url=page.url,
        title=await page.title(),
        tree=tree_text,
        token_estimate=token_estimate,
    )

    logger.info("Snapshot extracted: %d lines, ~%d tokens", len(lines), token_estimate)
    return result


def _walk_tree(
    node: dict,
    lines: list[str],
    depth: int,
    max_depth: int,
    interactive_only: bool,
    ref_counter: list[int] | None = None,
) -> None:
    """Recursively walk the accessibility tree and format it."""
    if depth > max_depth:
        return

    if ref_counter is None:
        ref_counter = [0]

    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    # Skip generic containers if interactive_only
    is_interactive = role in (
        "button", "link", "textbox", "checkbox", "radio",
        "combobox", "menuitem", "tab", "switch", "slider",
        "searchbox", "spinbutton", "option",
    )

    if interactive_only and not is_interactive and not node.get("children"):
        # Still recurse into children — interactive elements may be nested
        for child in node.get("children", []):
            _walk_tree(child, lines, depth, max_depth, interactive_only, ref_counter)
        return

    indent = "  " * depth
    ref = f"[e{ref_counter[0]}]" if is_interactive else ""
    ref_counter[0] += 1

    parts = [f"{indent}{ref} {role}"]
    if name:
        # Truncate long names
        display_name = name[:80] + "..." if len(name) > 80 else name
        parts.append(f'"{display_name}"')
    if value:
        display_value = value[:40] + "..." if len(value) > 40 else value
        parts.append(f"value={display_value}")

    # Add state hints
    states = []
    if node.get("checked") is not None:
        states.append("checked" if node["checked"] else "unchecked")
    if node.get("disabled"):
        states.append("disabled")
    if node.get("expanded") is not None:
        states.append("expanded" if node["expanded"] else "collapsed")
    if node.get("selected"):
        states.append("selected")
    if states:
        parts.append(f"[{', '.join(states)}]")

    lines.append(" ".join(parts))

    for child in node.get("children", []):
        _walk_tree(child, lines, depth + 1, max_depth, interactive_only, ref_counter)


async def _build_snapshot_from_dom(page: Page) -> dict | None:
    """Build an accessibility-like tree from DOM when native API is unavailable."""
    try:
        tree = await page.evaluate("""
        () => {
            function walk(el, depth) {
                if (depth > 8) return null;
                const tag = el.tagName?.toLowerCase() || '';
                const role = el.getAttribute?.('role') ||
                    ({'BUTTON': 'button', 'A': 'link', 'INPUT': 'textbox',
                     'TEXTAREA': 'textbox', 'SELECT': 'combobox',
                     'H1': 'heading', 'H2': 'heading', 'H3': 'heading',
                     'NAV': 'navigation', 'MAIN': 'main', 'FORM': 'form',
                     'TABLE': 'table', 'TR': 'row', 'TD': 'cell',
                     'IMG': 'img', 'UL': 'list', 'LI': 'listitem',
                    })[el.tagName] || 'generic';

                const name = el.getAttribute?.('aria-label') ||
                    el.getAttribute?.('alt') ||
                    ((['BUTTON', 'A', 'LABEL', 'H1', 'H2', 'H3', 'TD', 'LI', 'OPTION'].includes(el.tagName))
                        ? (el.textContent?.trim().slice(0, 60) || '') : '') ||
                    el.getAttribute?.('placeholder') || '';

                const node = { role, name };

                if (el.type) node.type = el.type;
                if (el.value && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName))
                    node.value = el.value.slice(0, 40);
                if (el.checked !== undefined) node.checked = el.checked;
                if (el.disabled) node.disabled = true;

                const kids = [];
                for (const child of (el.children || [])) {
                    const k = walk(child, depth + 1);
                    if (k) kids.push(k);
                }
                if (kids.length > 0) node.children = kids;
                return node;
            }
            return walk(document.body, 0);
        }
        """)
        return tree
    except Exception:
        return None
