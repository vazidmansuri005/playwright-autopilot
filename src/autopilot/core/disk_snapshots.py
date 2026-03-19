"""Disk-based snapshots — Playwright CLI approach for token efficiency.

Instead of returning page snapshots inline (which bloats the LLM context),
this module writes snapshots to disk and returns only a file path reference.
The agent reads the file only when it needs to.

This is the approach Microsoft recommends for coding agents:
"Do not force page data into the LLM."

Usage in MCP server:
    snap = DiskSnapshotManager(output_dir="~/.autopilot/snapshots")
    path = await snap.save_digest(page)      # Returns path, ~0 tokens in context
    path = await snap.save_snapshot(page)     # Full a11y tree to disk
    path = await snap.save_screenshot(page)   # PNG to disk

The agent decides whether to read the file — most of the time it doesn't need to.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from playwright.async_api import Page

from autopilot.core.digest import extract_digest
from autopilot.core.snapshot import extract_snapshot
from autopilot.core.vision import capture_for_vision

logger = logging.getLogger(__name__)


class DiskSnapshotManager:
    """Manage page snapshots on disk instead of inline in LLM context."""

    def __init__(self, output_dir: str | Path | None = None):
        self._dir = Path(output_dir) if output_dir else Path.home() / ".autopilot" / "snapshots"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def _next_path(self, ext: str) -> Path:
        self._counter += 1
        ts = int(time.time())
        return self._dir / f"snap_{ts}_{self._counter:03d}.{ext}"

    async def save_digest(self, page: Page, max_elements: int = 50) -> Path:
        """Save compact element digest to disk. Returns file path.

        The digest file contains ~500 tokens of interactive element data.
        The agent gets a one-line path reference (~10 tokens) instead.
        """
        digest = await extract_digest(page, max_elements=max_elements)
        path = self._next_path("yaml")

        content = f"# Page: {digest.title} ({digest.url})\n"
        content += f"# Elements: {len(digest.elements)}\n"
        content += f"# Token estimate: ~{digest.token_estimate}\n\n"

        for el in digest.elements:
            visible = "visible" if el.get("visible", True) else "hidden"
            parts = [f"- {el['ref']}: {el['tag']}"]
            if el.get("role"):
                parts.append(f"role={el['role']}")
            if el.get("name"):
                parts.append(f'"{el["name"]}"')
            if el.get("type"):
                parts.append(f"type={el['type']}")
            if el.get("placeholder"):
                parts.append(f'placeholder="{el["placeholder"]}"')
            content += " ".join(parts) + f"  [{visible}]\n"

        path.write_text(content)
        logger.info("Digest saved: %s (%d elements)", path, len(digest.elements))
        return path

    async def save_snapshot(self, page: Page, interactive_only: bool = True) -> Path:
        """Save accessibility tree to disk. Returns file path."""
        snapshot = await extract_snapshot(page, interactive_only=interactive_only)
        path = self._next_path("txt")

        content = f"Page: {snapshot.title} ({snapshot.url})\n"
        content += f"Token estimate: ~{snapshot.token_estimate}\n\n"
        content += snapshot.tree

        path.write_text(content)
        logger.info("Snapshot saved: %s (~%d tokens)", path, snapshot.token_estimate)
        return path

    async def save_screenshot(self, page: Page, full_page: bool = False) -> Path:
        """Save screenshot PNG to disk. Returns file path."""
        path = self._next_path("png")
        await page.screenshot(path=str(path), full_page=full_page)
        size_kb = path.stat().st_size / 1024
        logger.info("Screenshot saved: %s (%.1f KB)", path, size_kb)
        return path

    async def save_all(self, page: Page) -> dict[str, Path]:
        """Save all three formats to disk. Returns dict of paths."""
        return {
            "digest": await self.save_digest(page),
            "snapshot": await self.save_snapshot(page),
            "screenshot": await self.save_screenshot(page),
        }

    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Remove snapshots older than max_age_seconds. Returns count removed."""
        now = time.time()
        removed = 0
        for path in self._dir.iterdir():
            if path.name.startswith("snap_") and (now - path.stat().st_mtime) > max_age_seconds:
                path.unlink()
                removed += 1
        if removed:
            logger.info("Cleaned up %d old snapshots", removed)
        return removed
