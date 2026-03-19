"""MCP Server — 6 tools for token-efficient browser automation.

Run as:
    autopilot-mcp                              # stdio transport (Claude Code/Desktop)
    autopilot-mcp --transport sse --port 8080  # SSE transport (Cursor/Windsurf)

Add to .mcp.json:
    {
      "mcpServers": {
        "autopilot": {
          "command": "autopilot-mcp",
          "args": ["--llm", "claude-sonnet-4-20250514"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Global state for the MCP server
_browser = None
_runner = None
_playbook_dir = Path.home() / ".autopilot" / "playbooks"
_recording_session = None


def _get_mcp():
    """Lazy import MCP SDK."""
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError:
        raise ImportError(
            "MCP SDK required. Install with: pip install playwright-autopilot[mcp]"
        )


def create_server(
    llm_model: str | None = None,
    headless: bool = True,
    disk_snapshots: bool = False,
) -> Any:
    """Create and configure the MCP server with 7 tools."""
    FastMCP = _get_mcp()

    mode_note = ""
    if disk_snapshots:
        mode_note = (
            " Disk snapshot mode is ON — snapshots are saved to disk and only "
            "a file path is returned. Read the file only when you need to inspect "
            "the page. This saves ~99% of tokens compared to inline snapshots."
        )

    mcp = FastMCP(
        "playwright-autopilot",
        instructions=(
            "Token-efficient browser automation with tiered AI escalation. "
            "Use autopilot_run for goals, autopilot_replay for known flows. "
            "Playbooks drift toward determinism — first run uses AI, subsequent runs are free."
            + mode_note
        ),
    )

    # ---- Tool 1: autopilot_run ----
    @mcp.tool()
    async def autopilot_run(
        url: str,
        goal: str,
        variables: dict[str, str] | None = None,
        max_tier: int = 4,
    ) -> str:
        """Execute a browser goal with automatic tiered escalation.

        Tries cached playbook first (0 tokens), escalates to AI only when needed.
        First run records a playbook; subsequent runs replay it near-free.

        Args:
            url: Starting URL
            goal: Natural language description of the goal
            variables: Template variables (e.g., {"email": "user@test.com"})
            max_tier: Max escalation tier (0=replay-only, 1=+heal, 2=+compact AI, 3=+full AI, 4=+vision)
        """
        browser, runner = await _ensure_browser(llm_model, headless, max_tier)

        # Check for existing playbook
        playbook = _find_playbook(url, goal)
        if playbook:
            result = await runner.run(playbook, variables=variables or {})
            if result.playbook_updated:
                playbook.save(_playbook_path(playbook.name))
            return json.dumps(result.summary, indent=2)

        # No playbook — run in learning mode
        return json.dumps({
            "status": "no_playbook",
            "message": f"No playbook found for goal: {goal}. Use autopilot_record to create one.",
            "url": url,
        })

    # ---- Tool 2: autopilot_step ----
    @mcp.tool()
    async def autopilot_step(
        intent: str,
        action: str = "click",
        selector: str | None = None,
        value: str | None = None,
    ) -> str:
        """Execute a single browser step with tiered escalation.

        If selector is provided, tries it first (Tier 0). Falls back through
        healing tiers automatically. Use this for granular control.

        Args:
            intent: What this step does (e.g., "fill email field")
            action: Action type: click, fill, select, press, check, hover, clear
            selector: Optional CSS/role selector to try first
            value: Value for fill/select/press actions
        """
        from autopilot.core.playbook import PlaybookStep

        browser, runner = await _ensure_browser(llm_model, headless)
        page = browser.page

        step = PlaybookStep(
            intent=intent,
            selector=selector or "",
            action=action,
            value=value,
        )

        result = await runner._execute_step(page, step, 0, {})
        return json.dumps({
            "success": result.success,
            "tier": result.tier,
            "selector_used": result.selector_used,
            "strategy": result.strategy,
            "tokens_used": result.tokens_used,
            "duration_ms": round(result.duration_ms, 1),
        }, indent=2)

    # ---- Tool 3: autopilot_record ----
    @mcp.tool()
    async def autopilot_record(
        url: str,
        name: str,
    ) -> str:
        """Start recording a new playbook by navigating to a URL.

        After calling this, use autopilot_step to add steps.
        When done, call autopilot_save to persist the playbook.

        Args:
            url: Starting URL to navigate to
            name: Name for this playbook (used for saving/loading)
        """
        global _recording_session
        from autopilot.core.playbook import Playbook

        browser, _ = await _ensure_browser(llm_model, headless)
        await browser.page.goto(url, wait_until="domcontentloaded")

        _recording_session = Playbook(name=name, url=url)

        return json.dumps({
            "status": "recording",
            "name": name,
            "url": url,
            "message": "Recording started. Use autopilot_step to add steps, then autopilot_save to persist.",
        })

    # ---- Tool 4: autopilot_replay ----
    @mcp.tool()
    async def autopilot_replay(
        playbook_name: str,
        variables: dict[str, str] | None = None,
    ) -> str:
        """Replay a saved playbook. Near-zero token cost.

        Variables substitute templated values (e.g., ${email}, ${password}).
        Returns step-by-step results and total token usage.

        Args:
            playbook_name: Name of the saved playbook
            variables: Template variable substitutions
        """
        from autopilot.core.playbook import Playbook

        path = _playbook_path(playbook_name)
        if not path.exists():
            available = [p.stem for p in _playbook_dir.glob("*.json")]
            return json.dumps({
                "error": f"Playbook '{playbook_name}' not found",
                "available": available,
            })

        browser, runner = await _ensure_browser(llm_model, headless, max_tier=4)
        playbook = Playbook.load(path)
        result = await runner.run(playbook, variables=variables or {})

        if result.playbook_updated:
            playbook.save(path)

        return json.dumps(result.summary, indent=2)

    # ---- Tool 5: autopilot_heal ----
    @mcp.tool()
    async def autopilot_heal(
        playbook_name: str,
        step_index: int,
    ) -> str:
        """Self-heal a broken step in a playbook.

        Tries heuristic healing first (0 tokens), escalates to AI if needed.
        Updates the playbook with the new selector on success.

        Args:
            playbook_name: Name of the saved playbook
            step_index: Index of the broken step (0-based)
        """
        from autopilot.core.playbook import Playbook
        from autopilot.core.healer import heal

        path = _playbook_path(playbook_name)
        if not path.exists():
            return json.dumps({"error": f"Playbook '{playbook_name}' not found"})

        playbook = Playbook.load(path)
        if step_index >= len(playbook.steps):
            return json.dumps({"error": f"Step index {step_index} out of range (max {len(playbook.steps) - 1})"})

        browser, _ = await _ensure_browser(llm_model, headless)
        step = playbook.steps[step_index]

        # Navigate to playbook URL first
        await browser.page.goto(playbook.url, wait_until="domcontentloaded")

        result = await heal(browser.page, step)

        if result.success:
            playbook.update_step(step_index, selector=result.new_selector)
            playbook.save(path)

        return json.dumps({
            "success": result.success,
            "strategy": result.strategy,
            "new_selector": result.new_selector,
            "score": result.score,
            "tokens_used": 0,  # Tier 1 is always free
        }, indent=2)

    # ---- Tool 6: autopilot_snapshot ----
    @mcp.tool()
    async def autopilot_snapshot(
        format: str = "compact",
    ) -> str:
        """Get current page state in various formats.

        In disk-snapshot mode, returns a file path instead of inline content.
        The agent reads the file only when needed — saving ~99% of tokens.

        Use 'compact' for minimal token usage (~100 tokens), 'full' for
        complete accessibility tree (~5K tokens), or 'screenshot' for PNG image.

        Args:
            format: Output format — 'compact' (~100 tokens), 'full' (~5K tokens), or 'screenshot'
        """
        from autopilot.core.digest import extract_digest
        from autopilot.core.snapshot import extract_snapshot
        from autopilot.core.vision import capture_for_vision

        browser, _ = await _ensure_browser(llm_model, headless)
        page = browser.page

        # Disk snapshot mode — write to file, return path only
        if disk_snapshots:
            from autopilot.core.disk_snapshots import DiskSnapshotManager
            snap_mgr = DiskSnapshotManager()

            if format == "compact":
                path = await snap_mgr.save_digest(page)
                return f"Digest saved to: {path}\nRead this file to see interactive elements on the page."
            elif format == "full":
                path = await snap_mgr.save_snapshot(page)
                return f"Accessibility tree saved to: {path}\nRead this file to see the full page structure."
            elif format == "screenshot":
                path = await snap_mgr.save_screenshot(page)
                return f"Screenshot saved to: {path}\nView this file to see the page visually."
            else:
                return json.dumps({"error": f"Unknown format: {format}."})

        # Inline mode (default) — return content directly
        if format == "compact":
            digest = await extract_digest(page)
            return digest.to_prompt()

        elif format == "full":
            snapshot = await extract_snapshot(page, interactive_only=True)
            return f"Page: {snapshot.title} ({snapshot.url})\n\n{snapshot.tree}"

        elif format == "screenshot":
            capture = await capture_for_vision(page, include_accessibility=True)
            return json.dumps({
                "url": capture.url,
                "title": capture.title,
                "screenshot_b64": capture.screenshot_b64,
                "accessibility_summary": capture.accessibility_summary,
            })

        else:
            return json.dumps({"error": f"Unknown format: {format}. Use 'compact', 'full', or 'screenshot'."})

    # ---- Tool 7: autopilot_explore ----
    @mcp.tool()
    async def autopilot_explore(
        url: str,
        goal: str,
        variables: dict[str, str] | None = None,
        max_steps: int = 30,
    ) -> str:
        """AI explores a website to accomplish a goal, recording a playbook.

        Give it a URL and a natural language goal — the AI figures out the steps,
        executes them, and records a playbook for free replay on subsequent runs.

        Args:
            url: Starting URL
            goal: Natural language description of what to accomplish
            variables: Known values to templatize (e.g., {"email": "user@test.com"})
            max_steps: Maximum steps before giving up
        """
        from autopilot.core.explorer import Explorer

        browser, _ = await _ensure_browser(llm_model, headless)

        if not llm_model:
            return json.dumps({"error": "explore requires an LLM. Start with --llm claude-sonnet-4-20250514"})

        from autopilot.llm.factory import create_llm
        llm = create_llm(llm_model)
        explorer = Explorer(browser=browser, llm=llm, max_steps=max_steps)

        result = await explorer.explore(url=url, goal=goal, variables=variables)

        # Save playbook
        if result.playbook.steps:
            result.playbook.save(_playbook_path(result.playbook.name))

        return json.dumps(result.summary, indent=2)

    return mcp


# ---- Helpers ----

async def _ensure_browser(
    llm_model: str | None = None,
    headless: bool = True,
    max_tier: int = 4,
):
    """Ensure browser and runner are initialized."""
    global _browser, _runner

    from autopilot.core.browser import Browser, BrowserConfig
    from autopilot.core.runner import Runner
    from autopilot.llm.factory import create_llm

    if _browser is None:
        config = BrowserConfig(headless=headless)
        _browser = Browser(config)
        await _browser.start()

    if _runner is None:
        llm = create_llm(llm_model) if llm_model else None
        _runner = Runner(browser=_browser, llm=llm, max_tier=max_tier)

    return _browser, _runner


def _playbook_path(name: str) -> Path:
    """Get the file path for a named playbook."""
    _playbook_dir.mkdir(parents=True, exist_ok=True)
    return _playbook_dir / f"{name}.json"


def _find_playbook(url: str, goal: str):
    """Find a matching playbook for a URL/goal combination."""
    from autopilot.core.playbook import Playbook

    if not _playbook_dir.exists():
        return None

    for path in _playbook_dir.glob("*.json"):
        try:
            pb = Playbook.load(path)
            if pb.url == url:
                return pb
        except Exception:
            continue
    return None


def main():
    """CLI entry point for the MCP server."""
    parser = argparse.ArgumentParser(prog="autopilot-mcp")
    parser.add_argument("--llm", default=None, help="LLM model (e.g., claude-sonnet-4-20250514, gpt-4o)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser headlessly")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="MCP transport")
    parser.add_argument("--port", type=int, default=8080, help="Port for SSE transport")
    parser.add_argument(
        "--disk-snapshots", action="store_true",
        help="Save snapshots to disk instead of returning inline. "
             "Returns file paths (~10 tokens) instead of content (~5K tokens). "
             "Recommended for coding agents with filesystem access.",
    )
    args = parser.parse_args()

    headless = not args.headed

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,  # CRITICAL: never print to stdout in stdio mode
    )

    mcp = create_server(llm_model=args.llm, headless=headless, disk_snapshots=args.disk_snapshots)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
