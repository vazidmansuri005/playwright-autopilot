"""High-level Agent API — the 5-line quickstart.

Usage:
    from autopilot import Agent

    # Explore: give a goal, AI figures out the steps
    agent = Agent(llm="claude-sonnet-4-20250514")
    result = await agent.explore(
        url="https://example.com",
        goal="Log in with user@test.com / password123",
    )

    # Run: replay saved playbook (near-zero tokens)
    result = await agent.run("example-login", variables={...})

    # Extract: pull structured data from any page
    data = await agent.extract(
        url="https://news.ycombinator.com",
        goal="top 5 stories",
        schema={"title": "str", "points": "int"},
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import RunResult, Runner
from autopilot.llm.base import BaseLLM, LLMUsageTracker
from autopilot.llm.factory import create_llm

logger = logging.getLogger(__name__)


class Agent:
    """High-level browser automation agent.

    Three modes:
    - explore(): AI figures out the steps, records a playbook
    - run(): Replay a playbook with tiered escalation
    - extract(): Pull structured data from any page
    """

    def __init__(
        self,
        llm: str | BaseLLM | None = None,
        headless: bool = True,
        browser_type: str = "chromium",
        max_tier: int = 4,
        playbook_dir: str | Path | None = None,
        timeout: int = 30_000,
        browser_url: str | None = None,
    ):
        """Initialize the Agent.

        Args:
            llm: Model string ("claude-sonnet-4-20250514", "gpt-4o", etc.),
                 BaseLLM instance, or None for Tier 0-1 only mode.
            headless: Run browser in headless mode.
            browser_type: Browser engine — "chromium", "firefox", or "webkit".
            max_tier: Maximum escalation tier (0-4).
            playbook_dir: Directory to store/load playbooks. Defaults to ~/.autopilot/playbooks.
            timeout: Default timeout in ms for browser operations.
            browser_url: Connect to remote browser via CDP/WebSocket URL instead
                of launching locally. Works with any cloud browser provider or local CDP.
                Example: "ws://localhost:9222" or "wss://your-cloud-provider.com/..."
        """
        if isinstance(llm, str):
            self._llm = create_llm(llm)
        elif isinstance(llm, BaseLLM):
            self._llm = llm
        else:
            self._llm = None

        self._browser_config = BrowserConfig(
            headless=headless,
            browser_type=browser_type,
            timeout=timeout,
        )
        self._browser_url = browser_url
        self._max_tier = max_tier
        self._playbook_dir = Path(playbook_dir) if playbook_dir else Path.home() / ".autopilot" / "playbooks"
        self._playbook_dir.mkdir(parents=True, exist_ok=True)

        self._browser: Browser | None = None
        self._runner: Runner | None = None
        self.usage = LLMUsageTracker()

    async def explore(
        self,
        url: str,
        goal: str,
        name: str | None = None,
        variables: dict[str, str] | None = None,
        max_steps: int = 30,
        save: bool = True,
    ):
        """AI explores a website to accomplish a goal, recording a playbook.

        First run: AI figures out the steps (~500 tokens per step).
        Subsequent runs: use agent.run() to replay for free.

        Args:
            url: Starting URL
            goal: Natural language goal (e.g., "Log in and go to settings")
            name: Playbook name (auto-generated from goal if not provided)
            variables: Known values to templatize (e.g., {"email": "user@test.com"})
            max_steps: Maximum steps before giving up
            save: Save the recorded playbook to disk

        Returns:
            ExploreResult with the recorded playbook and token usage
        """
        from autopilot.core.explorer import Explorer

        if self._llm is None:
            raise ValueError("explore() requires an LLM. Pass llm='claude-sonnet-4-20250514' or similar.")

        browser = await self._ensure_browser()
        explorer = Explorer(browser=browser, llm=self._llm, max_steps=max_steps)

        result = await explorer.explore(url=url, goal=goal, name=name, variables=variables)

        # Aggregate usage
        self.usage.total_input_tokens += explorer.usage.total_input_tokens
        self.usage.total_output_tokens += explorer.usage.total_output_tokens
        self.usage.call_count += explorer.usage.call_count
        self.usage.calls.extend(explorer.usage.calls)

        # Save playbook
        if save and result.playbook.steps:
            path = self._playbook_dir / f"{result.playbook.name}.json"
            result.playbook.save(path)
            logger.info("Playbook saved: %s (%d steps)", path, len(result.playbook.steps))

        return result

    async def extract(
        self,
        url: str,
        goal: str,
        schema: dict[str, str] | None = None,
    ) -> list[dict] | str:
        """Extract structured data from any web page using AI.

        Args:
            url: Page URL
            goal: What to extract (e.g., "top 5 stories with title and points")
            schema: Optional schema hint (e.g., {"title": "str", "points": "int"})

        Returns:
            Extracted data as list of dicts (if parseable) or raw text
        """
        from autopilot.core.explorer import Explorer

        if self._llm is None:
            raise ValueError("extract() requires an LLM. Pass llm='claude-sonnet-4-20250514' or similar.")

        browser = await self._ensure_browser()
        explorer = Explorer(browser=browser, llm=self._llm)

        result = await explorer.extract(url=url, goal=goal, schema=schema)

        self.usage.total_input_tokens += explorer.usage.total_input_tokens
        self.usage.total_output_tokens += explorer.usage.total_output_tokens
        self.usage.call_count += explorer.usage.call_count
        self.usage.calls.extend(explorer.usage.calls)

        return result

    async def run(
        self,
        playbook: str | Path | Playbook,
        variables: dict[str, str] | None = None,
        save: bool = True,
    ) -> RunResult:
        """Execute a playbook with tiered escalation.

        Args:
            playbook: Path to playbook JSON, playbook name, or Playbook object.
            variables: Template variable substitutions.
            save: Save playbook updates after healing.

        Returns:
            RunResult with per-step details and token usage.
        """
        pb = self._resolve_playbook(playbook)
        browser = await self._ensure_browser()
        runner = self._ensure_runner(browser)

        result = await runner.run(pb, variables=variables or {})

        # Aggregate usage
        for call in runner.usage.calls:
            self.usage.calls.append(call)
        self.usage.total_input_tokens += runner.usage.total_input_tokens
        self.usage.total_output_tokens += runner.usage.total_output_tokens
        self.usage.call_count += runner.usage.call_count

        # Save updated playbook
        if save and result.playbook_updated:
            if isinstance(playbook, (str, Path)):
                pb.save(self._resolve_playbook_path(playbook))
            else:
                # Save to default location using playbook name
                pb.save(self._playbook_dir / f"{pb.name}.json")
            logger.info("Playbook saved after healing")

        return result

    async def run_chain(
        self,
        playbooks: list[str | Path | Playbook],
        variables: dict[str, str] | None = None,
    ) -> list[RunResult]:
        """Run multiple playbooks in sequence, sharing the same browser session.

        Each playbook picks up where the last left off (same page state).
        If any playbook fails, the chain stops.

        Args:
            playbooks: List of playbook names, paths, or objects
            variables: Shared variables across all playbooks

        Returns:
            List of RunResult, one per playbook
        """
        results = []
        for pb in playbooks:
            result = await self.run(pb, variables=variables)
            results.append(result)
            if not result.success:
                logger.warning("Chain stopped: playbook '%s' failed", pb)
                break
        return results

    async def run_parallel(
        self,
        tasks: list[tuple[str | Path | Playbook, dict[str, str]]],
        max_concurrent: int = 5,
    ) -> list[RunResult]:
        """Run multiple playbooks in parallel with separate browser instances.

        Each task gets its own browser. Useful for load testing or
        running the same flow with different credentials.

        Args:
            tasks: List of (playbook, variables) tuples
            max_concurrent: Max simultaneous browsers

        Returns:
            List of RunResult, one per task
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[RunResult | None] = [None] * len(tasks)

        async def run_task(index: int, pb, variables: dict):
            async with semaphore:
                agent = Agent(
                    llm=self._llm,
                    headless=self._browser_config.headless,
                    browser_type=self._browser_config.browser_type,
                    max_tier=self._max_tier,
                    playbook_dir=self._playbook_dir,
                    timeout=self._browser_config.timeout,
                )
                try:
                    async with agent:
                        results[index] = await agent.run(pb, variables=variables)
                except Exception as e:
                    logger.error("Parallel task %d failed: %s", index, e)
                    results[index] = RunResult(success=False)

        await asyncio.gather(*[
            run_task(i, pb, vars) for i, (pb, vars) in enumerate(tasks)
        ])

        return [r for r in results if r is not None]

    async def replay(
        self,
        playbook: str | Path | Playbook,
        variables: dict[str, str] | None = None,
    ) -> RunResult:
        """Replay a playbook at Tier 0 only (zero tokens, deterministic).

        Use this when you know the playbook is up-to-date and want
        maximum speed with zero LLM cost.
        """
        pb = self._resolve_playbook(playbook)
        browser = await self._ensure_browser()
        runner = Runner(browser=browser, llm=None, max_tier=0)
        return await runner.run(pb, variables=variables or {})

    def list_playbooks(self) -> list[dict]:
        """List all saved playbooks with metadata."""
        result = []
        for path in self._playbook_dir.glob("*.json"):
            try:
                pb = Playbook.load(path)
                result.append({
                    "name": pb.name,
                    "url": pb.url,
                    "steps": len(pb.steps),
                    "run_count": pb.run_count,
                    "success_rate": pb.success_rate,
                    "variables": pb.extract_variables(),
                    "path": str(path),
                })
            except Exception:
                continue
        return result

    def load_playbook(self, name: str) -> Playbook:
        """Load a playbook by name."""
        path = self._playbook_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Playbook '{name}' not found at {path}")
        return Playbook.load(path)

    async def close(self):
        """Close the browser and clean up resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._runner = None

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ---- Private ----

    def _resolve_playbook(self, playbook: str | Path | Playbook) -> Playbook:
        if isinstance(playbook, Playbook):
            return playbook
        path = self._resolve_playbook_path(playbook)
        return Playbook.load(path)

    def _resolve_playbook_path(self, playbook: str | Path) -> Path:
        path = Path(playbook)
        if path.exists():
            return path
        # Try in playbook directory
        named_path = self._playbook_dir / f"{playbook}.json"
        if named_path.exists():
            return named_path
        # Try with .json extension
        if not path.suffix:
            path = path.with_suffix(".json")
            if path.exists():
                return path
        raise FileNotFoundError(f"Playbook not found: {playbook}")

    async def _ensure_browser(self) -> Browser:
        if self._browser is None:
            if self._browser_url:
                # Connect to remote browser via CDP/WebSocket
                self._browser = await self._connect_remote(self._browser_url)
            else:
                self._browser = Browser(self._browser_config)
                await self._browser.start()
        return self._browser

    async def _connect_remote(self, url: str) -> Browser:
        """Connect to a remote browser via CDP or WebSocket URL."""
        from playwright.async_api import async_playwright

        browser = Browser(self._browser_config)
        browser._playwright = await async_playwright().start()

        if url.startswith("ws://") or url.startswith("wss://"):
            browser._browser = await browser._playwright.chromium.connect(url)
        else:
            browser._browser = await browser._playwright.chromium.connect_over_cdp(url)

        browser._context = browser._browser.contexts[0] if browser._browser.contexts else await browser._browser.new_context()
        browser._page = browser._context.pages[0] if browser._context.pages else await browser._context.new_page()

        logger.info("Connected to remote browser: %s", url[:60])
        return browser

    def _ensure_runner(self, browser: Browser) -> Runner:
        self._runner = Runner(
            browser=browser,
            llm=self._llm,
            max_tier=self._max_tier,
        )
        return self._runner
