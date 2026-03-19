"""Browser lifecycle management wrapping Playwright."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from playwright.async_api import (
    Browser as PWBrowser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger(__name__)


@dataclass
class BrowserConfig:
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport: dict | None = None
    user_agent: str | None = None
    timeout: int = 30_000  # ms
    extra_args: list[str] = field(default_factory=list)
    storage_state: str | None = None  # path to saved auth state


class Browser:
    """Manages Playwright browser lifecycle with session persistence."""

    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self._playwright: Playwright | None = None
        self._browser: PWBrowser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def start(self) -> Page:
        self._playwright = await async_playwright().start()

        launcher = getattr(self._playwright, self.config.browser_type)
        self._browser = await launcher.launch(
            headless=self.config.headless,
            args=self.config.extra_args or None,
        )

        context_kwargs: dict[str, Any] = {}
        if self.config.viewport:
            context_kwargs["viewport"] = self.config.viewport
        if self.config.user_agent:
            context_kwargs["user_agent"] = self.config.user_agent
        if self.config.storage_state:
            context_kwargs["storage_state"] = self.config.storage_state

        self._context = await self._browser.new_context(**context_kwargs)
        self._context.set_default_timeout(self.config.timeout)
        self._page = await self._context.new_page()

        logger.info("Browser started: %s headless=%s", self.config.browser_type, self.config.headless)
        return self._page

    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    @property
    def context(self) -> BrowserContext:
        if not self._context:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._context

    async def save_storage_state(self, path: str) -> None:
        await self.context.storage_state(path=path)
        logger.info("Storage state saved to %s", path)

    async def close(self) -> None:
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        logger.info("Browser closed")

    async def __aenter__(self) -> "Browser":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


@asynccontextmanager
async def create_browser(
    headless: bool = True,
    browser_type: str = "chromium",
    **kwargs: Any,
) -> AsyncGenerator[Browser, None]:
    """Convenience context manager for browser lifecycle."""
    config = BrowserConfig(headless=headless, browser_type=browser_type, **kwargs)
    browser = Browser(config)
    try:
        await browser.start()
        yield browser
    finally:
        await browser.close()
