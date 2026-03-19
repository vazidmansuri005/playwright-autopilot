"""Shared fixtures for the test harness."""

from __future__ import annotations

import pytest
import pytest_asyncio

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner
from autopilot.llm.base import MockLLM
from tests.harness.test_server import LocalTestServer


@pytest_asyncio.fixture
async def test_server():
    """Start a local test server for the duration of the test."""
    server = LocalTestServer(port=0)
    async with server:
        yield server


@pytest_asyncio.fixture
async def browser():
    """Create a headless browser instance."""
    config = BrowserConfig(headless=True, timeout=10_000)
    b = Browser(config)
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def browser_at_login(browser, test_server):
    """Browser navigated to the login page."""
    await browser.page.goto(f"{test_server.url}/login")
    return browser


@pytest_asyncio.fixture
async def browser_at_dashboard(browser, test_server):
    """Browser navigated to the dashboard page."""
    await browser.page.goto(f"{test_server.url}/dashboard")
    return browser


@pytest_asyncio.fixture
async def browser_at_settings(browser, test_server):
    """Browser navigated to the settings page."""
    await browser.page.goto(f"{test_server.url}/settings")
    return browser


@pytest.fixture
def mock_llm():
    """LLM that returns predefined responses."""
    return MockLLM()


@pytest.fixture
def login_playbook(test_server) -> Playbook:
    """A playbook for the login flow."""
    return Playbook(
        name="test-login",
        url=f"{test_server.url}/login",
        steps=[
            PlaybookStep(
                intent="fill email field",
                selector="#email",
                selector_alternatives=[
                    "input[name='email']",
                    "input[type='email']",
                    "role:textbox[name='Email']",
                ],
                action="fill",
                value_template="${email}",
            ),
            PlaybookStep(
                intent="fill password field",
                selector="#password",
                selector_alternatives=[
                    "input[name='password']",
                    "input[type='password']",
                ],
                action="fill",
                value_template="${password}",
            ),
            PlaybookStep(
                intent="click login button",
                selector="#login-btn",
                selector_alternatives=[
                    "button[type='submit']",
                    "role:button[name='Log In']",
                ],
                action="click",
            ),
        ],
        variables=["email", "password"],
    )


@pytest.fixture
def login_playbook_mutated(test_server) -> Playbook:
    """Playbook with selectors that DON'T match the mutated page.
    Tests healing capabilities."""
    return Playbook(
        name="test-login-mutated",
        url=f"{test_server.url}/login",
        steps=[
            PlaybookStep(
                intent="fill email field",
                selector="#email",  # Changed to #user-email on mutated page
                selector_alternatives=[
                    "input[name='email']",  # Changed to userEmail
                    "role:textbox[name='Email']",  # Changed to Email Address
                ],
                action="fill",
                value_template="${email}",
            ),
            PlaybookStep(
                intent="fill password field",
                selector="#password",  # Changed to #user-pass
                selector_alternatives=[
                    "input[name='password']",  # Changed to userPass
                ],
                action="fill",
                value_template="${password}",
            ),
            PlaybookStep(
                intent="click login button",
                selector="#login-btn",  # Changed to #sign-in-btn
                selector_alternatives=[
                    "role:button[name='Log In']",  # Changed to Sign In
                ],
                action="click",
            ),
        ],
        variables=["email", "password"],
    )


@pytest.fixture
def dashboard_playbook(test_server) -> Playbook:
    """A playbook for dashboard interactions."""
    return Playbook(
        name="test-dashboard",
        url=f"{test_server.url}/dashboard",
        steps=[
            PlaybookStep(
                intent="click create new button",
                selector="#btn-new",
                selector_alternatives=["button:has-text('Create New')"],
                action="click",
            ),
            PlaybookStep(
                intent="select active filter",
                selector="#filter-dropdown",
                selector_alternatives=["select[aria-label='Filter']"],
                action="select",
                value="active",
            ),
            PlaybookStep(
                intent="navigate to settings",
                selector="#nav-settings",
                selector_alternatives=["a:has-text('Settings')"],
                action="click",
            ),
        ],
    )
