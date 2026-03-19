"""Auto-generated Playwright test from autopilot playbook: log-in-with-standard-user-secret-sauce-a

Generated from: https://www.saucedemo.com
Steps: 4
Run with: pytest log_in_with_standard_user_secret_sauce_a.py -v

NO DEPENDENCY ON AUTOPILOT — this is standard Playwright.
"""

import re
import pytest
from playwright.sync_api import Page, expect

# Variables — change these or parametrize with pytest
VARIABLES = {
    "username": "standard_user",
    "password": "secret_sauce",
    "firstName": "John",
    "lastName": "Doe",
    "zip": "10001"
}


@pytest.fixture(scope="module")
def browser_context(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    yield context
    context.close()
    browser.close()


@pytest.fixture
def page(browser_context):
    page = browser_context.new_page()
    page.set_default_timeout(15000)
    yield page
    page.close()


def test_log_in_with_standard_user_secret_sauce_a(page: Page):
    """Test: log-in-with-standard-user-secret-sauce-a"""
    # Navigate to starting URL
    page.goto("https://www.saucedemo.com")
    page.wait_for_load_state("domcontentloaded")

    # Step 1: fill username field
    page.locator("[placeholder=\"Username\"]").or_(page.locator("input[type=\'text\']")).fill(VARIABLES["username"])
    page.wait_for_timeout(500)

    # Step 2: fill password field
    page.locator("[placeholder=\"Password\"]").or_(page.locator("input[type=\'password\']")).fill(VARIABLES["password"])
    page.wait_for_timeout(500)

    # Step 3: click login button
    page.locator("input[type=\"submit\"]").or_(page.locator("input[type=\'submit\']")).click()
    page.wait_for_timeout(500)

    # Step 4: add first product to cart
    page.get_by_role("button", name="Add to cart").click()
    page.wait_for_timeout(500)
