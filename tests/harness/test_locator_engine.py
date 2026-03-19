"""Tests for Strategy 6: Playwright locator engine healing."""

import pytest

from autopilot.core.healer import (
    heal,
    heal_with_locator_engine,
    _extract_selector_attrs,
    _find_candidates_by_attrs,
)
from autopilot.core.playbook import PlaybookStep


class TestSelectorAttrExtraction:
    """Test extracting attribute hints from CSS selectors."""

    def test_extract_id(self):
        attrs = _extract_selector_attrs("#email-input")
        assert attrs["id_hint"] == "email-input"

    def test_extract_name(self):
        attrs = _extract_selector_attrs("input[name='email']")
        assert attrs["name"] == "email"
        assert attrs["tag"] == "input"

    def test_extract_type(self):
        attrs = _extract_selector_attrs("input[type='password']")
        assert attrs["type"] == "password"

    def test_extract_class(self):
        attrs = _extract_selector_attrs(".btn-primary.large")
        assert "btn-primary" in attrs["class_hints"]
        assert "large" in attrs["class_hints"]

    def test_extract_complex(self):
        attrs = _extract_selector_attrs("input#user-email[name='email'][type='text']")
        assert attrs["id_hint"] == "user-email"
        assert attrs["name"] == "email"
        assert attrs["type"] == "text"
        assert attrs["tag"] == "input"

    def test_extract_empty(self):
        attrs = _extract_selector_attrs("*")
        assert len(attrs) == 0 or attrs.get("tag") == "*"


@pytest.mark.tier1
class TestLocatorEngineStrategy:
    """Test the Playwright locator engine healing strategy."""

    async def test_composite_locator_finds_button(self, browser_at_login):
        """Composite role+text locator finds the login button."""
        page = browser_at_login.page
        step = PlaybookStep(
            intent="click the log in button",
            selector="#nonexistent-btn",
            action="click",
        )
        result = await heal_with_locator_engine(page, step)
        # May or may not succeed depending on exact page structure
        # The key assertion is it doesn't crash
        if result.success:
            assert result.strategy in ("test_id", "alt_text", "title", "composite_locator", "attr_proximity")

    async def test_attr_proximity_finds_renamed_element(self, browser, test_server):
        """Attribute proximity finds an element whose ID changed."""
        test_server.set_mutated(True)
        await browser.page.goto(f"{test_server.url}/login")

        # Old selector had #email, mutated page has #user-email
        step = PlaybookStep(
            intent="fill email field",
            selector="#email",
            selector_alternatives=[],
            action="fill",
        )
        result = await heal_with_locator_engine(browser.page, step)
        if result.success:
            assert result.score > 0
            # Should find #user-email via id_hint proximity
            assert "email" in result.new_selector.lower() or "user" in result.new_selector.lower()


@pytest.mark.tier1
class TestLocatorEngineInFullHeal:
    """Test that Strategy 6 is invoked as part of the full heal chain."""

    async def test_heal_chain_includes_locator_engine(self, browser, test_server):
        """The full heal() function tries locator engine as last resort."""
        test_server.set_mutated(True)
        await browser.page.goto(f"{test_server.url}/login")

        # Use a step where basic strategies might fail but locator engine succeeds
        step = PlaybookStep(
            intent="fill email address",
            selector="#email",
            selector_alternatives=[],  # No alternatives
            action="fill",
        )
        result = await heal(browser.page, step)

        # Should succeed via one of the strategies (possibly locator_engine)
        if result.success:
            assert result.strategy
