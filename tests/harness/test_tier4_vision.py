"""Tier 4 tests: Vision fallback — screenshot + accessibility."""

import pytest

from autopilot.core.vision import capture_for_vision, format_vision_prompt


@pytest.mark.tier4
class TestVisionCapture:
    """Test the vision capture system."""

    async def test_capture_produces_screenshot(self, browser_at_login):
        """Capture returns a base64-encoded screenshot."""
        page = browser_at_login.page
        capture = await capture_for_vision(page)

        assert capture.screenshot_b64
        assert len(capture.screenshot_b64) > 100  # Non-trivial image
        assert capture.viewport_width > 0
        assert capture.viewport_height > 0
        assert capture.url.endswith("/login")

    async def test_capture_includes_accessibility_summary(self, browser_at_login):
        """Capture includes compact accessibility summary."""
        page = browser_at_login.page
        capture = await capture_for_vision(page, include_accessibility=True)

        assert capture.accessibility_summary
        # Should list some interactive elements
        assert "e0:" in capture.accessibility_summary

    async def test_capture_without_accessibility(self, browser_at_login):
        """Capture works without accessibility summary."""
        page = browser_at_login.page
        capture = await capture_for_vision(page, include_accessibility=False)

        assert capture.screenshot_b64
        assert capture.accessibility_summary == ""

    async def test_format_vision_prompt(self, browser_at_login):
        """format_vision_prompt produces a structured prompt."""
        page = browser_at_login.page
        capture = await capture_for_vision(page)
        prompt_data = format_vision_prompt(capture, intent="click login button")

        assert "click login button" in prompt_data["text"]
        assert prompt_data["image_b64"]
        assert prompt_data["media_type"] == "image/png"
        assert "Page:" in prompt_data["text"]
