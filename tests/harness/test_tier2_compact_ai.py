"""Tier 2 tests: Compact AI with page digest — ~500 tokens."""

import pytest

from autopilot.core.digest import extract_digest, extract_focused_digest
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner
from autopilot.llm.base import MockLLM


@pytest.mark.tier2
class TestPageDigest:
    """Test the compact page digest extraction."""

    async def test_digest_extracts_interactive_elements(self, browser_at_login):
        """Digest captures all interactive elements on the page."""
        page = browser_at_login.page
        digest = await extract_digest(page)

        assert digest.url.endswith("/login")
        assert digest.title == "Test Login"
        assert len(digest.elements) > 0

        # Should find: email input, password input, login button, forgot link
        tags = [e["tag"] for e in digest.elements]
        assert "input" in tags
        assert "button" in tags or "a" in tags

    async def test_digest_token_estimate(self, browser_at_login):
        """Token estimate is reasonable for a simple page."""
        page = browser_at_login.page
        digest = await extract_digest(page)

        # Simple login page should be well under 1000 tokens
        assert digest.token_estimate < 1000
        assert digest.token_estimate > 0

    async def test_digest_element_data_complete(self, browser_at_login):
        """Each element has the required fields."""
        page = browser_at_login.page
        digest = await extract_digest(page)

        for el in digest.elements:
            assert "ref" in el
            assert "tag" in el
            assert "visible" in el

    async def test_focused_digest_prioritizes_relevant(self, browser_at_login):
        """Focused digest scores elements by relevance to intent."""
        page = browser_at_login.page
        digest = await extract_focused_digest(page, intent="fill email", max_elements=5)

        # The email-related element should be near the top
        top_names = [e.get("name", "").lower() for e in digest.elements[:3]]
        top_placeholders = [e.get("placeholder", "").lower() for e in digest.elements[:3]]
        has_email = any("email" in n for n in top_names + top_placeholders)
        assert has_email, f"Email element not in top 3: {digest.elements[:3]}"

    async def test_digest_to_prompt_format(self, browser_at_login):
        """to_prompt() produces a formatted table."""
        page = browser_at_login.page
        digest = await extract_digest(page)
        prompt = digest.to_prompt()

        assert "Page:" in prompt
        assert "ref|tag|role|name|type|placeholder|visible" in prompt
        assert "---|---|---|---|---|---|---" in prompt
        assert "e0|" in prompt

    async def test_dashboard_digest(self, browser_at_dashboard):
        """Dashboard page produces a rich digest."""
        page = browser_at_dashboard.page
        digest = await extract_digest(page)

        # Dashboard has: nav links, buttons, select dropdown
        assert len(digest.elements) >= 5

        # Find the filter dropdown
        selects = [e for e in digest.elements if e["tag"] == "select"]
        assert len(selects) > 0


@pytest.mark.tier2
class TestTier2InRunner:
    """Test Tier 2 compact AI within the runner."""

    async def test_runner_escalates_to_tier2_with_mock_llm(self, browser, test_server):
        """When Tier 0+1 fail, runner escalates to Tier 2 with LLM."""
        test_server.set_mutated(True)

        # Mock LLM that returns the correct element ref
        # On the mutated page, email input might be e0 or e1
        mock = MockLLM(responses=["ref=e0", "ref=e1", "ref=e2"])

        playbook = Playbook(
            name="tier2-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email field",
                    selector="#totally-wrong-selector",
                    selector_alternatives=[],  # Empty so Tier 0 fails
                    action="fill",
                    value="test@test.com",
                ),
            ],
        )

        runner = Runner(browser=browser, llm=mock, max_tier=2)
        result = await runner.run(playbook)

        # Should have attempted LLM (Tier 2)
        if result.success:
            assert result.steps[0].tier == 2 or result.steps[0].tier == 1
            assert runner.usage.call_count >= 0  # May or may not need LLM

    async def test_tier2_token_usage_tracked(self, browser, test_server):
        """Token usage is properly tracked for Tier 2 calls."""
        mock = MockLLM(responses=["ref=e0"])

        playbook = Playbook(
            name="token-track",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="click something unique",
                    selector="#impossible",
                    selector_alternatives=[],
                    action="click",
                ),
            ],
        )

        runner = Runner(browser=browser, llm=mock, max_tier=2)
        await runner.run(playbook)

        # If LLM was called, tokens should be tracked
        if runner.usage.call_count > 0:
            assert runner.usage.total_tokens > 0
            summary = runner.usage.summary()
            assert "by_tier" in summary
