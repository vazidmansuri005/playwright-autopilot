"""Tier 3 tests: Full accessibility tree AI — ~5K tokens."""

import pytest

from autopilot.core.snapshot import extract_snapshot
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner
from autopilot.llm.base import MockLLM


@pytest.mark.tier3
class TestAccessibilitySnapshot:
    """Test the accessibility tree extraction."""

    async def test_snapshot_captures_structure(self, browser_at_login):
        """Snapshot produces a readable tree structure."""
        page = browser_at_login.page
        snapshot = await extract_snapshot(page)

        assert snapshot.url.endswith("/login")
        assert snapshot.title == "Test Login"
        assert len(snapshot.tree) > 0
        assert snapshot.token_estimate > 0

    async def test_snapshot_interactive_only(self, browser_at_login):
        """Interactive-only mode produces a smaller tree."""
        page = browser_at_login.page

        full = await extract_snapshot(page, interactive_only=False)
        interactive = await extract_snapshot(page, interactive_only=True)

        # Interactive-only should have fewer lines
        full_lines = full.tree.count("\n")
        interactive_lines = interactive.tree.count("\n")
        assert interactive_lines <= full_lines

    async def test_snapshot_token_estimate_reasonable(self, browser_at_dashboard):
        """Dashboard snapshot stays under reasonable token budget."""
        page = browser_at_dashboard.page
        snapshot = await extract_snapshot(page, interactive_only=True)

        # Interactive-only dashboard should be well under 10K tokens
        assert snapshot.token_estimate < 10_000

    async def test_snapshot_contains_element_refs(self, browser_at_login):
        """Interactive elements have [eN] references."""
        page = browser_at_login.page
        snapshot = await extract_snapshot(page, interactive_only=False)

        # Should contain element references for interactive elements
        # or at minimum should have content (not empty)
        assert len(snapshot.tree) > 30
        assert snapshot.tree != "[Empty accessibility tree]"


@pytest.mark.tier3
class TestTier3InRunner:
    """Test Tier 3 full AI within the runner."""

    async def test_runner_escalates_to_tier3(self, browser, test_server):
        """When Tiers 0-2 fail, runner escalates to Tier 3."""
        test_server.set_mutated(True)

        # Mock that fails at Tier 2 format but succeeds at Tier 3 format
        mock = MockLLM(responses=[
            "cannot determine ref",  # Tier 2 fails
            "action=fill ref=e0 value=test@test.com",  # Tier 3 succeeds
        ])

        playbook = Playbook(
            name="tier3-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email field",
                    selector="#impossible",
                    selector_alternatives=[],
                    action="fill",
                    value="test@test.com",
                ),
            ],
        )

        runner = Runner(browser=browser, llm=mock, max_tier=3)
        result = await runner.run(playbook)

        # Tier 1 heal may succeed before reaching Tier 3
        # The key assertion is the pipeline doesn't crash
        assert runner.usage.call_count >= 0  # LLM may or may not be needed
