"""Tier 0 tests: Deterministic replay — zero tokens."""

import pytest

from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner


@pytest.mark.tier0
class TestTier0Replay:
    """Test that cached selectors replay without any LLM calls."""

    async def test_login_replay_happy_path(self, browser, test_server, login_playbook):
        """All selectors match → all steps resolve at Tier 0."""
        runner = Runner(browser=browser, max_tier=0)  # Force Tier 0 only
        result = await runner.run(
            login_playbook,
            variables={"email": "test@example.com", "password": "secret123"},
        )

        assert result.success
        assert len(result.steps) == 3
        assert all(s.tier == 0 for s in result.steps)
        assert result.total_tokens == 0
        assert runner.usage.call_count == 0

    async def test_dashboard_replay(self, browser, test_server, dashboard_playbook):
        """Dashboard playbook replays at Tier 0."""
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(dashboard_playbook)

        assert result.success
        assert all(s.tier == 0 for s in result.steps)
        assert result.total_tokens == 0

    async def test_tier0_fails_on_missing_selector(self, browser, test_server):
        """When primary + alternatives don't match, Tier 0 fails."""
        test_server.set_mutated(True)

        # Playbook with selectors that DEFINITELY don't match anything
        playbook = Playbook(
            name="impossible",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="click nonexistent button",
                    selector="#absolutely-not-here",
                    selector_alternatives=["#also-missing", "#nope"],
                    action="click",
                ),
            ],
        )

        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)

        assert not result.success
        assert result.steps[0].tier == -1  # All tiers exhausted
        assert result.total_tokens == 0

    async def test_alternative_selector_used(self, browser, test_server):
        """When primary fails, alternatives at Tier 0 succeed."""
        from autopilot.core.playbook import Playbook, PlaybookStep

        playbook = Playbook(
            name="test-alt",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email",
                    selector="#nonexistent",  # Primary won't match
                    selector_alternatives=[
                        "#also-nonexistent",
                        "#email",  # This one will match
                    ],
                    action="fill",
                    value="test@test.com",
                ),
            ],
        )

        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)

        assert result.success
        assert result.steps[0].tier == 0
        assert result.steps[0].selector_used == "#email"

    async def test_variable_substitution(self, browser, test_server, login_playbook):
        """Variables are substituted into templated values."""
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(
            login_playbook,
            variables={"email": "custom@test.com", "password": "custom_pass"},
        )

        assert result.success
        # Verify the value was actually filled
        email_val = await browser.page.input_value("#email")
        assert email_val == "custom@test.com"

    async def test_run_count_tracked(self, browser, test_server, login_playbook):
        """Playbook tracks run count and success rate."""
        runner = Runner(browser=browser, max_tier=0)

        assert login_playbook.run_count == 0

        await runner.run(login_playbook, variables={"email": "a@b.com", "password": "p"})
        assert login_playbook.run_count == 1
        assert login_playbook.success_count == 1
        assert login_playbook.success_rate == 1.0
