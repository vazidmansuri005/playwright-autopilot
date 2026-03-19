"""Integration tests: Full pipeline through multiple tiers."""

import pytest

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner
from autopilot.llm.base import MockLLM


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end tests for the tiered escalation pipeline."""

    async def test_full_login_flow_tier0(self, browser, test_server, login_playbook):
        """Complete login flow at Tier 0 with token tracking."""
        runner = Runner(browser=browser, max_tier=4)
        result = await runner.run(
            login_playbook,
            variables={"email": "user@example.com", "password": "pass123"},
        )

        assert result.success
        assert result.total_tokens == 0  # All Tier 0
        assert result.tier_counts.get(0, 0) == 3

        summary = result.summary
        assert summary["steps_total"] == 3
        assert summary["steps_passed"] == 3

    async def test_drift_to_determinism(self, browser, test_server):
        """Demonstrates the core value prop: first run heals, second run replays."""
        test_server.set_mutated(True)

        # Playbook with wrong selectors
        playbook = Playbook(
            name="drift-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email field",
                    selector="#email",  # Wrong on mutated page
                    selector_alternatives=["input[type='email']"],  # This works!
                    action="fill",
                    value="test@test.com",
                ),
            ],
        )

        # Run 1: May need Tier 0 alternatives or Tier 1 healing
        runner1 = Runner(browser=browser, max_tier=1, save_on_heal=True)
        result1 = await runner1.run(playbook)
        assert result1.success

        # Navigate away
        await browser.page.goto("about:blank")

        # Run 2: Should be pure Tier 0 (playbook was updated)
        runner2 = Runner(browser=browser, max_tier=0)
        result2 = await runner2.run(playbook)
        assert result2.success
        assert result2.total_tokens == 0

    async def test_multi_page_flow(self, browser, test_server):
        """Multi-page flow: login → dashboard → settings."""
        playbook = Playbook(
            name="multi-page",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email",
                    selector="#email",
                    action="fill",
                    value="admin@test.com",
                ),
                PlaybookStep(
                    intent="fill password",
                    selector="#password",
                    action="fill",
                    value="admin",
                ),
                PlaybookStep(
                    intent="submit login",
                    selector="#login-btn",
                    action="click",
                    wait_after_ms=500,
                ),
            ],
        )

        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)

        assert result.success
        assert len(result.steps) == 3

    async def test_playbook_persistence_after_heal(self, browser, test_server, tmp_path):
        """Healed playbook saves to disk and loads correctly."""
        test_server.set_mutated(True)

        playbook = Playbook(
            name="persist-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email field",
                    selector="#email",
                    selector_alternatives=["input[type='email']"],
                    action="fill",
                    value="test@test.com",
                ),
            ],
        )

        runner = Runner(browser=browser, max_tier=1, save_on_heal=True)
        result = await runner.run(playbook)

        # Save
        save_path = tmp_path / "healed_playbook.json"
        playbook.save(save_path)

        # Load and verify
        loaded = Playbook.load(save_path)
        assert loaded.name == "persist-test"
        assert len(loaded.steps) == 1
        assert loaded.run_count == 1

    async def test_step_timing(self, browser, test_server, login_playbook):
        """Each step has duration tracking."""
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(
            login_playbook,
            variables={"email": "a@b.com", "password": "p"},
        )

        for step_result in result.steps:
            assert step_result.duration_ms > 0

        assert result.total_duration_ms > 0

    async def test_dynamic_page_wait(self, browser, test_server):
        """Handle pages with dynamically loaded content."""
        playbook = Playbook(
            name="dynamic-test",
            url=f"{test_server.url}/dynamic",
            steps=[
                PlaybookStep(
                    intent="click submit button",
                    selector="#delayed-btn",
                    action="click",
                    wait_after_ms=0,
                ),
            ],
        )

        # Need longer timeout since elements appear after 1s delay
        runner = Runner(browser=browser, max_tier=0, tier0_timeout=5000)
        result = await runner.run(playbook)

        assert result.success

    async def test_settings_form_flow(self, browser, test_server):
        """Complex form interaction: text + select + checkbox."""
        playbook = Playbook(
            name="settings-flow",
            url=f"{test_server.url}/settings",
            steps=[
                PlaybookStep(
                    intent="clear and fill display name",
                    selector="#display-name",
                    action="fill",
                    value="New Name",
                ),
                PlaybookStep(
                    intent="select pacific timezone",
                    selector="#timezone",
                    action="select",
                    value="PST",
                ),
                PlaybookStep(
                    intent="enable dark mode",
                    selector="#dark-mode",
                    action="check",
                ),
                PlaybookStep(
                    intent="save changes",
                    selector="#save-btn",
                    action="click",
                ),
            ],
        )

        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)

        assert result.success
        assert len(result.steps) == 4

        # Verify values were set
        name_val = await browser.page.input_value("#display-name")
        assert name_val == "New Name"

        tz_val = await browser.page.input_value("#timezone")
        assert tz_val == "PST"

        dark_checked = await browser.page.is_checked("#dark-mode")
        assert dark_checked

    async def test_empty_playbook(self, browser, test_server):
        """Empty playbook succeeds immediately."""
        playbook = Playbook(name="empty", url=f"{test_server.url}/login", steps=[])
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)

        assert result.success
        assert len(result.steps) == 0
        assert result.total_tokens == 0


@pytest.mark.integration
class TestTokenBudget:
    """Verify token economics match our promises."""

    async def test_cached_run_zero_tokens(self, browser, test_server, login_playbook):
        """A fully cached run costs exactly zero tokens."""
        runner = Runner(browser=browser, max_tier=4)
        result = await runner.run(
            login_playbook,
            variables={"email": "a@b.com", "password": "p"},
        )

        assert result.success
        assert result.total_tokens == 0
        assert runner.usage.total_tokens == 0
        assert runner.usage.call_count == 0

    async def test_tier1_zero_tokens(self, browser, test_server, login_playbook_mutated):
        """Tier 1 healing costs zero tokens."""
        test_server.set_mutated(True)
        runner = Runner(browser=browser, max_tier=1)
        result = await runner.run(
            login_playbook_mutated,
            variables={"email": "a@b.com", "password": "p"},
        )

        assert result.total_tokens == 0
        assert runner.usage.total_tokens == 0
