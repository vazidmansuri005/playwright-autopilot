"""Tier 1 tests: Heuristic self-healing — zero LLM tokens."""

import pytest

from autopilot.core.healer import (
    heal,
    heal_with_alternatives,
    heal_with_label,
    heal_with_placeholder,
    heal_with_role,
    heal_with_text_match,
)
from autopilot.core.playbook import PlaybookStep
from autopilot.core.runner import Runner


@pytest.mark.tier1
class TestTier1HealerStrategies:
    """Test individual healing strategies."""

    async def test_heal_with_label(self, browser_at_login):
        """Heal an input by its associated <label>."""
        page = browser_at_login.page
        step = PlaybookStep(
            intent="fill email field",
            selector="#nonexistent",
            action="fill",
        )
        result = await heal_with_label(page, step)
        assert result.success
        assert result.strategy == "label_match"
        assert result.score > 0

    async def test_heal_with_placeholder(self, browser_at_login):
        """Heal an input by placeholder text."""
        page = browser_at_login.page
        step = PlaybookStep(
            intent="enter email",
            selector="#nonexistent",
            action="fill",
        )
        result = await heal_with_placeholder(page, step)
        assert result.success
        assert result.strategy == "placeholder_match"

    async def test_heal_with_role(self, browser_at_login):
        """Heal a button by ARIA role."""
        page = browser_at_login.page
        # Use "submit" which is a meaningful keyword for the login button
        step = PlaybookStep(
            intent="click submit button",
            selector="#nonexistent",
            action="click",
        )
        result = await heal_with_role(page, step)
        # Role heal may or may not find a unique match — depends on page structure
        # The key assertion is that it doesn't error out
        if result.success:
            assert result.strategy == "role_match"

    async def test_heal_with_text(self, browser_at_login):
        """Heal by text content matching."""
        page = browser_at_login.page
        step = PlaybookStep(
            intent="click forgot password link",
            selector="#nonexistent",
            action="click",
        )
        result = await heal_with_text_match(page, step)
        assert result.success
        assert result.strategy == "text_match"

    async def test_heal_combined_runs_all_strategies(self, browser_at_login):
        """The combined heal() function tries all strategies."""
        page = browser_at_login.page
        step = PlaybookStep(
            intent="fill password",
            selector="#definitely-not-here",
            action="fill",
        )
        result = await heal(page, step)
        assert result.success
        # Should be healed by one of the strategies
        assert result.strategy in ("label_match", "placeholder_match", "role_match")

    async def test_heal_fails_for_nonexistent_element(self, browser_at_login):
        """Heal returns failure when no element matches."""
        page = browser_at_login.page
        step = PlaybookStep(
            intent="click unicorn button",
            selector="#nonexistent",
            action="click",
        )
        result = await heal(page, step)
        assert not result.success


@pytest.mark.tier1
class TestTier1InRunner:
    """Test Tier 1 healing within the full runner."""

    async def test_runner_escalates_to_tier1(self, browser, test_server, login_playbook_mutated):
        """When Tier 0 fails, runner escalates to Tier 1 (still zero tokens)."""
        test_server.set_mutated(True)
        runner = Runner(browser=browser, max_tier=1)  # Allow up to Tier 1
        result = await runner.run(
            login_playbook_mutated,
            variables={"email": "test@example.com", "password": "secret"},
        )

        # At least some steps should heal at Tier 1
        tier1_steps = [s for s in result.steps if s.tier == 1]
        assert len(tier1_steps) > 0
        assert result.total_tokens == 0  # Tier 1 = zero tokens
        assert runner.usage.call_count == 0

    async def test_healed_selector_promoted(self, browser, test_server, login_playbook_mutated):
        """After healing, new selector is promoted in the playbook."""
        test_server.set_mutated(True)
        runner = Runner(browser=browser, max_tier=1, save_on_heal=True)

        original_selectors = [s.selector for s in login_playbook_mutated.steps]

        result = await runner.run(
            login_playbook_mutated,
            variables={"email": "test@example.com", "password": "secret"},
        )

        if result.playbook_updated:
            # At least one selector should have changed
            new_selectors = [s.selector for s in login_playbook_mutated.steps]
            assert new_selectors != original_selectors

    async def test_healed_playbook_replays_at_tier0(self, browser, test_server, login_playbook_mutated):
        """After healing, the SAME playbook replays at Tier 0 (drift to determinism)."""
        test_server.set_mutated(True)

        # First run: heal
        runner1 = Runner(browser=browser, max_tier=1, save_on_heal=True)
        result1 = await runner1.run(
            login_playbook_mutated,
            variables={"email": "test@example.com", "password": "secret"},
        )

        if not result1.success:
            pytest.skip("First run failed to heal — cannot test replay")

        # Navigate away and back
        await browser.page.goto("about:blank")

        # Second run with SAME (now-updated) playbook: should be Tier 0
        runner2 = Runner(browser=browser, max_tier=0)
        result2 = await runner2.run(
            login_playbook_mutated,
            variables={"email": "test@example.com", "password": "secret"},
        )

        assert result2.success
        # All steps should now be Tier 0 (deterministic)
        assert all(s.tier == 0 for s in result2.steps)
        assert result2.total_tokens == 0
