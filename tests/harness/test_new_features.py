"""Tests for all new features: conditionals, assertions, import, chaining, etc."""

import pytest
from pathlib import Path

from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner
from autopilot.core.assertions import evaluate_assertion, _heuristic_assert
from autopilot.core.audit import AuditTrail
from autopilot.importers import import_playwright, _parse_playwright_ts, _parse_playwright_py


# === Conditional Steps ===

@pytest.mark.integration
class TestConditionalSteps:

    async def test_if_visible_runs_when_present(self, browser, test_server):
        """Step with if_visible runs when element exists."""
        playbook = Playbook(
            name="cond-visible",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email if visible",
                    selector="#email", action="fill", value="test@test.com",
                    condition="if_visible",
                ),
            ],
        )
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)
        assert result.success
        assert result.steps[0].strategy != "skipped"

    async def test_if_visible_skips_when_absent(self, browser, test_server):
        """Step with if_visible is skipped when element doesn't exist."""
        playbook = Playbook(
            name="cond-skip",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="close cookie banner if visible",
                    selector="#cookie-accept-nonexistent",
                    action="click",
                    condition="if_visible",
                ),
                PlaybookStep(
                    intent="fill email",
                    selector="#email", action="fill", value="test@test.com",
                ),
            ],
        )
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)
        assert result.success
        assert result.steps[0].strategy == "skipped"  # First step skipped
        assert result.steps[1].success  # Second step ran

    async def test_skip_on_fail(self, browser, test_server):
        """skip_on_fail=True allows flow to continue after failure."""
        playbook = Playbook(
            name="skip-fail",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="click nonexistent thing",
                    selector="#absolutely-missing",
                    action="click",
                    skip_on_fail=True,
                ),
                PlaybookStep(
                    intent="fill email",
                    selector="#email", action="fill", value="a@b.com",
                ),
            ],
        )
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)
        assert result.success  # Flow continues despite first step failing
        assert result.steps[0].strategy == "skipped_on_fail"
        assert result.steps[1].success

    async def test_if_url_contains(self, browser, test_server):
        """Step with if_url_contains runs when URL matches."""
        playbook = Playbook(
            name="cond-url",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email on login page",
                    selector="#email", action="fill", value="a@b.com",
                    condition="if_url_contains", condition_value="login",
                ),
            ],
        )
        runner = Runner(browser=browser, max_tier=0)
        result = await runner.run(playbook)
        assert result.success


# === Heuristic Assertions ===

class TestHeuristicAssertions:

    async def test_title_assertion(self, browser_at_login):
        """Heuristic assertion for page title."""
        passed, msg = await _heuristic_assert(browser_at_login.page, "page title contains 'Test Login'")
        assert passed

    async def test_title_assertion_fails(self, browser_at_login):
        """Heuristic assertion fails when title doesn't match."""
        passed, msg = await _heuristic_assert(browser_at_login.page, "page title contains 'Dashboard'")
        assert not passed

    async def test_url_assertion(self, browser_at_login):
        """Heuristic assertion for URL."""
        passed, msg = await _heuristic_assert(browser_at_login.page, "url contains login")
        assert passed

    async def test_text_visible_assertion(self, browser_at_login):
        """Heuristic assertion for visible text."""
        passed, msg = await _heuristic_assert(browser_at_login.page, "'Login' is visible")
        assert passed

    async def test_text_not_visible(self, browser_at_login):
        """Heuristic assertion for hidden text."""
        passed, msg = await _heuristic_assert(browser_at_login.page, "'Nonexistent text xyz' is not visible")
        assert passed

    async def test_unknown_assertion_returns_none(self, browser_at_login):
        """Unknown assertion patterns fall back to LLM (return None)."""
        result = await _heuristic_assert(browser_at_login.page, "the vibe is good")
        assert result is None  # Cannot evaluate heuristically


# === Playwright Import ===

class TestPlaywrightImport:

    def test_parse_ts_click(self):
        code = "await page.click('#submit-btn');"
        steps = _parse_playwright_ts(code)
        assert len(steps) == 1
        assert steps[0].action == "click"
        assert steps[0].selector == "#submit-btn"

    def test_parse_ts_fill(self):
        code = "await page.fill('#email', 'user@test.com');"
        steps = _parse_playwright_ts(code)
        assert len(steps) == 1
        assert steps[0].action == "fill"
        assert steps[0].value == "user@test.com"

    def test_parse_ts_goto(self):
        code = "await page.goto('https://example.com');"
        steps = _parse_playwright_ts(code)
        assert len(steps) == 1
        assert steps[0].action == "navigate"
        assert steps[0].value == "https://example.com"

    def test_parse_ts_select(self):
        code = "await page.selectOption('#timezone', 'PST');"
        steps = _parse_playwright_ts(code)
        assert len(steps) == 1
        assert steps[0].action == "select"
        assert steps[0].value == "PST"

    def test_parse_ts_get_by_role(self):
        code = "await page.getByRole('button', { name: 'Submit' }).click();"
        steps = _parse_playwright_ts(code)
        assert any(s.action == "click" and "Submit" in s.selector for s in steps)

    def test_parse_py_fill(self):
        code = 'page.fill("#email", "user@test.com")'
        steps = _parse_playwright_py(code)
        assert len(steps) == 1
        assert steps[0].value == "user@test.com"

    def test_parse_py_click(self):
        code = 'page.click("#login-btn")'
        steps = _parse_playwright_py(code)
        assert len(steps) == 1
        assert steps[0].action == "click"

    def test_full_import(self, tmp_path):
        script = tmp_path / "recording.ts"
        script.write_text("""
            await page.goto('https://example.com/login');
            await page.fill('#email', 'user@test.com');
            await page.fill('#password', 'secret');
            await page.click('#login-btn');
        """)
        playbook = import_playwright(script, name="test-import")
        assert playbook.name == "test-import"
        assert playbook.url == "https://example.com/login"
        assert len(playbook.steps) == 3  # goto becomes URL, 3 action steps


# === Audit Trail ===

class TestAuditTrail:

    def test_record_and_save(self, tmp_path):
        audit = AuditTrail(output_dir=tmp_path)
        audit.record(
            step_index=0, intent="fill email", action="fill",
            selector="#email", tier=0, strategy="replay",
            success=True, duration_ms=50.0, tokens_used=0,
            value="user@secret.com",
        )
        path = audit.save("test-playbook")
        assert path.exists()

        import json
        data = json.loads(path.read_text())
        masked = data["entries"][0]["value_masked"]
        assert masked.startswith("us") and masked.endswith("om") and "*" in masked
        assert data["summary"]["total_actions"] == 1

    def test_value_masking(self):
        audit = AuditTrail(mask_values=True)
        assert audit._mask("password123") == "pa*******23"
        assert audit._mask("ab") == "***"
        assert audit._mask("") == ""

    def test_to_text(self, tmp_path):
        audit = AuditTrail(output_dir=tmp_path)
        audit.record(
            step_index=0, intent="click login", action="click",
            selector="#btn", tier=0, strategy="replay",
            success=True, duration_ms=100.0, tokens_used=0,
        )
        text = audit.to_text()
        assert "Step 0" in text
        assert "click" in text


# === Playbook Chaining ===

@pytest.mark.integration
class TestPlaybookChaining:

    async def test_chain_two_playbooks(self, browser, test_server, tmp_path):
        from autopilot import Agent

        pb1 = Playbook(
            name="chain-step1",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(intent="fill email", selector="#email", action="fill", value="a@b.com"),
            ],
        )
        pb2 = Playbook(
            name="chain-step2",
            url=f"{test_server.url}/login",  # Same page, continues where pb1 left off
            steps=[
                PlaybookStep(intent="fill password", selector="#password", action="fill", value="pass"),
            ],
        )
        pb1.save(tmp_path / "chain-step1.json")
        pb2.save(tmp_path / "chain-step2.json")

        async with Agent(llm=None, headless=True, playbook_dir=tmp_path) as agent:
            results = await agent.run_chain(["chain-step1", "chain-step2"])
            assert len(results) == 2
            assert all(r.success for r in results)


# === Runner with Audit and Profile ===

@pytest.mark.integration
class TestRunnerFeatures:

    async def test_runner_with_audit(self, browser, test_server, login_playbook):
        runner = Runner(browser=browser, max_tier=0, audit=True)
        result = await runner.run(
            login_playbook,
            variables={"email": "a@b.com", "password": "p"},
        )
        assert result.success
        assert result.audit_path
        assert Path(result.audit_path).exists()

    async def test_runner_with_profiling(self, browser, test_server, login_playbook):
        runner = Runner(browser=browser, max_tier=0, profile=True)
        result = await runner.run(
            login_playbook,
            variables={"email": "a@b.com", "password": "p"},
        )
        assert result.success
        assert result.performance
        assert "avg_step_ms" in result.performance

    async def test_step_run_count_tracking(self, browser, test_server, login_playbook):
        runner = Runner(browser=browser, max_tier=0)
        await runner.run(login_playbook, variables={"email": "a@b.com", "password": "p"})

        for step in login_playbook.steps:
            assert step.run_count >= 1
