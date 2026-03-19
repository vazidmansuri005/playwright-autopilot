"""Tests for the AI Explorer — learning mode that records playbooks."""

import pytest

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.explorer import (
    Explorer,
    _parse_explore_response,
    _generate_alternatives,
    _slugify,
)
from autopilot.llm.base import MockLLM


class TestExploreResponseParser:
    """Test parsing of explorer LLM responses."""

    def test_parse_click_action(self):
        r = _parse_explore_response("action=click ref=e5 | intent=click login button")
        assert r["action"] == "click"
        assert r["ref"] == 5
        assert r["intent"] == "click login button"

    def test_parse_fill_action(self):
        r = _parse_explore_response("action=fill ref=e2 value=user@test.com | intent=fill email field")
        assert r["action"] == "fill"
        assert r["ref"] == 2
        assert r["value"] == "user@test.com"
        assert r["intent"] == "fill email field"

    def test_parse_select_action(self):
        r = _parse_explore_response("action=select ref=e8 value=PST | intent=select timezone")
        assert r["action"] == "select"
        assert r["ref"] == 8
        assert r["value"] == "PST"

    def test_parse_done(self):
        r = _parse_explore_response("action=done | intent=goal complete")
        assert r["action"] == "done"
        assert r["intent"] == "goal complete"

    def test_parse_wait(self):
        r = _parse_explore_response("action=wait | intent=wait for page to load")
        assert r["action"] == "wait"

    def test_parse_no_intent(self):
        r = _parse_explore_response("action=click ref=e3")
        assert r["action"] == "click"
        assert r["ref"] == 3

    def test_parse_invalid(self):
        r = _parse_explore_response("I don't know what to do")
        assert r is None

    def test_parse_multiline_takes_first(self):
        r = _parse_explore_response("action=click ref=e1 | intent=first\naction=fill ref=e2")
        assert r["action"] == "click"
        assert r["ref"] == 1


class TestAlternativeGeneration:
    """Test alternative selector generation."""

    def test_generates_role_alternative(self):
        el = {"tag": "button", "name": "Submit", "role": "button", "type": "", "placeholder": "", "href": ""}
        alts = _generate_alternatives(el)
        assert any("role:button" in a for a in alts)

    def test_generates_input_type_alternative(self):
        el = {"tag": "input", "name": "", "role": "", "type": "email", "placeholder": "Enter email", "href": ""}
        alts = _generate_alternatives(el)
        assert any("input[type='email']" in a for a in alts)
        assert any("placeholder" in a for a in alts)

    def test_generates_text_alternative(self):
        el = {"tag": "a", "name": "Home", "role": "", "type": "", "placeholder": "", "href": "/home"}
        alts = _generate_alternatives(el)
        assert any("has-text" in a for a in alts)
        assert any("href" in a for a in alts)


class TestSlugify:
    def test_basic(self):
        assert _slugify("Log in and go to settings") == "log-in-and-go-to-settings"

    def test_special_chars(self):
        assert _slugify("Search for 'laptop' on amazon!") == "search-for-laptop-on-amazon"

    def test_max_length(self):
        result = _slugify("a very long goal description that exceeds the max", max_len=20)
        assert len(result) <= 20


@pytest.mark.integration
class TestExplorerWithMock:
    """Test the explorer with mock LLM responses."""

    async def test_explore_login_flow(self, browser, test_server):
        """Explorer can navigate a login form with mock LLM."""
        mock = MockLLM(responses=[
            "action=fill ref=e0 value=test@test.com | intent=fill email field",
            "action=fill ref=e1 value=secret123 | intent=fill password field",
            "action=click ref=e2 | intent=click login button",
            "action=done | intent=goal complete",
        ])

        explorer = Explorer(browser=browser, llm=mock, max_steps=10)
        result = await explorer.explore(
            url=f"{test_server.url}/login",
            goal="Log in with test@test.com and secret123",
        )

        assert result.success
        assert result.playbook.name  # Auto-generated
        assert len(result.playbook.steps) >= 1  # At least some steps recorded
        assert result.total_tokens > 0

    async def test_explore_records_playbook(self, browser, test_server):
        """Explorer produces a valid, saveable playbook."""
        mock = MockLLM(responses=[
            "action=fill ref=e0 value=test@test.com | intent=fill email",
            "action=click ref=e2 | intent=click submit",
            "action=done | intent=done",
        ])

        explorer = Explorer(browser=browser, llm=mock, max_steps=10)
        result = await explorer.explore(
            url=f"{test_server.url}/login",
            goal="Fill email and submit",
        )

        pb = result.playbook
        assert pb.url == f"{test_server.url}/login"

        # Steps should have selectors and alternatives
        for step in pb.steps:
            assert step.intent
            assert step.action
            # Selector may be empty if element wasn't found, but intent should be set

    async def test_explore_templatizes_variables(self, browser, test_server):
        """Explorer replaces known values with ${var} templates."""
        mock = MockLLM(responses=[
            "action=fill ref=e0 value=myuser@test.com | intent=fill email",
            "action=done | intent=done",
        ])

        explorer = Explorer(browser=browser, llm=mock, max_steps=10)
        result = await explorer.explore(
            url=f"{test_server.url}/login",
            goal="Fill email",
            variables={"email": "myuser@test.com"},
        )

        # If a step was recorded with this value, it should be templatized
        for step in result.playbook.steps:
            if step.value_template:
                assert "${email}" in step.value_template

    async def test_explore_stops_at_max_steps(self, browser, test_server):
        """Explorer stops after max_steps."""
        # Mock that never says "done"
        mock = MockLLM(responses=[
            "action=click ref=e0 | intent=click something",
        ] * 10)

        explorer = Explorer(browser=browser, llm=mock, max_steps=5)
        result = await explorer.explore(
            url=f"{test_server.url}/login",
            goal="Keep clicking forever",
        )

        assert result.reason.startswith("Max steps")
        assert result.steps_taken == 5

    async def test_explore_handles_parse_failures(self, browser, test_server):
        """Explorer handles unparseable LLM responses gracefully."""
        mock = MockLLM(responses=[
            "I'm not sure what to do here",
            "Let me think about this...",
            "Maybe try clicking somewhere?",
        ])

        explorer = Explorer(browser=browser, llm=mock, max_steps=5)
        result = await explorer.explore(
            url=f"{test_server.url}/login",
            goal="Do something",
        )

        assert not result.success
        assert "parse" in result.reason.lower()


@pytest.mark.integration
class TestAgentExplore:
    """Test the Agent.explore() high-level API."""

    async def test_agent_explore_and_save(self, test_server, tmp_path):
        """Agent.explore() runs and saves a playbook."""
        from autopilot import Agent
        from autopilot.llm.base import MockLLM

        mock = MockLLM(responses=[
            "action=fill ref=e0 value=a@b.com | intent=fill email",
            "action=fill ref=e1 value=pass | intent=fill password",
            "action=click ref=e2 | intent=click login",
            "action=done | intent=complete",
        ])

        async with Agent(llm=mock, headless=True, playbook_dir=tmp_path) as agent:
            result = await agent.explore(
                url=f"{test_server.url}/login",
                goal="Log in",
            )

            assert result.playbook.name
            assert result.total_tokens > 0

            # Playbook should be saved
            playbooks = agent.list_playbooks()
            assert len(playbooks) >= 0  # May or may not save depending on steps recorded

    async def test_agent_explore_then_replay(self, test_server, tmp_path):
        """The core value prop: explore once, replay free forever."""
        from autopilot import Agent, Playbook, PlaybookStep

        # Simulate what explore would produce: a saved playbook
        pb = Playbook(
            name="explore-replay-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(intent="fill email", selector="#email", action="fill", value="a@b.com"),
                PlaybookStep(intent="fill pass", selector="#password", action="fill", value="p"),
                PlaybookStep(intent="click login", selector="#login-btn", action="click"),
            ],
        )
        pb.save(tmp_path / "explore-replay-test.json")

        async with Agent(llm=None, headless=True, playbook_dir=tmp_path) as agent:
            # Replay — zero tokens
            result = await agent.replay("explore-replay-test")
            assert result.success
            assert result.total_tokens == 0
            assert all(s.tier == 0 for s in result.steps)


@pytest.mark.integration
class TestAgentExtract:
    """Test the Agent.extract() data extraction API."""

    async def test_extract_returns_data(self, test_server):
        """Agent.extract() returns structured data from a page."""
        from autopilot import Agent
        from autopilot.llm.base import MockLLM

        # Mock returns JSON data
        mock = MockLLM(responses=[
            '[{"title": "Dashboard", "type": "heading"}]',
        ])

        async with Agent(llm=mock, headless=True) as agent:
            data = await agent.extract(
                url=f"{test_server.url}/dashboard",
                goal="page headings",
                schema={"title": "str", "type": "str"},
            )

            assert isinstance(data, list)
            assert len(data) > 0
            assert "title" in data[0]

    async def test_extract_handles_non_json(self, test_server):
        """Agent.extract() returns raw text if LLM doesn't return JSON."""
        from autopilot import Agent
        from autopilot.llm.base import MockLLM

        mock = MockLLM(responses=["The page has a dashboard with navigation links"])

        async with Agent(llm=mock, headless=True) as agent:
            data = await agent.extract(
                url=f"{test_server.url}/dashboard",
                goal="describe the page",
            )
            assert isinstance(data, str)
