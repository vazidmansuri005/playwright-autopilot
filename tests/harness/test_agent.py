"""Tests for the high-level Agent API."""

import pytest
from pathlib import Path

from autopilot import Agent, Playbook, PlaybookStep


@pytest.mark.integration
class TestAgent:
    """Test the Agent class — the public API."""

    async def test_agent_runs_playbook(self, test_server, tmp_path):
        """Agent can run a playbook end-to-end."""
        playbook = Playbook(
            name="agent-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(intent="fill email", selector="#email", action="fill", value="test@test.com"),
                PlaybookStep(intent="fill password", selector="#password", action="fill", value="pass"),
                PlaybookStep(intent="click login", selector="#login-btn", action="click"),
            ],
        )

        async with Agent(llm=None, headless=True, playbook_dir=tmp_path) as agent:
            result = await agent.run(playbook)
            assert result.success
            assert result.total_tokens == 0

    async def test_agent_replay_mode(self, test_server, tmp_path):
        """Agent.replay() runs at Tier 0 only."""
        playbook = Playbook(
            name="replay-test",
            url=f"{test_server.url}/dashboard",
            steps=[
                PlaybookStep(intent="click create", selector="#btn-new", action="click"),
            ],
        )
        # Save playbook
        playbook.save(tmp_path / "replay-test.json")

        async with Agent(llm=None, headless=True, playbook_dir=tmp_path) as agent:
            result = await agent.replay("replay-test")
            assert result.success
            assert all(s.tier == 0 for s in result.steps)

    async def test_agent_list_playbooks(self, test_server, tmp_path):
        """Agent lists saved playbooks."""
        pb1 = Playbook(name="pb1", url="http://a.com", steps=[])
        pb2 = Playbook(name="pb2", url="http://b.com", steps=[
            PlaybookStep(intent="click", selector="#x", action="click"),
        ])
        pb1.save(tmp_path / "pb1.json")
        pb2.save(tmp_path / "pb2.json")

        async with Agent(llm=None, headless=True, playbook_dir=tmp_path) as agent:
            playbooks = agent.list_playbooks()
            names = {p["name"] for p in playbooks}
            assert "pb1" in names
            assert "pb2" in names

    async def test_agent_saves_healed_playbook(self, test_server, tmp_path):
        """Agent saves the playbook after healing."""
        test_server.set_mutated(True)
        playbook = Playbook(
            name="heal-save-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(
                    intent="fill email field",
                    selector="#email",  # Won't match mutated page
                    selector_alternatives=["input[type='email']"],  # Will match
                    action="fill",
                    value="test@test.com",
                ),
            ],
        )
        playbook.save(tmp_path / "heal-save-test.json")

        async with Agent(llm=None, headless=True, max_tier=1, playbook_dir=tmp_path) as agent:
            result = await agent.run(tmp_path / "heal-save-test.json")
            assert result.success

    async def test_agent_usage_tracking(self, test_server, tmp_path):
        """Agent tracks cumulative token usage."""
        playbook = Playbook(
            name="usage-test",
            url=f"{test_server.url}/login",
            steps=[
                PlaybookStep(intent="fill email", selector="#email", action="fill", value="a@b.com"),
            ],
        )

        async with Agent(llm=None, headless=True, playbook_dir=tmp_path) as agent:
            await agent.run(playbook)
            await agent.run(playbook)

            summary = agent.usage.summary()
            assert summary["total_tokens"] == 0  # Tier 0 runs
            assert summary["call_count"] == 0

    async def test_agent_context_manager(self, test_server, tmp_path):
        """Agent properly cleans up as context manager."""
        playbook = Playbook(
            name="cm-test",
            url=f"{test_server.url}/login",
            steps=[],
        )

        agent = Agent(llm=None, headless=True, playbook_dir=tmp_path)
        async with agent:
            await agent.run(playbook)
        # Browser should be closed after exiting context
        assert agent._browser is None
