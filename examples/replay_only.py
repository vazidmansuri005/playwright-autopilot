"""Replay a playbook with zero tokens — no LLM needed.

This is the most cost-effective mode. Use it for:
- CI/CD pipelines
- Regression testing
- Any flow where selectors are stable
"""

import asyncio
from autopilot import Agent


async def main():
    # No LLM needed for pure replay
    async with Agent(llm=None, headless=True) as agent:
        # List available playbooks
        playbooks = agent.list_playbooks()
        print(f"Available playbooks: {len(playbooks)}")
        for pb in playbooks:
            print(f"  - {pb['name']}: {pb['steps']} steps, {pb['success_rate']:.0%} success rate")

        if playbooks:
            # Replay the first one
            result = await agent.replay(
                playbooks[0]["name"],
                variables={"email": "test@test.com", "password": "test"},
            )
            print(f"\nResult: {'PASS' if result.success else 'FAIL'}")
            print(f"Tokens: {result.total_tokens}")  # Should be 0
            print(f"Duration: {result.total_duration_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
