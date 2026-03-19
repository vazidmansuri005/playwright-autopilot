"""5-line quickstart for playwright-autopilot.

This example shows the simplest way to automate a browser.
First run: AI explores and records a playbook (~25K tokens).
Second run: Deterministic replay (~0 tokens).
"""

import asyncio
from autopilot import Agent, Playbook, PlaybookStep


async def main():
    # === Option 1: Run an existing playbook ===
    playbook = Playbook(
        name="example-login",
        url="https://example.com/login",
        steps=[
            PlaybookStep(intent="fill email", selector="#email", action="fill", value_template="${email}"),
            PlaybookStep(intent="fill password", selector="#password", action="fill", value_template="${password}"),
            PlaybookStep(intent="click login", selector="button[type='submit']", action="click"),
        ],
    )

    async with Agent(llm="claude-sonnet-4-20250514", headless=False) as agent:
        result = await agent.run(
            playbook,
            variables={"email": "user@example.com", "password": "secret"},
        )
        print(f"Success: {result.success}")
        print(f"Tokens used: {result.total_tokens}")
        print(f"Tier distribution: {result.tier_counts}")


if __name__ == "__main__":
    asyncio.run(main())
