"""Demonstrates the tiered self-healing escalation.

Run this to see how autopilot handles selector breakage:
1. First run: All selectors match → Tier 0 (0 tokens)
2. Second run: Selectors changed → Tier 1 heals (still 0 tokens)
3. Third run: Healed playbook replays at Tier 0 (drift to determinism)
"""

import asyncio
import json
from pathlib import Path

from autopilot import Agent, Playbook, PlaybookStep


async def main():
    playbook = Playbook(
        name="heal-demo",
        url="https://example.com/login",
        steps=[
            PlaybookStep(
                intent="fill email field",
                selector="#email",
                selector_alternatives=["input[type='email']", "input[name='email']"],
                action="fill",
                value_template="${email}",
            ),
            PlaybookStep(
                intent="fill password field",
                selector="#password",
                selector_alternatives=["input[type='password']"],
                action="fill",
                value_template="${password}",
            ),
            PlaybookStep(
                intent="click login button",
                selector="#login-btn",
                selector_alternatives=["button[type='submit']", "role:button[name='Log In']"],
                action="click",
            ),
        ],
        variables=["email", "password"],
    )

    variables = {"email": "demo@test.com", "password": "demo123"}

    async with Agent(llm="claude-sonnet-4-20250514", max_tier=4) as agent:
        # Run 1: should succeed at Tier 0
        print("=== Run 1: Fresh playbook ===")
        result = await agent.run(playbook, variables=variables)
        print(f"Success: {result.success}")
        print(f"Tokens: {result.total_tokens}")
        print(f"Tiers: {result.tier_counts}")
        print()

        # If selectors broke, healing would kick in automatically
        for step in result.steps:
            print(f"  Step '{step.intent}': Tier {step.tier}, {step.strategy}")

        print()
        print(f"Total agent token usage: {agent.usage.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
