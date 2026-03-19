"""Using autopilot tools directly with the Claude API.

This example shows how to integrate autopilot into your own
agentic loop using the Anthropic SDK.
"""

import asyncio
import anthropic
from autopilot.integrations.claude_api import AutopilotTools


async def main():
    client = anthropic.AsyncAnthropic()
    tools = AutopilotTools(headless=False, llm_model="claude-sonnet-4-20250514")

    try:
        # Get tool definitions for Claude API
        tool_defs = tools.get_tool_definitions()

        messages = [
            {"role": "user", "content": "Go to https://news.ycombinator.com and get the page snapshot"}
        ]

        # Agentic loop
        while True:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=tool_defs,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                # Final response
                for block in response.content:
                    if hasattr(block, "text"):
                        print(block.text)
                break

            # Handle tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"Calling tool: {block.name}({block.input})")
                    result = await tools.handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    finally:
        await tools.close()


if __name__ == "__main__":
    asyncio.run(main())
