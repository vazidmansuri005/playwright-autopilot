"""Interactive REPL — type natural language, watch the browser respond.

Usage:
    autopilot interactive --url https://myapp.com --headed

Every action is recorded to a playbook. Type 'save <name>' to persist.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.digest import extract_digest
from autopilot.core.explorer import Explorer, _parse_explore_response, _slugify
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import _build_selector, _find_by_snapshot_ref
from autopilot.llm.base import BaseLLM
from autopilot.llm.factory import create_llm

logger = logging.getLogger(__name__)

REPL_SYSTEM = """\
You are a browser automation assistant in interactive mode.
The user types natural language commands. Convert each to a browser action.

Respond in EXACTLY this format:
action=ACTION ref=REF [value=VALUE] | intent=DESCRIPTION

Where ACTION is: click, fill, select, press, check, hover, scroll, wait, navigate, done
REF is an element reference from the page digest (e.g., e5)
VALUE is required for fill/select/press/navigate actions

If the user says "save", "done", or "quit", respond: action=done | intent=save and exit
If the user wants to navigate, respond: action=navigate value=URL | intent=go to URL
"""


async def run_repl(
    url: str,
    llm_model: str,
    headless: bool = False,
) -> Playbook | None:
    """Run the interactive REPL session."""
    llm = create_llm(llm_model)
    if llm is None:
        print("Error: REPL requires an LLM. Provide --llm argument.")
        return None

    config = BrowserConfig(headless=headless)
    browser = Browser(config)
    await browser.start()

    playbook = Playbook(name="interactive", url=url)

    try:
        page = browser.page
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(1)

        print(f"\n  autopilot interactive — {url}")
        print(f"  Type natural language commands. 'save <name>' to save. 'quit' to exit.\n")

        step_num = 0
        while True:
            try:
                user_input = input("autopilot> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                break

            if not user_input:
                continue

            # Built-in commands
            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.lower().startswith("save"):
                parts = user_input.split(maxsplit=1)
                name = parts[1] if len(parts) > 1 else _slugify(url)
                playbook.name = name
                from pathlib import Path
                save_dir = Path.home() / ".autopilot" / "playbooks"
                save_dir.mkdir(parents=True, exist_ok=True)
                path = save_dir / f"{name}.json"
                playbook.save(path)
                print(f"  Saved: {path} ({len(playbook.steps)} steps)")
                continue

            if user_input.lower() == "steps":
                for i, s in enumerate(playbook.steps):
                    print(f"  {i+1}. {s.intent} ({s.action} {s.selector})")
                continue

            if user_input.lower() == "undo":
                if playbook.steps:
                    removed = playbook.steps.pop()
                    print(f"  Removed: {removed.intent}")
                else:
                    print("  No steps to undo.")
                continue

            # Get page state
            digest = await extract_digest(page, max_elements=50, visible_only=True)

            # Ask LLM
            prompt = (
                f'User command: "{user_input}"\n\n'
                f"Current page:\n{digest.to_prompt()}\n\n"
                f"Convert to browser action (one line):"
            )
            response = await llm.complete(prompt, system=REPL_SYSTEM)
            parsed = _parse_explore_response(response.text)

            if not parsed:
                print(f"  Could not parse: {response.text[:100]}")
                continue

            action = parsed["action"]
            intent = parsed.get("intent", user_input)
            ref_idx = parsed.get("ref")
            value = parsed.get("value", "")

            if action == "done":
                break

            if action == "navigate":
                await page.goto(value, wait_until="domcontentloaded")
                await asyncio.sleep(1)
                print(f"  Navigated to {value}")
                continue

            if action == "wait":
                await asyncio.sleep(2)
                print(f"  Waited 2s")
                continue

            # Execute
            selector = ""
            success = False

            if ref_idx is not None and ref_idx < len(digest.elements):
                element_data = digest.elements[ref_idx]
                selector = _build_selector(element_data)
                locator = await _find_by_snapshot_ref(page, ref_idx)
                if locator:
                    try:
                        match action:
                            case "click":
                                await locator.click(timeout=5000)
                            case "fill" | "type":
                                await locator.fill(value, timeout=5000)
                            case "select":
                                await locator.select_option(value, timeout=5000)
                            case "press":
                                await locator.press(value)
                            case "check":
                                await locator.check(timeout=5000)
                            case "hover":
                                await locator.hover(timeout=5000)
                        success = True
                    except Exception as e:
                        print(f"  Failed: {e}")

            if success:
                step_num += 1
                step = PlaybookStep(
                    intent=intent, selector=selector, action=action,
                    value=value if action in ("fill", "select", "press") else None,
                    wait_after_ms=500,
                )
                playbook.add_step(step)
                print(f"  {step_num}. {intent} ({selector})")
                await asyncio.sleep(0.5)
            else:
                print(f"  Could not execute: {action} on ref e{ref_idx}")

    finally:
        await browser.close()

    if playbook.steps:
        return playbook
    return None
