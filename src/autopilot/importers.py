"""Import existing test scripts as autopilot playbooks.

Supports:
  - Playwright codegen recordings (.ts, .py)
  - Playwright test files
  - Generic Selenium/Puppeteer (best-effort)

Usage:
    from autopilot.importers import import_playwright

    playbook = import_playwright("recording.ts")
    playbook.save("converted.json")

CLI:
    autopilot import recording.ts --output login-flow.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from autopilot.core.playbook import Playbook, PlaybookStep

logger = logging.getLogger(__name__)


def import_playwright(path: str | Path, name: str | None = None) -> Playbook:
    """Import a Playwright codegen recording or test file as a Playbook.

    Handles both TypeScript and Python Playwright scripts.
    Extracts page.goto, page.fill, page.click, page.selectOption, etc.
    """
    path = Path(path)
    content = path.read_text()
    ext = path.suffix.lower()

    if ext in (".ts", ".js", ".mjs"):
        steps = _parse_playwright_ts(content)
    elif ext == ".py":
        steps = _parse_playwright_py(content)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .ts, .js, or .py")

    if not steps:
        raise ValueError(f"No Playwright actions found in {path}")

    # Extract starting URL
    url = ""
    for step in steps:
        if step.action == "navigate":
            url = step.value or ""
            break

    # Filter out navigation steps (they become the playbook URL)
    action_steps = [s for s in steps if s.action != "navigate"]

    pb_name = name or path.stem
    playbook = Playbook(name=pb_name, url=url, steps=action_steps)
    playbook.extract_variables()

    logger.info("Imported %d steps from %s", len(action_steps), path)
    return playbook


def _parse_playwright_ts(content: str) -> list[PlaybookStep]:
    """Parse TypeScript/JavaScript Playwright actions."""
    steps: list[PlaybookStep] = []

    # page.goto('url')
    for m in re.finditer(r"page\.goto\(['\"](.+?)['\"]\)", content):
        steps.append(PlaybookStep(
            intent="navigate to page",
            selector="", action="navigate", value=m.group(1),
        ))

    # page.fill('selector', 'value') or page.locator('sel').fill('val')
    for m in re.finditer(r"(?:page\.fill|\.fill)\(['\"](.+?)['\"]\s*,\s*['\"](.+?)['\"]\)", content):
        sel, val = m.group(1), m.group(2)
        intent = _infer_intent("fill", sel, val)
        steps.append(PlaybookStep(
            intent=intent, selector=sel, action="fill", value=val,
            selector_alternatives=_generate_alt_selectors(sel),
        ))

    # page.click('selector') or page.locator('sel').click()
    for m in re.finditer(r"(?:page\.click|\.click)\(['\"](.+?)['\"]\)", content):
        sel = m.group(1)
        intent = _infer_intent("click", sel)
        steps.append(PlaybookStep(
            intent=intent, selector=sel, action="click",
            selector_alternatives=_generate_alt_selectors(sel),
        ))

    # page.selectOption('selector', 'value')
    for m in re.finditer(r"(?:page\.selectOption|\.selectOption)\(['\"](.+?)['\"]\s*,\s*['\"](.+?)['\"]\)", content):
        sel, val = m.group(1), m.group(2)
        intent = _infer_intent("select", sel, val)
        steps.append(PlaybookStep(
            intent=intent, selector=sel, action="select", value=val,
            selector_alternatives=_generate_alt_selectors(sel),
        ))

    # page.press('selector', 'key')
    for m in re.finditer(r"(?:page\.press|\.press)\(['\"](.+?)['\"]\s*,\s*['\"](.+?)['\"]\)", content):
        sel, key = m.group(1), m.group(2)
        steps.append(PlaybookStep(
            intent=f"press {key}", selector=sel, action="press", value=key,
        ))

    # page.check('selector')
    for m in re.finditer(r"(?:page\.check|\.check)\(['\"](.+?)['\"]\)", content):
        sel = m.group(1)
        steps.append(PlaybookStep(
            intent=_infer_intent("check", sel), selector=sel, action="check",
        ))

    # page.getByRole('role', { name: 'name' }).click()
    for m in re.finditer(
        r"page\.getByRole\(['\"](\w+)['\"]\s*,\s*\{\s*name:\s*['\"](.+?)['\"]\s*\}\)\.(\w+)\(",
        content,
    ):
        role, name, action = m.group(1), m.group(2), m.group(3)
        sel = f"role:{role}[name='{name}']"
        steps.append(PlaybookStep(
            intent=f"{action} {name} {role}", selector=sel, action=action,
        ))

    # page.getByLabel('label').fill('value')
    for m in re.finditer(r"page\.getByLabel\(['\"](.+?)['\"]\)\.(\w+)\(['\"]?(.+?)?['\"]?\)", content):
        label, action, val = m.group(1), m.group(2), m.group(3)
        sel = f"label={label}"
        steps.append(PlaybookStep(
            intent=f"{action} {label} field", selector=sel, action=action, value=val,
        ))

    # page.getByText('text').click()
    for m in re.finditer(r"page\.getByText\(['\"](.+?)['\"]\)\.(\w+)\(", content):
        text, action = m.group(1), m.group(2)
        sel = f"text={text}"
        steps.append(PlaybookStep(
            intent=f"{action} '{text}'", selector=sel, action=action,
        ))

    return steps


def _parse_playwright_py(content: str) -> list[PlaybookStep]:
    """Parse Python Playwright actions."""
    steps: list[PlaybookStep] = []

    # page.goto("url")
    for m in re.finditer(r'page\.goto\(["\'](.+?)["\']\)', content):
        steps.append(PlaybookStep(
            intent="navigate to page", selector="", action="navigate", value=m.group(1),
        ))

    # page.fill("selector", "value")
    for m in re.finditer(r'page\.fill\(["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\)', content):
        sel, val = m.group(1), m.group(2)
        steps.append(PlaybookStep(
            intent=_infer_intent("fill", sel, val), selector=sel, action="fill", value=val,
            selector_alternatives=_generate_alt_selectors(sel),
        ))

    # page.click("selector")
    for m in re.finditer(r'page\.click\(["\'](.+?)["\']\)', content):
        sel = m.group(1)
        steps.append(PlaybookStep(
            intent=_infer_intent("click", sel), selector=sel, action="click",
            selector_alternatives=_generate_alt_selectors(sel),
        ))

    # page.select_option("selector", "value")
    for m in re.finditer(r'page\.select_option\(["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\)', content):
        sel, val = m.group(1), m.group(2)
        steps.append(PlaybookStep(
            intent=_infer_intent("select", sel, val), selector=sel, action="select", value=val,
        ))

    # page.get_by_role("role", name="name").click()
    for m in re.finditer(r'page\.get_by_role\(["\'](\w+)["\']\s*,\s*name=["\'](.+?)["\']\)\.(\w+)\(', content):
        role, name, action = m.group(1), m.group(2), m.group(3)
        sel = f"role:{role}[name='{name}']"
        steps.append(PlaybookStep(
            intent=f"{action} {name} {role}", selector=sel, action=action,
        ))

    # page.get_by_label("label").fill("value")
    for m in re.finditer(r'page\.get_by_label\(["\'](.+?)["\']\)\.(\w+)\(["\']?(.+?)?["\']?\)', content):
        label, action, val = m.group(1), m.group(2), m.group(3)
        steps.append(PlaybookStep(
            intent=f"{action} {label} field", selector=f"label={label}", action=action, value=val,
        ))

    return steps


def _infer_intent(action: str, selector: str, value: str = "") -> str:
    """Infer human-readable intent from action + selector."""
    # Extract meaningful name from selector
    name = ""
    for pattern in [
        r'#([\w-]+)',           # ID
        r"\[name=['\"]?(\w+)",  # name attribute
        r"\[placeholder=['\"]?(.+?)['\"]",  # placeholder
        r":has-text\(['\"](.+?)['\"]\)",     # text content
        r"label=(.+)",          # label
        r"text=(.+)",           # text
    ]:
        m = re.search(pattern, selector)
        if m:
            name = m.group(1).replace("-", " ").replace("_", " ")
            break

    if not name:
        name = selector[:30]

    if value and action == "fill":
        return f"fill {name} field"
    elif action == "click":
        return f"click {name}"
    elif action == "select":
        return f"select {value} in {name}"
    elif action == "check":
        return f"check {name}"
    else:
        return f"{action} {name}"


def _generate_alt_selectors(selector: str) -> list[str]:
    """Generate alternative selectors from a CSS selector."""
    alts = []

    # If it's an ID selector, also try by name
    id_match = re.match(r"#([\w-]+)", selector)
    if id_match:
        name = id_match.group(1)
        alts.append(f"[name='{name}']")
        alts.append(f"[id='{name}']")

    return alts
