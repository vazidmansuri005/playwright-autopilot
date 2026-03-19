"""Code Generator — converts playbooks to real, standalone Playwright scripts.

This is the bridge between MCP and CLI. The AI explores (MCP-like),
records a playbook, then generates a REAL Playwright script (CLI-like)
that runs independently with zero dependency on autopilot.

The generated script:
  - Is standard Playwright Python/TypeScript
  - Runs via `pytest` or `npx playwright test`
  - Works with trace viewer, reporters, CI/CD
  - Has NO autopilot dependency
  - Includes resilient selectors (primary + fallbacks)
  - Has proper waits, error handling, and assertions

Usage:
    from autopilot.codegen import generate_python, generate_typescript

    # From a playbook
    code = generate_python(playbook, variables={"email": "test@test.com"})

    # From explore result
    agent = Agent(llm="claude-sonnet-4-20250514")
    result = await agent.explore(url="...", goal="...")
    code = generate_python(result.playbook)

    # Save and run independently
    Path("test_login.py").write_text(code)
    # pytest test_login.py  ← runs without autopilot!

CLI:
    autopilot generate login-flow.json --lang python --output test_login.py
    autopilot generate login-flow.json --lang typescript --output test_login.spec.ts
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent, indent

from autopilot.core.playbook import Playbook, PlaybookStep


def generate_python(
    playbook: Playbook,
    variables: dict[str, str] | None = None,
    parametrize: bool = True,
) -> str:
    """Generate a standalone Playwright Python test from a playbook.

    The output is a complete pytest file that runs WITHOUT autopilot.
    """
    variables = variables or {}
    var_lines = ""
    if playbook.variables and parametrize:
        var_dict = {v: variables.get(v, f"<{v}>") for v in playbook.variables}
        var_lines = f"\n# Variables — change these or parametrize with pytest\nVARIABLES = {json.dumps(var_dict, indent=4)}\n"

    steps_code = []
    for i, step in enumerate(playbook.steps):
        code = _step_to_python(step, i, variables, parametrize)
        steps_code.append(code)

    steps_joined = "\n\n".join(steps_code)

    script = f'''\
"""Auto-generated Playwright test from autopilot playbook: {playbook.name}

Generated from: {playbook.url}
Steps: {len(playbook.steps)}
Run with: pytest {playbook.name.replace("-", "_")}.py -v

NO DEPENDENCY ON AUTOPILOT — this is standard Playwright.
"""

import re
import pytest
from playwright.sync_api import Page, expect
{var_lines}

@pytest.fixture(scope="module")
def browser_context(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    yield context
    context.close()
    browser.close()


@pytest.fixture
def page(browser_context):
    page = browser_context.new_page()
    page.set_default_timeout(15000)
    yield page
    page.close()


def test_{_sanitize_name(playbook.name)}(page: Page):
    """Test: {playbook.name}"""
    # Navigate to starting URL
    page.goto("{playbook.url}")
    page.wait_for_load_state("domcontentloaded")

{indent(steps_joined, "    ")}
'''
    return script


def generate_typescript(
    playbook: Playbook,
    variables: dict[str, str] | None = None,
) -> str:
    """Generate a standalone Playwright TypeScript test from a playbook."""
    variables = variables or {}

    steps_code = []
    for i, step in enumerate(playbook.steps):
        code = _step_to_typescript(step, i, variables)
        steps_code.append(code)

    steps_joined = "\n\n".join(steps_code)

    var_block = ""
    if playbook.variables:
        var_dict = {v: variables.get(v, f"<{v}>") for v in playbook.variables}
        var_block = f"const VARIABLES = {json.dumps(var_dict, indent=2)};\n"

    script = f'''\
/**
 * Auto-generated Playwright test from autopilot playbook: {playbook.name}
 *
 * Generated from: {playbook.url}
 * Steps: {len(playbook.steps)}
 * Run with: npx playwright test {playbook.name}.spec.ts
 *
 * NO DEPENDENCY ON AUTOPILOT — this is standard Playwright.
 */

import {{ test, expect }} from '@playwright/test';

{var_block}
test('{playbook.name}', async ({{ page }}) => {{
  // Navigate to starting URL
  await page.goto('{playbook.url}');
  await page.waitForLoadState('domcontentloaded');

{indent(steps_joined, "  ")}
}});
'''
    return script


def generate_from_file(
    playbook_path: str | Path,
    lang: str = "python",
    variables: dict[str, str] | None = None,
) -> str:
    """Generate code from a playbook JSON file."""
    playbook = Playbook.load(playbook_path)
    if lang == "python":
        return generate_python(playbook, variables)
    elif lang in ("typescript", "ts"):
        return generate_typescript(playbook, variables)
    else:
        raise ValueError(f"Unsupported language: {lang}. Use 'python' or 'typescript'.")


# --- Internal helpers ---

def _step_to_python(step: PlaybookStep, index: int, variables: dict, parametrize: bool) -> str:
    """Convert a single step to Python Playwright code."""
    lines = [f"# Step {index + 1}: {step.intent}"]

    # Conditional
    if step.condition == "if_visible":
        lines.append(f'if page.locator("{_escape(step.selector)}").count() > 0:')
        inner_lines = _action_to_python(step, variables, parametrize)
        for line in inner_lines:
            lines.append(f"    {line}")
        return "\n".join(lines)

    if step.skip_on_fail:
        lines.append("try:")
        inner_lines = _action_to_python(step, variables, parametrize)
        for line in inner_lines:
            lines.append(f"    {line}")
        lines.append("except Exception:")
        lines.append(f'    pass  # skip_on_fail: {step.intent}')
        return "\n".join(lines)

    lines.extend(_action_to_python(step, variables, parametrize))

    # Wait after
    if step.wait_after_ms > 0:
        lines.append(f"page.wait_for_timeout({step.wait_after_ms})")

    # Assertion
    if step.assert_after:
        assertion_code = _assertion_to_python(step.assert_after)
        if assertion_code:
            lines.append(assertion_code)

    return "\n".join(lines)


def _action_to_python(step: PlaybookStep, variables: dict, parametrize: bool) -> list[str]:
    """Generate the action lines for a Python step."""
    lines = []

    # Use resilient locator chain
    sel = _escape(step.selector)
    alts = step.selector_alternatives[:3]  # Top 3 alternatives

    # Build locator with fallback
    if alts:
        locator_expr = _build_resilient_locator_python(step)
    else:
        locator_expr = f'page.locator("{sel}")'

    # Resolve value
    value = step.resolve_value(variables) or step.value or ""
    if parametrize and step.value_template:
        # Use VARIABLES dict
        var_name = re.search(r"\$\{(\w+)\}", step.value_template)
        if var_name:
            value_expr = f'VARIABLES["{var_name.group(1)}"]'
        else:
            value_expr = f'"{_escape(value)}"'
    else:
        value_expr = f'"{_escape(value)}"'

    match step.action:
        case "click":
            lines.append(f'{locator_expr}.click()')
        case "fill" | "type":
            lines.append(f'{locator_expr}.fill({value_expr})')
        case "select":
            lines.append(f'{locator_expr}.select_option({value_expr})')
        case "press":
            lines.append(f'{locator_expr}.press("{_escape(value)}")')
        case "check":
            lines.append(f'{locator_expr}.check()')
        case "uncheck":
            lines.append(f'{locator_expr}.uncheck()')
        case "hover":
            lines.append(f'{locator_expr}.hover()')
        case "clear":
            lines.append(f'{locator_expr}.clear()')

    return lines


def _build_resilient_locator_python(step: PlaybookStep) -> str:
    """Build a resilient locator with fallback — tries primary, then alternatives."""
    sel = _escape(step.selector)

    # Check if we can use Playwright's built-in resilient locators
    for alt in [step.selector] + step.selector_alternatives:
        if alt.startswith("role:"):
            match = re.match(r"role:(\w+)\[name=['\"]?(.+?)['\"]?\]", alt)
            if match:
                role, name = match.groups()
                return f'page.get_by_role("{role}", name="{_escape(name)}")'

        if alt.startswith("label="):
            label = alt[6:]
            return f'page.get_by_label("{_escape(label)}")'

        if alt.startswith("placeholder="):
            ph = alt[12:]
            return f'page.get_by_placeholder("{_escape(ph)}")'

        if alt.startswith("text="):
            text = alt[5:]
            return f'page.get_by_text("{_escape(text)}")'

    # Fallback to CSS with .or_()
    if step.selector_alternatives:
        first_alt = _escape(step.selector_alternatives[0])
        return f'page.locator("{sel}").or_(page.locator("{first_alt}"))'

    return f'page.locator("{sel}")'


def _step_to_typescript(step: PlaybookStep, index: int, variables: dict) -> str:
    """Convert a single step to TypeScript Playwright code."""
    lines = [f"// Step {index + 1}: {step.intent}"]

    sel = _escape(step.selector)
    value = step.resolve_value(variables) or step.value or ""

    if step.condition == "if_visible":
        lines.append(f'if (await page.locator("{sel}").count() > 0) {{')

    locator = _build_resilient_locator_ts(step)

    match step.action:
        case "click":
            lines.append(f"await {locator}.click();")
        case "fill" | "type":
            lines.append(f'await {locator}.fill("{_escape(value)}");')
        case "select":
            lines.append(f'await {locator}.selectOption("{_escape(value)}");')
        case "press":
            lines.append(f'await {locator}.press("{_escape(value)}");')
        case "check":
            lines.append(f"await {locator}.check();")
        case "uncheck":
            lines.append(f"await {locator}.uncheck();")

    if step.condition == "if_visible":
        lines.append("}")

    if step.wait_after_ms > 0:
        lines.append(f"await page.waitForTimeout({step.wait_after_ms});")

    if step.assert_after:
        ts_assert = _assertion_to_typescript(step.assert_after)
        if ts_assert:
            lines.append(ts_assert)

    return "\n".join(lines)


def _build_resilient_locator_ts(step: PlaybookStep) -> str:
    """Build a resilient TypeScript locator."""
    for alt in [step.selector] + step.selector_alternatives:
        if alt.startswith("role:"):
            match = re.match(r"role:(\w+)\[name=['\"]?(.+?)['\"]?\]", alt)
            if match:
                role, name = match.groups()
                return f"page.getByRole('{role}', {{ name: '{_escape(name)}' }})"

        if alt.startswith("label="):
            return f"page.getByLabel('{_escape(alt[6:])}')"

        if alt.startswith("placeholder="):
            return f"page.getByPlaceholder('{_escape(alt[12:])}')"

        if alt.startswith("text="):
            return f"page.getByText('{_escape(alt[5:])}')"

    return f"page.locator('{_escape(step.selector)}')"


def _assertion_to_python(assertion: str) -> str | None:
    """Convert a natural language assertion to Playwright expect()."""
    a = assertion.lower().strip()

    if "url contains" in a:
        match = re.search(r"url\s+contains?\s+['\"]?(.+?)['\"]?$", a)
        if match:
            return f'expect(page).to_have_url(re.compile(r"{_escape(match.group(1))}"))'

    if "title contains" in a:
        match = re.search(r"title\s+contains?\s+['\"]?(.+?)['\"]?$", a)
        if match:
            return f'expect(page).to_have_title(re.compile(r"{_escape(match.group(1))}"))'

    if "is visible" in a:
        match = re.search(r"['\"](.+?)['\"] is visible", a)
        if match:
            return f'expect(page.get_by_text("{_escape(match.group(1))}")).to_be_visible()'

    return f"# TODO: assert {assertion}"


def _assertion_to_typescript(assertion: str) -> str | None:
    """Convert assertion to TypeScript Playwright expect()."""
    a = assertion.lower().strip()

    if "url contains" in a:
        match = re.search(r"url\s+contains?\s+['\"]?(.+?)['\"]?$", a)
        if match:
            return f"await expect(page).toHaveURL(/{_escape(match.group(1))}/);"

    if "is visible" in a:
        match = re.search(r"['\"](.+?)['\"] is visible", a)
        if match:
            return f"await expect(page.getByText('{_escape(match.group(1))}')).toBeVisible();"

    return f"// TODO: assert {assertion}"


def _escape(s: str) -> str:
    """Escape special characters for string literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")


def _sanitize_name(name: str) -> str:
    """Convert a playbook name to a valid Python function name."""
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()
