"""Stress test: Run autopilot against multiple real websites with complex flows.

Tests the tool against REAL sites to validate claims before going public.
Each test is a progressively harder challenge.

Run with:
    ANTHROPIC_API_KEY=... pytest tests/real_world/test_stress.py -v -s --timeout=120
"""

import asyncio
import json
import os
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No ANTHROPIC_API_KEY set",
)

from autopilot import Agent
from autopilot.codegen import generate_python

LLM = os.environ.get("AUTOPILOT_LLM", "claude-sonnet-4-20250514")
RESULTS: list[dict] = []


def report(name, result, playbook=None, generated_code=None):
    """Print and collect test results."""
    entry = {
        "test": name,
        "success": result.success if hasattr(result, "success") else bool(result),
        "steps": len(result.playbook.steps) if hasattr(result, "playbook") else 0,
        "tokens": result.total_tokens if hasattr(result, "total_tokens") else 0,
    }
    RESULTS.append(entry)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Success: {entry['success']}")
    print(f"  Steps:   {entry['steps']}")
    print(f"  Tokens:  {entry['tokens']}")
    if playbook and playbook.steps:
        for i, s in enumerate(playbook.steps):
            print(f"    {i+1}. {s.intent} ({s.action} → {s.selector[:50]})")
    if generated_code:
        lines = generated_code.strip().split("\n")
        print(f"  Generated script: {len(lines)} lines")
    print(f"{'='*60}")


# =====================================================
# TEST 1: Simple search (Wikipedia)
# Difficulty: Easy — clean HTML, good accessibility
# =====================================================
class TestLevel1Easy:

    async def test_wikipedia_search(self, tmp_path):
        """Search Wikipedia for 'Playwright testing'."""
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=15000) as agent:
            result = await agent.explore(
                url="https://en.wikipedia.org",
                goal="Search for 'Playwright testing framework' using the search box",
                max_steps=5,
            )
            report("Wikipedia Search", result, result.playbook)

            if result.playbook.steps:
                code = generate_python(result.playbook)
                report("Wikipedia Codegen", result, generated_code=code)
                assert "page.g" in code or "page.locator" in code

    async def test_github_navigation(self, tmp_path):
        """Navigate GitHub explore page."""
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=15000) as agent:
            result = await agent.explore(
                url="https://github.com/explore",
                goal="Click on the Trending repositories tab",
                max_steps=5,
            )
            report("GitHub Trending Nav", result, result.playbook)


# =====================================================
# TEST 2: Form interaction (DemoQA)
# Difficulty: Medium — multiple form fields, dropdowns
# =====================================================
class TestLevel2Medium:

    async def test_demoqa_form(self, tmp_path):
        """Fill out a practice form on DemoQA."""
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=15000) as agent:
            result = await agent.explore(
                url="https://demoqa.com/text-box",
                goal="Fill the form: name='John Doe', email='john@test.com', current address='123 Main St', then click Submit",
                variables={"name": "John Doe", "email": "john@test.com", "address": "123 Main St"},
                max_steps=8,
            )
            report("DemoQA Form Fill", result, result.playbook)

            if result.playbook.steps:
                code = generate_python(result.playbook)
                (tmp_path / "test_demoqa.py").write_text(code)
                print(f"  Script saved: {tmp_path / 'test_demoqa.py'}")

    async def test_saucedemo_login(self, tmp_path):
        """Login to SauceDemo (standard test site)."""
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=15000) as agent:
            result = await agent.explore(
                url="https://www.saucedemo.com",
                goal="Log in with username 'standard_user' and password 'secret_sauce'",
                variables={"username": "standard_user", "password": "secret_sauce"},
                max_steps=6,
            )
            report("SauceDemo Login", result, result.playbook)


# =====================================================
# TEST 3: Multi-step e-commerce flow (SauceDemo)
# Difficulty: Hard — login + browse + cart + checkout
# =====================================================
class TestLevel3Hard:

    async def test_saucedemo_checkout(self, tmp_path):
        """Full e-commerce flow: login → add to cart → checkout."""
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=20000) as agent:
            result = await agent.explore(
                url="https://www.saucedemo.com",
                goal=(
                    "Log in with 'standard_user' / 'secret_sauce', "
                    "click on the first product, add it to cart, "
                    "go to cart, click checkout"
                ),
                variables={"username": "standard_user", "password": "secret_sauce"},
                max_steps=12,
            )
            report("SauceDemo Full Checkout", result, result.playbook)

            # Generate script and verify it's valid
            if result.playbook.steps:
                code = generate_python(result.playbook)
                (tmp_path / "test_checkout.py").write_text(code)
                assert "def test_" in code
                assert "page.goto" in code


# =====================================================
# TEST 4: Data extraction
# Difficulty: Medium — structured data from real page
# =====================================================
class TestLevel4Extract:

    async def test_extract_hackernews(self, tmp_path):
        """Extract top stories from Hacker News."""
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=15000) as agent:
            data = await agent.extract(
                url="https://news.ycombinator.com",
                goal="Top 3 stories with title and points",
                schema={"title": "str", "points": "str"},
            )
            print(f"\n  HN Extract Result:")
            if isinstance(data, list):
                for item in data[:3]:
                    print(f"    {item}")
                assert len(data) >= 1
            else:
                print(f"    Raw: {str(data)[:200]}")

            report("HN Data Extract", type("R", (), {"success": isinstance(data, list) and len(data) > 0, "total_tokens": agent.usage.total_tokens})())

    async def test_extract_quotes(self, tmp_path):
        """Extract quotes from quotes.toscrape.com."""
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=15000) as agent:
            data = await agent.extract(
                url="http://quotes.toscrape.com",
                goal="First 3 quotes with text, author, and tags",
                schema={"text": "str", "author": "str", "tags": "list"},
            )
            print(f"\n  Quotes Extract Result:")
            if isinstance(data, list):
                for item in data[:3]:
                    print(f"    {item.get('author', 'N/A')}: {str(item.get('text', ''))[:60]}")
                assert len(data) >= 1

            report("Quotes Extract", type("R", (), {"success": isinstance(data, list) and len(data) > 0, "total_tokens": agent.usage.total_tokens})())


# =====================================================
# TEST 5: Explore → Generate → Run independently
# Difficulty: The real test — full pipeline
# =====================================================
class TestLevel5FullPipeline:

    async def test_explore_generate_run_wikipedia(self, tmp_path):
        """Full pipeline: explore → generate → run the script independently."""
        # Phase 1: Explore
        async with Agent(llm=LLM, headless=True, playbook_dir=tmp_path, timeout=15000) as agent:
            result = await agent.explore(
                url="https://en.wikipedia.org",
                goal="Search for 'Python programming language'",
                variables={"query": "Python programming language"},
                max_steps=5,
            )
            explore_tokens = agent.usage.total_tokens

        if not result.playbook.steps:
            pytest.skip("Explorer failed to record steps")

        # Phase 2: Generate script
        code = generate_python(result.playbook, variables={"query": "Python programming language"})
        script_path = tmp_path / "test_wiki.py"
        script_path.write_text(code)

        # Phase 3: Run independently
        import subprocess
        proc = subprocess.run(
            ["python", "-m", "pytest", str(script_path), "-v", "--timeout=30"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": ""},
        )
        passed = proc.returncode == 0

        print(f"\n  Pipeline Result:")
        print(f"    Explore: {len(result.playbook.steps)} steps, {explore_tokens} tokens")
        print(f"    Script: {len(code.splitlines())} lines")
        print(f"    pytest: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print(f"    stdout: {proc.stdout[-500:]}")
            print(f"    stderr: {proc.stderr[-500:]}")

        report("Full Pipeline (Wikipedia)", type("R", (), {
            "success": passed, "total_tokens": explore_tokens,
            "playbook": result.playbook,
        })(), result.playbook, code)


# =====================================================
# Summary
# =====================================================
def pytest_sessionfinish(session, exitstatus):
    if RESULTS:
        print(f"\n\n{'='*60}")
        print(f"  STRESS TEST SUMMARY")
        print(f"{'='*60}")
        passed = sum(1 for r in RESULTS if r["success"])
        total = len(RESULTS)
        total_tokens = sum(r["tokens"] for r in RESULTS)
        print(f"  Passed: {passed}/{total}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Estimated cost: ${total_tokens * 0.000003 + total_tokens * 0.000015:.4f}")
        print()
        for r in RESULTS:
            status = "PASS" if r["success"] else "FAIL"
            print(f"    [{status}] {r['test']} ({r['steps']} steps, {r['tokens']} tokens)")
        print(f"{'='*60}")
