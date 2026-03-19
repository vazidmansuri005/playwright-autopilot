"""Benchmark tests: Measure token costs and timing across tiers.

These are NOT unit tests — they're measurement tools for us to validate
the token economics we're promising. Run with:
    pytest tests/harness/test_benchmark.py -v --tb=short -s
"""

import json
import time
from pathlib import Path

import pytest

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.digest import extract_digest
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.runner import Runner
from autopilot.core.snapshot import extract_snapshot
from autopilot.core.vision import capture_for_vision
from autopilot.llm.base import MockLLM


BENCHMARK_OUTPUT = Path(__file__).parent.parent / "fixtures" / "benchmark_results.json"


@pytest.mark.harness
class TestBenchmark:
    """Measure real token costs and timing."""

    async def test_measure_digest_tokens(self, browser, test_server):
        """Measure actual token estimates for digest across pages."""
        results = {}

        for page_name, path in [
            ("login", "/login"),
            ("dashboard", "/dashboard"),
            ("settings", "/settings"),
        ]:
            await browser.page.goto(f"{test_server.url}{path}")
            digest = await extract_digest(browser.page)
            results[page_name] = {
                "elements": len(digest.elements),
                "token_estimate": digest.token_estimate,
                "prompt_length": len(digest.to_prompt()),
            }

        print("\n=== Digest Token Estimates ===")
        for name, data in results.items():
            print(f"  {name}: {data['elements']} elements, ~{data['token_estimate']} tokens")

        # All should be under 1000 tokens
        for name, data in results.items():
            assert data["token_estimate"] < 1000, f"{name} digest too large: {data['token_estimate']}"

    async def test_measure_snapshot_tokens(self, browser, test_server):
        """Measure actual token estimates for accessibility snapshots."""
        results = {}

        for page_name, path in [
            ("login", "/login"),
            ("dashboard", "/dashboard"),
            ("settings", "/settings"),
        ]:
            await browser.page.goto(f"{test_server.url}{path}")
            full = await extract_snapshot(browser.page, interactive_only=False)
            interactive = await extract_snapshot(browser.page, interactive_only=True)
            results[page_name] = {
                "full_tokens": full.token_estimate,
                "interactive_tokens": interactive.token_estimate,
                "reduction_pct": round(
                    (1 - interactive.token_estimate / max(full.token_estimate, 1)) * 100, 1
                ),
            }

        print("\n=== Snapshot Token Estimates ===")
        for name, data in results.items():
            print(
                f"  {name}: full={data['full_tokens']}, "
                f"interactive={data['interactive_tokens']} "
                f"({data['reduction_pct']}% reduction)"
            )

    async def test_measure_vision_size(self, browser, test_server):
        """Measure screenshot sizes for vision tier."""
        results = {}

        for page_name, path in [
            ("login", "/login"),
            ("dashboard", "/dashboard"),
        ]:
            await browser.page.goto(f"{test_server.url}{path}")
            capture = await capture_for_vision(browser.page)
            img_size_kb = len(capture.screenshot_b64) * 3 / 4 / 1024  # base64 → bytes → KB
            results[page_name] = {
                "image_size_kb": round(img_size_kb, 1),
                "token_estimate": capture.token_estimate,
                "has_accessibility": bool(capture.accessibility_summary),
            }

        print("\n=== Vision Capture Sizes ===")
        for name, data in results.items():
            print(f"  {name}: {data['image_size_kb']}KB, ~{data['token_estimate']} tokens")

    async def test_measure_tier0_latency(self, browser, test_server, login_playbook):
        """Measure Tier 0 replay latency (baseline)."""
        runner = Runner(browser=browser, max_tier=0)

        # Warmup
        await runner.run(login_playbook, variables={"email": "a@b.com", "password": "p"})
        await browser.page.goto("about:blank")

        # Measure
        start = time.monotonic()
        result = await runner.run(login_playbook, variables={"email": "a@b.com", "password": "p"})
        elapsed_ms = (time.monotonic() - start) * 1000

        print(f"\n=== Tier 0 Replay Latency ===")
        print(f"  Total: {elapsed_ms:.0f}ms for {len(result.steps)} steps")
        print(f"  Per step: {elapsed_ms / max(len(result.steps), 1):.0f}ms")

        assert result.success
        assert elapsed_ms < 10_000  # Should complete in under 10s

    async def test_measure_tier1_latency(self, browser, test_server, login_playbook_mutated):
        """Measure Tier 1 healing latency."""
        test_server.set_mutated(True)
        runner = Runner(browser=browser, max_tier=1)

        start = time.monotonic()
        result = await runner.run(
            login_playbook_mutated,
            variables={"email": "a@b.com", "password": "p"},
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        healed = sum(1 for s in result.steps if s.tier == 1)
        print(f"\n=== Tier 1 Heal Latency ===")
        print(f"  Total: {elapsed_ms:.0f}ms, {healed}/{len(result.steps)} steps healed")
        for s in result.steps:
            print(f"  Step '{s.intent}': Tier {s.tier}, {s.duration_ms:.0f}ms, strategy={s.strategy}")

    async def test_token_comparison_table(self, browser, test_server):
        """Generate the comparison table: what would each approach cost?"""
        pages = [
            ("login", "/login"),
            ("dashboard", "/dashboard"),
            ("settings", "/settings"),
        ]

        print("\n=== Token Cost Comparison (per page) ===")
        print(f"{'Page':<12} {'Digest(T2)':<12} {'Snapshot(T3)':<14} {'MCP estimate':<14}")
        print("-" * 52)

        for name, path in pages:
            await browser.page.goto(f"{test_server.url}{path}")
            digest = await extract_digest(browser.page)
            snapshot = await extract_snapshot(browser.page, interactive_only=True)
            # MCP estimate: full a11y tree is roughly 5-10x the interactive snapshot
            mcp_estimate = snapshot.token_estimate * 7

            print(
                f"{name:<12} {digest.token_estimate:<12} "
                f"{snapshot.token_estimate:<14} ~{mcp_estimate:<13}"
            )
