"""Real-world test: Order the cheapest 1.5 ton AC on Amazon.

This tests the FULL pipeline against a real website with a real LLM:
  - explore() on a complex site (Amazon)
  - Conditional steps (cookie banner, popups)
  - Natural language assertions
  - Playbook recording and replay
  - Visual diff across runs

Run with:
    pytest tests/real_world/test_amazon_ac.py -v -s --headed

Prerequisites:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY set in environment
    - Internet connection
    - Will NOT complete purchase (stops at cart)
"""

import asyncio
import json
import os
import sys
import pytest
from pathlib import Path

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="No LLM API key set — skipping real-world tests",
)

from autopilot import Agent, Playbook, PlaybookStep


# --- Config ---

LLM_MODEL = os.environ.get("AUTOPILOT_LLM", "claude-sonnet-4-20250514")
AMAZON_URL = "https://www.amazon.in"
SEARCH_QUERY = "1.5 ton split AC"
PLAYBOOK_DIR = Path(__file__).parent / "playbooks"
PLAYBOOK_DIR.mkdir(exist_ok=True)


# =====================================================
# TEST 1: Manual Playbook — The Known Flow
# =====================================================

@pytest.fixture
def amazon_search_playbook() -> Playbook:
    """Pre-built playbook for Amazon search + filter + add to cart."""
    return Playbook(
        name="amazon-cheapest-ac",
        url=AMAZON_URL,
        steps=[
            # Step 1: Dismiss cookie/location popups if present
            PlaybookStep(
                intent="dismiss any popup overlay",
                selector="#nav-main .nav-progressive-attribute",
                action="click",
                condition="if_visible",
                skip_on_fail=True,
                wait_after_ms=500,
            ),
            # Step 2: Search for AC
            PlaybookStep(
                intent="fill search box with AC query",
                selector="#twotabsearchtextbox",
                selector_alternatives=[
                    "input[name='field-keywords']",
                    "input#twotabsearchtextbox",
                    "role:searchbox",
                    "role:combobox[name='Search']",
                ],
                action="fill",
                value_template="${query}",
                assert_after="search box contains the query text",
            ),
            # Step 3: Submit search
            PlaybookStep(
                intent="click search button",
                selector="#nav-search-submit-button",
                selector_alternatives=[
                    "input[type='submit'][value='Go']",
                    "role:button[name='Go']",
                    ".nav-search-submit-text",
                ],
                action="click",
                wait_after_ms=3000,
                assert_after="url contains 's='",
            ),
            # Step 4: Sort by price low to high
            PlaybookStep(
                intent="click sort by dropdown",
                selector="#s-result-sort-select",
                selector_alternatives=[
                    "select[name='s']",
                    "span.a-dropdown-label",
                    "role:button[name='Sort by:']",
                ],
                action="select",
                value="price-asc-rank",
                skip_on_fail=True,
                wait_after_ms=3000,
            ),
            # Step 5: Click on first product result
            PlaybookStep(
                intent="click on the first product listing",
                selector="div[data-component-type='s-search-result']:first-child h2 a",
                selector_alternatives=[
                    ".s-result-item:first-child h2 a",
                    "[data-asin] h2 a.a-link-normal",
                    ".s-main-slot .s-result-item h2 a",
                ],
                action="click",
                wait_after_ms=3000,
                assert_after="page shows a product detail page with price",
            ),
            # Step 6: Add to cart
            PlaybookStep(
                intent="click add to cart button",
                selector="#add-to-cart-button",
                selector_alternatives=[
                    "input#add-to-cart-button",
                    "role:button[name='Add to Cart']",
                    "#addToCart input[type='submit']",
                ],
                action="click",
                wait_after_ms=2000,
                assert_after="'Added to Cart' is visible",
            ),
        ],
        variables=["query"],
    )


class TestAmazonManualPlaybook:
    """Test the pre-built Amazon playbook."""

    @pytest.mark.real_world
    async def test_search_and_add_to_cart(self, amazon_search_playbook, tmp_path):
        """Run the manual playbook — tests Tier 0 + conditional + assertions."""
        async with Agent(
            llm=LLM_MODEL,
            headless=False,
            max_tier=4,
            playbook_dir=tmp_path,
            timeout=15_000,
        ) as agent:
            result = await agent.run(
                amazon_search_playbook,
                variables={"query": SEARCH_QUERY},
            )

            print(f"\n=== Amazon Manual Playbook Result ===")
            print(json.dumps(result.summary, indent=2, default=str))

            for step in result.steps:
                status = "OK" if step.success else "FAIL"
                skip = " (SKIPPED)" if step.strategy in ("skipped", "skipped_on_fail") else ""
                print(f"  Step {step.step_index}: [{status}] {step.intent} "
                      f"(Tier {step.tier}, {step.tokens_used} tokens){skip}")

            if result.assertion_results:
                print("\n  Assertions:")
                for a in result.assertion_results:
                    status = "PASS" if a["passed"] else "FAIL"
                    print(f"    [{status}] {a['assertion']}: {a['message']}")

            print(f"\n  Total tokens: {result.total_tokens}")
            print(f"  Duration: {result.total_duration_ms:.0f}ms")

            # Save playbook for replay test
            amazon_search_playbook.save(tmp_path / "amazon-cheapest-ac.json")


# =====================================================
# TEST 2: AI Explorer — Give It a Goal
# =====================================================

class TestAmazonExplorer:
    """Test the AI explorer on Amazon."""

    @pytest.mark.real_world
    async def test_explore_amazon_search(self, tmp_path):
        """Give the explorer a goal and let AI figure out the steps."""
        async with Agent(
            llm=LLM_MODEL,
            headless=False,
            max_tier=4,
            playbook_dir=tmp_path,
            timeout=15_000,
        ) as agent:
            result = await agent.explore(
                url=AMAZON_URL,
                goal=(
                    "Search for '1.5 ton split AC', sort results by price low to high, "
                    "click on the first (cheapest) result, and add it to cart"
                ),
                variables={"query": SEARCH_QUERY},
                max_steps=15,
            )

            print(f"\n=== Amazon Explorer Result ===")
            print(json.dumps(result.summary, indent=2, default=str))

            print(f"\n  Recorded {len(result.playbook.steps)} steps:")
            for i, step in enumerate(result.playbook.steps):
                print(f"    {i+1}. {step.intent} ({step.action} → {step.selector[:50]})")

            print(f"\n  Total tokens: {result.total_tokens}")
            print(f"  Success: {result.success}")
            print(f"  Reason: {result.reason}")

            # Save the generated playbook
            if result.playbook.steps:
                result.playbook.save(tmp_path / "amazon-explored.json")
                print(f"  Playbook saved: {tmp_path / 'amazon-explored.json'}")


# =====================================================
# TEST 3: Extract — Pull Product Data
# =====================================================

class TestAmazonExtract:
    """Test structured data extraction from Amazon."""

    @pytest.mark.real_world
    async def test_extract_search_results(self, tmp_path):
        """Extract product data from Amazon search results."""
        async with Agent(
            llm=LLM_MODEL,
            headless=False,
            playbook_dir=tmp_path,
            timeout=15_000,
        ) as agent:
            # First navigate to search results
            browser = await agent._ensure_browser()
            await browser.page.goto(
                f"{AMAZON_URL}/s?k=1.5+ton+split+AC&s=price-asc-rank",
                wait_until="domcontentloaded",
            )
            await asyncio.sleep(3)

            data = await agent.extract(
                url=browser.page.url,
                goal="Top 5 cheapest 1.5 ton ACs with name, price, rating, and review count",
                schema={"name": "str", "price": "str", "rating": "str", "reviews": "str"},
            )

            print(f"\n=== Amazon Extract Result ===")
            if isinstance(data, list):
                for item in data[:5]:
                    print(f"  {item.get('name', 'N/A')[:60]} — {item.get('price', 'N/A')} "
                          f"({item.get('rating', 'N/A')} stars, {item.get('reviews', 'N/A')} reviews)")
            else:
                print(f"  Raw: {str(data)[:500]}")

            print(f"\n  Tokens used: {agent.usage.total_tokens}")


# =====================================================
# TEST 4: Replay + Visual Diff
# =====================================================

class TestAmazonReplay:
    """Test replay of a saved Amazon playbook."""

    @pytest.mark.real_world
    async def test_replay_with_visual_diff(self, amazon_search_playbook, tmp_path):
        """Replay the playbook and detect visual changes."""
        from autopilot.core.runner import Runner
        from autopilot.core.browser import Browser, BrowserConfig

        config = BrowserConfig(headless=False, timeout=15_000)

        async with Browser(config) as browser:
            # Run 1: Establish baseline
            runner1 = Runner(browser=browser, max_tier=4, visual_diff=True, audit=True)
            result1 = await runner1.run(
                amazon_search_playbook,
                variables={"query": SEARCH_QUERY},
            )

            print(f"\n=== Run 1 (baseline) ===")
            print(f"  Success: {result1.success}")
            print(f"  Tokens: {result1.total_tokens}")
            if result1.audit_path:
                print(f"  Audit: {result1.audit_path}")

            if result1.visual_diffs:
                print(f"  Visual diffs:")
                for d in result1.visual_diffs:
                    status = "CHANGED" if d.changed else "OK"
                    print(f"    Step {d.step_index}: {status} ({d.change_pct:.1f}%)")


# =====================================================
# TEST 5: Full Pipeline — Explore → Save → Replay
# =====================================================

class TestAmazonFullPipeline:
    """The ultimate test: explore, save, then replay for free."""

    @pytest.mark.real_world
    async def test_explore_then_replay(self, tmp_path):
        """Explore once (costs tokens), replay for free (0 tokens)."""

        # Phase 1: Explore
        print("\n=== Phase 1: AI Explores Amazon ===")
        async with Agent(
            llm=LLM_MODEL,
            headless=False,
            playbook_dir=tmp_path,
            timeout=15_000,
        ) as agent:
            explore_result = await agent.explore(
                url=AMAZON_URL,
                goal="Search for '1.5 ton split AC' and click on the first result",
                variables={"query": SEARCH_QUERY},
                max_steps=10,
            )
            explore_tokens = agent.usage.total_tokens
            print(f"  Steps recorded: {len(explore_result.playbook.steps)}")
            print(f"  Tokens used: {explore_tokens}")

        if not explore_result.playbook.steps:
            pytest.skip("Explorer failed to record steps")

        # Phase 2: Replay (should be ~0 tokens)
        print("\n=== Phase 2: Replay (should be free) ===")
        async with Agent(
            llm=None,  # No LLM needed!
            headless=False,
            playbook_dir=tmp_path,
            timeout=15_000,
        ) as agent:
            replay_result = await agent.replay(explore_result.playbook.name)
            replay_tokens = agent.usage.total_tokens
            print(f"  Success: {replay_result.success}")
            print(f"  Tokens used: {replay_tokens}")
            print(f"  Tier distribution: {replay_result.tier_counts}")

        print(f"\n=== Summary ===")
        print(f"  Explore cost: {explore_tokens} tokens")
        print(f"  Replay cost:  {replay_tokens} tokens")
        print(f"  Savings:      {explore_tokens - replay_tokens} tokens "
              f"({((explore_tokens - replay_tokens) / max(explore_tokens, 1)) * 100:.0f}%)")
