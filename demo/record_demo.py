"""Record a LinkedIn demo — complex e-commerce flow on SauceDemo.

Shows: Login → Browse → Add to Cart → Cart → Checkout → Fill Form

SAFE: Reads API key from environment (set by shell script), never prints it.

Run via: bash demo/record_full_demo.sh
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.explorer import Explorer
from autopilot.core.runner import Runner
from autopilot.codegen import generate_python
from autopilot.llm.claude import ClaudeLLM

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(text):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def step(num, text):
    print(f"  {BOLD}{GREEN}[{num}]{RESET} {text}")


def info(text):
    print(f"      {text}")


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ERROR: ANTHROPIC_API_KEY not set. Use: bash demo/record_full_demo.sh")
        sys.exit(1)

    llm = ClaudeLLM(api_key=api_key, max_tokens=512)

    banner("playwright-autopilot")
    print("  AI writes your Playwright tests. You run them for free.\n")
    await asyncio.sleep(2)

    # --- Setup browser with video recording ---
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser_instance = await pw.chromium.launch(headless=False, slow_mo=400)
    context = await browser_instance.new_context(
        record_video_dir=str(OUTPUT_DIR),
        record_video_size={"width": 1280, "height": 720},
        viewport={"width": 1280, "height": 720},
    )
    page = await context.new_page()

    browser = Browser(BrowserConfig(headless=False))
    browser._playwright = pw
    browser._browser = browser_instance
    browser._context = context
    browser._page = page

    # ===========================================
    # PHASE 1: AI EXPLORES A COMPLEX FLOW
    # ===========================================
    banner("PHASE 1 — AI Explores (one-time cost)")

    goal = (
        "Log in with 'standard_user' / 'secret_sauce', "
        "add the first product to cart, "
        "click the cart icon in the top right, "
        "click the Checkout button, "
        "fill the checkout form with first name 'John', last name 'Doe', zip '10001', "
        "then click Continue"
    )

    step(1, "Goal: Full e-commerce checkout on saucedemo.com")
    info(f"  \"{goal[:80]}...\"")
    print()
    step(2, "AI is exploring...")
    print()

    explorer = Explorer(browser=browser, llm=llm, max_steps=15)
    t0 = time.monotonic()

    result = await explorer.explore(
        url="https://www.saucedemo.com",
        goal=goal,
        variables={
            "username": "standard_user",
            "password": "secret_sauce",
            "firstName": "John",
            "lastName": "Doe",
            "zip": "10001",
        },
    )

    explore_time = time.monotonic() - t0
    tokens = explorer.usage.total_tokens

    step(3, f"Done in {explore_time:.1f}s — {len(result.playbook.steps)} steps recorded")
    info(f"Tokens: {tokens:,}")
    info(f"Cost:   ~${tokens * 0.000018:.4f}")
    print()

    for i, s in enumerate(result.playbook.steps):
        info(f"  {i+1}. {s.intent}")

    await asyncio.sleep(3)

    # ===========================================
    # PHASE 2: GENERATE REAL PLAYWRIGHT SCRIPT
    # ===========================================
    banner("PHASE 2 — Generate Playwright Script (free)")

    if result.playbook.steps:
        code = generate_python(result.playbook, variables={
            "username": "standard_user", "password": "secret_sauce",
            "firstName": "John", "lastName": "Doe", "zip": "10001",
        })
        script_path = OUTPUT_DIR / "test_generated.py"
        script_path.write_text(code)

        step(4, f"Generated: {script_path.name} ({len(code.splitlines())} lines)")
        print()
        info("  Key lines from generated script:")
        info("  ─────────────────────────────────")
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith(("page.", "def test_", "VARIABLES")):
                info(f"  {stripped[:75]}")
        info("  ─────────────────────────────────")
        print()
        info(f"  {BOLD}Standard Playwright. No autopilot dependency. Runs with pytest.{RESET}")
    else:
        info("  No steps recorded.")
        code = ""

    await asyncio.sleep(3)

    # ===========================================
    # PHASE 3: REPLAY AT ZERO COST
    # ===========================================
    if result.playbook.steps:
        banner("PHASE 3 — Replay (zero tokens, no AI)")

        await page.goto("about:blank")
        await asyncio.sleep(1)

        step(5, "Replaying with NO LLM — pure Playwright selectors...")
        print()

        runner = Runner(browser=browser, llm=None, max_tier=0)
        t1 = time.monotonic()
        replay = await runner.run(result.playbook, variables={
            "username": "standard_user", "password": "secret_sauce",
            "firstName": "John", "lastName": "Doe", "zip": "10001",
        })
        replay_time = time.monotonic() - t1

        passed = sum(1 for s in replay.steps if s.success)
        total = len(replay.steps)

        step(6, f"Replay: {passed}/{total} steps passed in {replay_time:.1f}s")
        info(f"Tokens: {replay.total_tokens} (zero)")
        print()

        for s in replay.steps:
            status = f"{GREEN}PASS{RESET}" if s.success else f"{YELLOW}FAIL{RESET}"
            info(f"  [{status}] {s.intent} — Tier {s.tier}")

    await asyncio.sleep(3)

    # ===========================================
    # SUMMARY
    # ===========================================
    banner("RESULT")
    info(f"  Explore (AI, once):    {tokens:,} tokens   (~${tokens * 0.000018:.4f})")
    info(f"  Generate script:       0 tokens        ($0.00)")
    info(f"  Replay (every run):    0 tokens        ($0.00)")
    info(f"  Run 1000x:             0 tokens        ($0.00)")
    print()
    info(f"  Playwright MCP 1000x:  ~114,000,000 tokens (~$150)")
    print()
    info(f"  {BOLD}{GREEN}AI writes your Playwright tests. You run them for free.{RESET}")
    info(f"  {BOLD}github.com/vazidmansuri005/playwright-autopilot{RESET}")

    await asyncio.sleep(5)

    # Save video
    video_path = await page.video.path()
    await context.close()
    await browser_instance.close()
    await pw.stop()

    final_video = OUTPUT_DIR / "demo_browser.webm"
    if video_path and Path(video_path).exists():
        Path(video_path).rename(final_video)
        info(f"\n  Browser video: {final_video}")


if __name__ == "__main__":
    asyncio.run(main())
