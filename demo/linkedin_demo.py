"""LinkedIn Demo — records ONLY the browser, clean and professional.

Records two browser videos:
  1. AI exploring SauceDemo (login → add to cart)
  2. Generated script replaying independently

Run from a FRESH clone:
    git clone https://github.com/vazidmansuri005/playwright-autopilot.git /tmp/autopilot-demo
    cd /tmp/autopilot-demo
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[claude]" && playwright install chromium
    echo "your-key" > ~/.autopilot_key
    python demo/linkedin_demo.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# Read key from file — never shown
API_KEY = ""
if Path("~/.autopilot_key").expanduser().exists():
    API_KEY = Path("~/.autopilot_key").expanduser().read_text().strip()
elif os.environ.get("ANTHROPIC_API_KEY"):
    API_KEY = os.environ["ANTHROPIC_API_KEY"]

if not API_KEY:
    print("No API key. Run: echo 'your-key' > ~/.autopilot_key")
    sys.exit(1)


async def main():
    from playwright.async_api import async_playwright
    from autopilot.core.explorer import Explorer
    from autopilot.core.runner import Runner
    from autopilot.core.browser import Browser, BrowserConfig
    from autopilot.codegen import generate_python
    from autopilot.llm.claude import ClaudeLLM

    llm = ClaudeLLM(api_key=API_KEY, max_tokens=512)
    pw = await async_playwright().start()

    # ==========================================
    # VIDEO 1: AI Explores SauceDemo
    # ==========================================
    print("\n  Recording Video 1: AI Explores SauceDemo...")

    browser1 = await pw.chromium.launch(headless=False, slow_mo=500)
    ctx1 = await browser1.new_context(
        record_video_dir=str(OUTPUT),
        record_video_size={"width": 1280, "height": 720},
        viewport={"width": 1280, "height": 720},
    )
    page1 = await ctx1.new_page()

    # Wrap for autopilot
    b = Browser(BrowserConfig())
    b._playwright = pw
    b._browser = browser1
    b._context = ctx1
    b._page = page1

    explorer = Explorer(browser=b, llm=llm, max_steps=8)
    result = await explorer.explore(
        url="https://www.saucedemo.com",
        goal="Log in with 'standard_user' / 'secret_sauce', then add the first product (Sauce Labs Backpack) to the cart",
        variables={"username": "standard_user", "password": "secret_sauce"},
    )

    explore_tokens = explorer.usage.total_tokens
    print(f"  Steps: {len(result.playbook.steps)}, Tokens: {explore_tokens}")
    for i, s in enumerate(result.playbook.steps):
        print(f"    {i+1}. {s.intent}")

    # Pause on final state
    await asyncio.sleep(3)

    # Get video path before closing
    video1_raw = await page1.video.path()
    await ctx1.close()
    await browser1.close()

    # Rename video
    video1 = OUTPUT / "1_ai_explores.webm"
    if video1_raw and Path(video1_raw).exists():
        Path(video1_raw).rename(video1)
        print(f"  Saved: {video1}")

    # Generate script
    if result.playbook.steps:
        code = generate_python(result.playbook, variables={
            "username": "standard_user", "password": "secret_sauce",
        })
        script = OUTPUT / "test_saucedemo.py"
        script.write_text(code)
        print(f"  Generated: {script} ({len(code.splitlines())} lines)")

    await asyncio.sleep(2)

    # ==========================================
    # VIDEO 2: Replay at Zero Tokens
    # ==========================================
    if result.playbook.steps:
        print("\n  Recording Video 2: Replay (Zero Tokens)...")

        browser2 = await pw.chromium.launch(headless=False, slow_mo=300)
        ctx2 = await browser2.new_context(
            record_video_dir=str(OUTPUT),
            record_video_size={"width": 1280, "height": 720},
            viewport={"width": 1280, "height": 720},
        )
        page2 = await ctx2.new_page()

        b2 = Browser(BrowserConfig())
        b2._playwright = pw
        b2._browser = browser2
        b2._context = ctx2
        b2._page = page2

        runner = Runner(browser=b2, llm=None, max_tier=0)
        replay = await runner.run(result.playbook, variables={
            "username": "standard_user", "password": "secret_sauce",
        })

        passed = sum(1 for s in replay.steps if s.success)
        print(f"  Replay: {passed}/{len(replay.steps)} passed, {replay.total_tokens} tokens")

        await asyncio.sleep(3)

        video2_raw = await page2.video.path()
        await ctx2.close()
        await browser2.close()

        video2 = OUTPUT / "2_replay_free.webm"
        if video2_raw and Path(video2_raw).exists():
            Path(video2_raw).rename(video2)
            print(f"  Saved: {video2}")

    await pw.stop()

    # ==========================================
    # Combine videos into one MP4
    # ==========================================
    print("\n  Combining videos...")

    import subprocess

    combined = OUTPUT / "linkedin_demo.mp4"

    # Check if both videos exist
    v1_exists = (OUTPUT / "1_ai_explores.webm").exists()
    v2_exists = (OUTPUT / "2_replay_free.webm").exists()

    if v1_exists and v2_exists:
        # Create concat file
        concat_file = OUTPUT / "concat.txt"
        concat_file.write_text(
            f"file '1_ai_explores.webm'\nfile '2_replay_free.webm'\n"
        )

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1280:720",
            str(combined),
        ], capture_output=True)

        concat_file.unlink()

        if combined.exists():
            size = combined.stat().st_size / 1024 / 1024
            print(f"  Combined video: {combined} ({size:.1f} MB)")
        else:
            print("  Combine failed — use individual videos")
    elif v1_exists:
        # Just convert v1
        subprocess.run([
            "ffmpeg", "-y", "-i", str(OUTPUT / "1_ai_explores.webm"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", str(combined),
        ], capture_output=True)
        print(f"  Single video: {combined}")

    # ==========================================
    # Summary
    # ==========================================
    print(f"\n  ============================")
    print(f"  Explore: {explore_tokens} tokens (~${explore_tokens * 0.000018:.4f})")
    print(f"  Replay:  0 tokens ($0.00)")
    print(f"  ============================")
    print(f"\n  Files in {OUTPUT}/:")
    for f in sorted(OUTPUT.iterdir()):
        if not f.name.startswith("."):
            print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    print(f"\n  Upload {combined.name} to LinkedIn!")


if __name__ == "__main__":
    asyncio.run(main())
