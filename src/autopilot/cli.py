"""CLI entry point for playwright-autopilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from autopilot.core.browser import Browser, BrowserConfig
from autopilot.core.playbook import Playbook
from autopilot.core.runner import Runner


def main():
    parser = argparse.ArgumentParser(
        prog="autopilot",
        description="Browser automation that gets cheaper every time you run it.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # replay command
    replay_p = sub.add_parser("replay", help="Replay a saved playbook")
    replay_p.add_argument("playbook", help="Path to playbook JSON file")
    replay_p.add_argument("--vars", type=json.loads, default={}, help='Variables as JSON')
    replay_p.add_argument("--max-tier", type=int, default=4, help="Maximum escalation tier (0-4)")
    replay_p.add_argument("--headed", action="store_true", help="Show browser window")
    replay_p.add_argument("--save", action="store_true", help="Save updated playbook after heal")
    replay_p.add_argument("--visual-diff", action="store_true", help="Enable screenshot diffing")
    replay_p.add_argument("--audit", action="store_true", help="Enable audit trail logging")
    replay_p.add_argument("--profile", action="store_true", help="Enable performance profiling")

    # info command
    info_p = sub.add_parser("info", help="Show playbook details with health stats")
    info_p.add_argument("playbook", help="Path to playbook JSON file")

    # interactive command
    interactive_p = sub.add_parser("interactive", help="Interactive REPL mode")
    interactive_p.add_argument("--url", required=True, help="Starting URL")
    interactive_p.add_argument("--llm", required=True, help="LLM model")
    interactive_p.add_argument("--headed", action="store_true", default=True, help="Show browser")

    # import command
    import_p = sub.add_parser("import", help="Import Playwright recording as playbook")
    import_p.add_argument("file", help="Path to Playwright script (.ts, .js, .py)")
    import_p.add_argument("--output", "-o", help="Output playbook JSON path")
    import_p.add_argument("--name", help="Playbook name")

    # stats command
    stats_p = sub.add_parser("stats", help="Show playbook statistics and health")
    stats_p.add_argument("--dir", help="Playbook directory", default=None)

    # generate command
    gen_p = sub.add_parser("generate", help="Generate standalone Playwright script from playbook")
    gen_p.add_argument("playbook", help="Playbook JSON file")
    gen_p.add_argument("--lang", choices=["python", "typescript", "ts"], default="python", help="Output language")
    gen_p.add_argument("--output", "-o", help="Output file path")
    gen_p.add_argument("--vars", type=json.loads, default={}, help="Variables as JSON")

    # chain command
    chain_p = sub.add_parser("chain", help="Run multiple playbooks in sequence")
    chain_p.add_argument("playbooks", nargs="+", help="Playbook files to chain")
    chain_p.add_argument("--vars", type=json.loads, default={}, help="Shared variables")
    chain_p.add_argument("--headed", action="store_true", help="Show browser")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    commands = {
        "replay": lambda: asyncio.run(_replay(args)),
        "info": lambda: _info(args),
        "interactive": lambda: asyncio.run(_interactive(args)),
        "import": lambda: _import(args),
        "generate": lambda: _generate(args),
        "stats": lambda: _stats(args),
        "chain": lambda: asyncio.run(_chain(args)),
    }

    if args.command in commands:
        commands[args.command]()
    else:
        parser.print_help()


async def _replay(args):
    playbook = Playbook.load(args.playbook)
    config = BrowserConfig(headless=not args.headed)

    async with Browser(config) as browser:
        runner = Runner(
            browser=browser, max_tier=args.max_tier,
            visual_diff=args.visual_diff, audit=args.audit, profile=args.profile,
        )
        result = await runner.run(playbook, variables=args.vars)

    print(json.dumps(result.summary, indent=2))

    if args.save and result.playbook_updated:
        playbook.save(args.playbook)
        print(f"Playbook saved: {args.playbook}")

    if result.audit_path:
        print(f"Audit trail: {result.audit_path}")

    sys.exit(0 if result.success else 1)


async def _interactive(args):
    from autopilot.core.repl import run_repl
    await run_repl(url=args.url, llm_model=args.llm, headless=not args.headed)


def _import(args):
    from autopilot.importers import import_playwright

    playbook = import_playwright(args.file, name=args.name)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.file).with_suffix(".json")

    playbook.save(out_path)
    print(f"Imported {len(playbook.steps)} steps → {out_path}")


def _generate(args):
    from autopilot.codegen import generate_from_file

    code = generate_from_file(args.playbook, lang=args.lang, variables=args.vars)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(code)
        print(f"Generated: {out_path}")
    else:
        print(code)


def _stats(args):
    playbook_dir = Path(args.dir) if args.dir else Path.home() / ".autopilot" / "playbooks"
    if not playbook_dir.exists():
        print(f"No playbooks found at {playbook_dir}")
        return

    playbooks = []
    for path in sorted(playbook_dir.glob("*.json")):
        try:
            pb = Playbook.load(path)
            playbooks.append(pb)
        except Exception:
            continue

    if not playbooks:
        print("No playbooks found.")
        return

    print(f"\n  {'Name':<25} {'Steps':>5} {'Runs':>5} {'Success':>8} {'Heals':>6} {'Health'}")
    print(f"  {'─'*25} {'─'*5} {'─'*5} {'─'*8} {'─'*6} {'─'*20}")

    for pb in playbooks:
        total_heals = sum(s.heal_count for s in pb.steps)
        total_fails = sum(s.fail_count for s in pb.steps)

        # Health bar based on success rate
        if pb.run_count == 0:
            health = "░░░░░░░░░░  (no runs)"
        else:
            pct = pb.success_rate
            filled = int(pct * 10)
            health = "█" * filled + "░" * (10 - filled) + f"  ({pct:.0%})"
            if total_heals > 0:
                health += f"  healed:{total_heals}"

        print(
            f"  {pb.name:<25} {len(pb.steps):>5} {pb.run_count:>5} "
            f"{pb.success_count:>8} {total_heals:>6} {health}"
        )

    # Flaky step detection
    flaky_steps = []
    for pb in playbooks:
        for i, step in enumerate(pb.steps):
            if step.heal_count >= 3 or (step.fail_count > 0 and step.run_count > 5):
                flaky_steps.append((pb.name, i, step))

    if flaky_steps:
        print(f"\n  Flaky steps detected:")
        for pb_name, idx, step in flaky_steps:
            ratio = f"{step.heal_count}/{step.run_count}" if step.run_count > 0 else "N/A"
            print(f"    {pb_name} step {idx+1}: '{step.intent}' — healed {ratio} runs")
            print(f"      Selector: {step.selector}")
            print(f"      Consider adding data-testid or more alternatives.")


def _info(args):
    playbook = Playbook.load(args.playbook)
    print(f"Name:         {playbook.name}")
    print(f"URL:          {playbook.url}")
    print(f"Steps:        {len(playbook.steps)}")
    print(f"Variables:    {playbook.extract_variables() or 'none'}")
    print(f"Run count:    {playbook.run_count}")
    print(f"Success rate: {playbook.success_rate:.0%}")
    print(f"Created:      {playbook.created_at}")
    print(f"Updated:      {playbook.updated_at}")
    print()
    for i, step in enumerate(playbook.steps):
        flags = []
        if step.tier_resolved > 0:
            flags.append(f"T{step.tier_resolved}")
        if step.condition:
            flags.append(f"if:{step.condition}")
        if step.skip_on_fail:
            flags.append("skip_on_fail")
        if step.assert_after:
            flags.append(f"assert:'{step.assert_after[:30]}'")
        if step.heal_count > 0:
            flags.append(f"healed:{step.heal_count}x")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"  {i+1}. {step.intent} → {step.action}{flag_str}")
        print(f"     selector: {step.selector}")
        if step.selector_alternatives:
            print(f"     alternatives: {len(step.selector_alternatives)}")


async def _chain(args):
    from autopilot.agent import Agent

    async with Agent(llm=None, headless=not args.headed) as agent:
        results = await agent.run_chain(args.playbooks, variables=args.vars)

    for i, result in enumerate(results):
        status = "PASS" if result.success else "FAIL"
        print(f"  {i+1}. {args.playbooks[i]}: {status} ({result.total_tokens} tokens, {result.total_duration_ms:.0f}ms)")

    all_pass = all(r.success for r in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
