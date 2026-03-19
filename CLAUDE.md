# playwright-autopilot — Contributing with Claude Code

## Project Overview
Token-efficient browser automation with tiered AI escalation. Core concept: "pay only for the intelligence each step demands."

## Architecture
- `src/autopilot/core/runner.py` — the heart (tiered escalation engine)
- `src/autopilot/core/healer.py` — 5 zero-token healing strategies
- `src/autopilot/core/digest.py` — compact page representation (~500 tokens)
- `src/autopilot/llm/factory.py` — model string → provider routing
- `src/autopilot/mcp/server.py` — MCP server (6 tools)
- `src/autopilot/agent.py` — high-level Agent API

## Development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium
pytest tests/harness/ -v
```

## Testing
- `tests/harness/` — private test harness (not for public release)
- Uses a local HTTP server (`test_server.py`) with controlled HTML pages
- Each tier has dedicated tests: `test_tier0_replay.py` through `test_tier4_vision.py`
- Integration tests verify the full pipeline and token economics
- Benchmark tests measure actual token costs

## Rules
- Keep MCP tools at 6 or fewer — tool bloat is the problem we're solving
- Tier 0-1 must ALWAYS cost 0 tokens — this is non-negotiable
- Playbooks are JSON, human-readable, and portable
- LLM provider is always optional — the framework works without one
- Never break the `Agent()` 5-line quickstart API
