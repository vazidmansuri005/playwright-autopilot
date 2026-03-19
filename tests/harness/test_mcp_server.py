"""Tests for MCP server tool definitions and routing."""

import pytest


class TestMCPServerSetup:
    """Test MCP server can be created and tools are well-formed."""

    def test_server_creates_without_error(self):
        """MCP server creation doesn't fail."""
        try:
            from autopilot.mcp.server import create_server
            server = create_server(llm_model=None, headless=True)
            assert server is not None
        except ImportError:
            pytest.skip("MCP SDK not installed")

    def test_claude_api_tool_definitions(self):
        """Claude API tool definitions are valid."""
        from autopilot.integrations.claude_api import TOOL_DEFINITIONS

        assert len(TOOL_DEFINITIONS) == 4

        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "autopilot_run" in names
        assert "autopilot_step" in names
        assert "autopilot_navigate" in names
        assert "autopilot_snapshot" in names

        # Each tool has required fields
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_definitions_token_estimate(self):
        """Tool definitions should be compact (< 2000 tokens estimated)."""
        import json
        from autopilot.integrations.claude_api import TOOL_DEFINITIONS

        text = json.dumps(TOOL_DEFINITIONS)
        # Rough estimate: 1.3 tokens per word
        token_estimate = int(len(text.split()) * 1.3)

        # 4 tools should be well under 2000 tokens (vs Playwright MCP's 3600+)
        assert token_estimate < 2000, f"Tool definitions too large: ~{token_estimate} tokens"
