"""MCP 文件引用通过 AgentConfig 的既有路径上下文解析。"""

from pathlib import Path

from iris.agents import AgentMCPConfig, MCPServerOverride, load_agent_config


def test_yaml_resolves_mcp_reference_without_loading_server_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: a\nmodel: openai/test\nsystem: test\nmcp:\n  path: config/mcp.json\n"
        "  overrides:\n    Local-服务:\n      required: false\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert isinstance(config.mcp, AgentMCPConfig)
    assert config.mcp.path == tmp_path / "config" / "mcp.json"
    assert config.mcp.overrides["Local-服务"] == MCPServerOverride(required=False)
