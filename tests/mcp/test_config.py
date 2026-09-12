"""MCP 文件导入与声明归一化契约。"""

import json
from pathlib import Path

import pytest

from iris.agents.config.mcp import AgentMCPConfig, MCPServerOverride
from iris.exceptions import IrisConfigError
from iris.mcp.config import load_mcp_config


def write_config(tmp_path: Path, servers: dict, root: str = "mcpServers") -> Path:
    """写入合成配置，避免依赖真实客户端文件。"""
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({root: servers, "application": {"theme": "dark"}}))
    return path


@pytest.mark.parametrize("root", ["mcpServers", "servers", "mcp_servers"])
def test_import_common_roots_preserves_identity(tmp_path: Path, root: str) -> None:
    path = write_config(tmp_path, {"Docs-Local": {"command": "python", "args": ["s.py"]}}, root)
    server = load_mcp_config(path, overrides={}).servers[0]
    assert (server.server_id, server.transport, server.args) == ("Docs-Local", "stdio", ("s.py",))
    assert server.required and not server.trust_annotations
    assert server.startup_timeout_sec == server.tool_timeout_sec == 30


@pytest.mark.parametrize(
    ("suffix", "source"),
    [
        (".jsonc", '{// comment\n"servers":{"Docs-Local":{"command":"python",},},}'),
        (".toml", '[mcp_servers.Docs-Local]\ncommand = "python"\n'),
    ],
)
def test_jsonc_and_toml(tmp_path: Path, suffix: str, source: str) -> None:
    path = tmp_path / f"mcp{suffix}"
    path.write_text(source)
    assert load_mcp_config(path, overrides={}).servers[0].command == "python"


@pytest.mark.parametrize("source", ['{"servers": {}, "mcpServers": {}}', '{"servers":', "{}"])
def test_invalid_document_is_not_an_optional_failure(tmp_path: Path, source: str) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(source)
    with pytest.raises(IrisConfigError):
        load_mcp_config(path, overrides={})


def test_empty_catalog_and_missing_override(tmp_path: Path) -> None:
    path = write_config(tmp_path, {})
    assert load_mcp_config(path, overrides={}).servers == ()
    with pytest.raises(IrisConfigError, match="missing"):
        load_mcp_config(path, overrides={"missing": MCPServerOverride(required=False)})


@pytest.mark.parametrize(
    ("decl", "transport"),
    [
        ({"command": "python"}, "stdio"),
        ({"command": "python", "type": "stdio"}, "stdio"),
        ({"url": "https://example.test/sse"}, "streamable-http"),
        *[
            ({"serverUrl": "https://example.test/mcp", "type": value}, "streamable-http")
            for value in ("http", "streamable-http", "streamable_http")
        ],
        ({"url": "https://example.test/events", "type": "sse"}, "sse"),
    ],
)
def test_transport_normalization(tmp_path: Path, decl: dict, transport: str) -> None:
    server = load_mcp_config(write_config(tmp_path, {"s": decl}), overrides={}).servers[0]
    assert server.transport == transport


@pytest.mark.parametrize(
    "decl",
    [
        {"command": "python", "url": "https://example.test/mcp"},
        {"url": "https://a.test", "serverUrl": "https://b.test"},
        {"command": "python", "startup_timeout_sec": 2, "startup_timeout_ms": 3000},
        {"command": "python", "headers": {"x": "y"}},
        {"url": "https://example.test", "env": {"A": "B"}},
        {"command": "python", "tool_timeout_sec": 0},
        {"command": "python", "tool_timeout_sec": "30"},
    ],
)
def test_conflicting_or_invalid_declarations(tmp_path: Path, decl: dict) -> None:
    with pytest.raises(IrisConfigError):
        load_mcp_config(write_config(tmp_path, {"s": decl}), overrides={})


def test_aliases_overrides_and_empty_allowlist(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "s": {
                "url": "https://example.test",
                "serverUrl": "https://example.test",
                "startup_timeout_ms": 2000,
                "startup_timeout_sec": 2,
                "required": False,
                "enabled_tools": [],
                "disabled_tools": ["write"],
            }
        },
    )
    server = load_mcp_config(
        path,
        overrides={
            "s": MCPServerOverride(
                required=True,
                trust_annotations=True,
                startup_timeout_sec=5,
                tool_timeout_sec=7,
            )
        },
    ).servers[0]
    assert server.required and server.trust_annotations
    assert (server.startup_timeout_sec, server.tool_timeout_sec) == (5, 7)
    assert server.enabled_tools == () and server.disabled_tools == ("write",)


def test_disabled_skips_all_unused_fields(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "off": {
                "enabled": False,
                "disabled": True,
                "required": "invalid",
                "inputs": {},
                "startup_timeout_ms": "invalid",
                "envFile": "missing.env",
                "env": {"A": "${MISSING}"},
            }
        },
    )
    assert load_mcp_config(path, overrides={}).servers == ()
    with pytest.raises(IrisConfigError, match="enabled|disabled"):
        load_mcp_config(
            write_config(tmp_path, {"off": {"enabled": False, "disabled": False}}), overrides={}
        )


@pytest.mark.parametrize("field", ["inputs", "http_headers_helper", "approval_policy", "remote"])
def test_unsupported_fields_follow_required_policy(tmp_path: Path, field: str) -> None:
    decl = {"command": "python", field: "CLIENT_ONLY"}
    with pytest.raises(IrisConfigError, match=field):
        load_mcp_config(write_config(tmp_path, {"s": decl}), overrides={})
    config = load_mcp_config(
        write_config(tmp_path, {"s": decl}),
        overrides={
            "s": MCPServerOverride(required=False),
        },
    )
    assert config.servers == ()
    assert config.diagnostics[0].server_id == "s"
    assert config.diagnostics[0].field == field
    assert "CLIENT_ONLY" not in config.diagnostics[0].message


def test_optional_invalid_type_is_diagnostic(tmp_path: Path) -> None:
    config = load_mcp_config(
        write_config(
            tmp_path,
            {
                "s": {
                    "command": "python",
                    "required": False,
                    "env_vars": [{"source": "remote"}],
                }
            },
        ),
        overrides={},
    )
    assert config.servers == ()
    assert config.diagnostics[0].field == "env_vars"


def test_yaml_reference_resolves_relative_to_agent(tmp_path: Path) -> None:
    reference = AgentMCPConfig.model_validate(
        {"path": "mcp.json"},
        context={
            "config_path": tmp_path / "agent.yaml",
        },
    )
    assert reference.path == tmp_path / "mcp.json"
