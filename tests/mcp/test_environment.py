"""MCP 环境插值、路径与 header 合并契约。"""

import json
import os
from pathlib import Path

import pytest

from iris.exceptions import IrisConfigError
from iris.mcp.config import load_mcp_config, resolve_server_config
from iris.mcp.models import MCPServerConfig


def load_server(tmp_path: Path, declaration: dict) -> MCPServerConfig:
    """从真实 parser 获取归一声明。"""
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"Docs-Local": declaration}}))
    return load_mcp_config(path, overrides={}).servers[0]


def test_environment_is_single_pass_and_case_sensitive(tmp_path: Path) -> None:
    server = load_server(
        tmp_path,
        {
            "command": "${PYTHON}",
            "args": [
                "${env:Token}",
                "${MISSING:-fallback}",
                "${EMPTY:-default}",
                "${EMPTY}",
                "${NESTED}",
            ],
        },
    )
    resolved = resolve_server_config(
        server,
        environ={
            "PYTHON": "python",
            "Token": "Mixed",
            "EMPTY": "",
            "NESTED": "${UNRESOLVED}",
        },
        workspace_root=tmp_path,
    )
    assert resolved.command == "python"
    assert resolved.args == ("Mixed", "fallback", "default", "", "${UNRESOLVED}")
    assert resolved.cwd == tmp_path


@pytest.mark.parametrize(
    "expression",
    [
        "${MISSING}",
        "${input:token}",
        "${command:token}",
        "${workspaceFolder}",
        "${userHome}",
        "${CLAUDE_PLUGIN_ROOT}",
        "${file:token}",
    ],
)
def test_missing_and_client_expressions_are_rejected(tmp_path: Path, expression: str) -> None:
    server = load_server(tmp_path, {"command": "python", "args": [expression]})
    with pytest.raises(IrisConfigError, match="args"):
        resolve_server_config(server, environ={}, workspace_root=tmp_path)


def test_stdio_environment_precedence_and_paths(tmp_path: Path) -> None:
    (tmp_path / "server.env").write_text("A=file\nB=${HOST}\nC=${NESTED}\n")
    server = load_server(
        tmp_path,
        {
            "command": "python",
            "args": ["relative.py"],
            "env_vars": ["A", "PASSED"],
            "envFile": "server.env",
            "cwd": "subdir",
            "env": {"A": "explicit", "D": "${HOST}"},
        },
    )
    before = dict(os.environ)
    resolved = resolve_server_config(
        server,
        environ={
            "A": "host",
            "PASSED": "yes",
            "HOST": "value",
            "NESTED": "${HOST}",
            "UNUSED": "no",
        },
        workspace_root=tmp_path / "workspace",
    )
    assert resolved.env == {
        "A": "explicit",
        "PASSED": "yes",
        "B": "value",
        "C": "${HOST}",
        "D": "value",
    }
    assert resolved.cwd == tmp_path / "subdir"
    assert resolved.args == ("relative.py",)
    assert dict(os.environ) == before


def test_headers_merge_after_expansion(tmp_path: Path) -> None:
    server = load_server(
        tmp_path,
        {
            "url": "https://${HOST}/mcp",
            "headers": {"Authorization": "Bearer ${TOKEN}", "X-Client": "iris"},
            "http_headers": {"authorization": "Bearer ${env:TOKEN}"},
            "env_http_headers": {"X-Tenant": "TENANT"},
            "bearer_token_env_var": "TOKEN",
        },
    )
    assert len(server.header_values) == 3
    resolved = resolve_server_config(
        server,
        environ={
            "HOST": "example.test",
            "TOKEN": "secret",
            "TENANT": "one",
        },
        workspace_root=tmp_path,
    )
    assert resolved.url == "https://example.test/mcp"
    assert resolved.headers == {
        "authorization": "Bearer secret",
        "x-client": "iris",
        "x-tenant": "one",
    }


def test_header_conflict_does_not_expose_effective_values(tmp_path: Path) -> None:
    server = load_server(
        tmp_path,
        {
            "url": "https://example.test",
            "headers": {"Authorization": "Bearer ${TOKEN}"},
            "http_headers": {"authorization": "Bearer different-secret"},
        },
    )
    with pytest.raises(IrisConfigError) as error:
        resolve_server_config(server, environ={"TOKEN": "secret-value"}, workspace_root=tmp_path)
    assert "authorization" in str(error.value)
    assert "different-secret" not in str(error.value) and "secret-value" not in str(error.value)


@pytest.mark.parametrize(
    "decl",
    [
        {"command": "python", "envFile": "missing.env"},
        {"command": "python", "env_vars": ["MISSING"]},
        {"url": "https://example.test", "bearer_token_env_var": "MISSING"},
    ],
)
def test_missing_environment_inputs_fail_at_resolution(tmp_path: Path, decl: dict) -> None:
    server = load_server(tmp_path, decl)
    with pytest.raises(IrisConfigError):
        resolve_server_config(server, environ={}, workspace_root=tmp_path)
