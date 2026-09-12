"""导入通用 MCP 文件，并在准备边界解析环境与相对路径。"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

import json5
from dotenv import dotenv_values
from pydantic import StrictBool, TypeAdapter, ValidationError

from ..agents.config.mcp import MCPServerOverride
from ..exceptions import IrisConfigError
from .models import MCPConfig, MCPDiagnostic, MCPResolvedServer, MCPServerConfig

_ROOTS = ("mcpServers", "servers", "mcp_servers")
_SOURCE_FIELDS = {
    "command",
    "args",
    "cwd",
    "env",
    "env_vars",
    "envFile",
    "type",
    "url",
    "serverUrl",
    "headers",
    "http_headers",
    "env_http_headers",
    "bearer_token_env_var",
    "enabled",
    "disabled",
    "required",
    "startup_timeout_sec",
    "startup_timeout_ms",
    "tool_timeout_sec",
    "enabled_tools",
    "disabled_tools",
}
_BOOL = TypeAdapter(StrictBool)
_NUMBER = TypeAdapter(float)
_HEADERS = TypeAdapter(dict[str, str])
_EXPRESSION = re.compile(r"\$\{([^}]*)\}")
_VARIABLE = re.compile(r"(?:env:)?([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?", re.DOTALL)
_CLIENT_VARIABLES = {"workspaceFolder", "userHome", "CLAUDE_PLUGIN_ROOT", "CLAUDE_PROJECT_DIR"}


def _config_error(server_id: str, field: str, message: str) -> IrisConfigError:
    """错误仅保留定位信息，避免包含环境或 header 的原始输入。"""
    return IrisConfigError(message, server_id=server_id, field=field)


def _normalize_server(server_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    """归一已启用 server 的来源别名，不读取宿主环境。"""
    unknown = raw.keys() - _SOURCE_FIELDS
    if unknown:
        raise _config_error(server_id, sorted(unknown)[0], "该来源字段首版不支持")
    data = {key: value for key, value in raw.items() if key not in {"enabled", "disabled"}}
    for alias, field in (("serverUrl", "url"), ("envFile", "env_file")):
        if alias in data:
            value = data.pop(alias)
            if field in data and data[field] != value:
                raise _config_error(server_id, alias, "同义字段冲突")
            data[field] = value
    if "startup_timeout_ms" in data:
        try:
            seconds = _NUMBER.validate_python(data.pop("startup_timeout_ms"), strict=True) / 1000
        except ValidationError:
            raise _config_error(server_id, "startup_timeout_ms", "超时必须为数值") from None
        if "startup_timeout_sec" in data and data["startup_timeout_sec"] != seconds:
            raise _config_error(server_id, "startup_timeout_sec", "同义超时字段冲突")
        data["startup_timeout_sec"] = seconds
    candidates: list[tuple[str, str]] = []
    for field in ("headers", "http_headers"):
        if field in data:
            try:
                candidates.extend(_HEADERS.validate_python(data.pop(field)).items())
            except ValidationError:
                raise _config_error(server_id, field, "header 必须是字符串 mapping") from None
    if "headers" in raw or "http_headers" in raw:
        data["header_values"] = tuple(candidates)
    transport = data.pop("type", "stdio" if "command" in data else "streamable-http")
    data["transport"] = "streamable-http" if transport in ("http", "streamable_http") else transport
    return data


def load_mcp_config(path: Path, *, overrides: Mapping[str, MCPServerOverride]) -> MCPConfig:
    """读取一个明确的 MCP 文件并归一其启用声明。

    Args:
        path: JSON、JSONC 或 TOML 文件。
        overrides: 以原始 server 名为键的 Iris 策略。

    Returns:
        有效声明及 optional server 的诊断；尚不解析环境。

    Raises:
        IrisConfigError: 文件、引用、根区段或 required server 声明无效。
    """
    try:
        source = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".toml":
            document = tomllib.loads(source)
        elif path.suffix.lower() in (".json", ".jsonc"):
            document = json5.loads(source)
        else:
            raise IrisConfigError("MCP 文件只支持 JSON/JSONC/TOML")
    except (OSError, UnicodeError, ValueError) as error:
        raise IrisConfigError("MCP 文件无法读取或语法无效", path=str(path)) from error
    if not isinstance(document, dict):
        raise IrisConfigError("MCP 文件顶层必须是 mapping")
    roots = [root for root in _ROOTS if root in document]
    if len(roots) != 1 or not isinstance(document[roots[0]], dict):
        raise IrisConfigError("MCP 文件必须包含且仅包含一个 server mapping 根区段")
    declarations = document[roots[0]]
    missing = overrides.keys() - declarations.keys()
    if missing:
        raise IrisConfigError("MCP override 引用了不存在的 server", server_id=sorted(missing)[0])
    servers: list[MCPServerConfig] = []
    diagnostics: list[MCPDiagnostic] = []
    for server_id, raw in declarations.items():
        if not isinstance(raw, dict):
            raise _config_error(server_id, "server", "server 必须是 mapping")
        try:
            enabled = _BOOL.validate_python(raw.get("enabled", True))
            disabled = _BOOL.validate_python(raw.get("disabled", not enabled))
        except ValidationError:
            raise _config_error(server_id, "enabled/disabled", "开关必须为布尔值") from None
        if "enabled" in raw and "disabled" in raw and enabled == disabled:
            raise _config_error(server_id, "enabled/disabled", "开关字段冲突")
        if disabled:
            continue
        override = overrides.get(server_id)
        policy = override.model_dump(exclude_none=True) if override is not None else {}
        required = policy.get("required", raw.get("required", True))
        try:
            data = _normalize_server(server_id, raw)
            data.update(policy)
            servers.append(
                MCPServerConfig.model_validate(
                    {
                        **data,
                        "server_id": server_id,
                        "source_dir": path.resolve().parent,
                    }
                )
            )
        except ValidationError as error:
            detail = error.errors(include_input=False)[0]
            field = str(detail["loc"][0]) if detail["loc"] else "server"
            cause = _config_error(server_id, field, detail["msg"])
            if required is not False:
                raise cause from None
            diagnostics.append(
                MCPDiagnostic(server_id, "config", "MCP_CONFIG_INVALID", str(cause), field)
            )
        except IrisConfigError as error:
            if required is not False:
                raise
            diagnostics.append(
                MCPDiagnostic(
                    server_id,
                    "config",
                    "MCP_CONFIG_INVALID",
                    str(error),
                    error.context.get("field"),
                )
            )
    return MCPConfig(tuple(servers), tuple(diagnostics))


def _expand_environment(
    value: str,
    *,
    environ: Mapping[str, str],
    server_id: str,
    field: str,
) -> str:
    """单次替换支持的环境表达式，替换结果不再次解释。"""

    def replace(match: re.Match[str]) -> str:
        expression = match[1]
        variable = _VARIABLE.fullmatch(expression)
        if variable is None or expression in _CLIENT_VARIABLES:
            raise _config_error(server_id, field, "首版不支持该客户端表达式")
        name, default = variable.groups()
        current = environ.get(name)
        if default is not None:
            return current or default
        if current is None:
            raise _config_error(server_id, field, f"缺少环境变量 {name}")
        return current

    return _EXPRESSION.sub(replace, value)


def resolve_server_config(
    server: MCPServerConfig,
    *,
    environ: Mapping[str, str],
    workspace_root: Path,
) -> MCPResolvedServer:
    """解析一次宿主环境、dotenv 和路径，产出 SDK 可直接消费的配置。

    Args:
        server: 已验证的 server 声明。
        environ: 本次准备使用的宿主环境，不会被修改。
        workspace_root: 未声明 STDIO cwd 时使用的工作区。

    Returns:
        有效连接字段与本地策略。

    Raises:
        IrisConfigError: 环境缺失、客户端表达式、dotenv 读取失败或有效 header 冲突。
    """

    def expand(value: str, field: str) -> str:
        return _expand_environment(value, environ=environ, server_id=server.server_id, field=field)

    def env_value(name: str, field: str) -> str:
        if name not in environ:
            raise _config_error(server.server_id, field, f"缺少环境变量 {name}")
        return environ[name]

    def resolve_path(value: str, field: str) -> Path:
        path = Path(expand(value, field))
        return path if path.is_absolute() else (server.source_dir / path).resolve()

    env: dict[str, str] = {}
    headers: dict[str, str] = {}
    cwd: Path | None = None
    if server.transport == "stdio":
        env.update((name, env_value(name, "env_vars")) for name in server.env_vars)
        if server.env_file is not None:
            path = resolve_path(server.env_file, "envFile")
            try:
                source = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError):
                raise _config_error(server.server_id, "envFile", "无法读取 dotenv 文件") from None
            values = dotenv_values(stream=StringIO(source), interpolate=False)
            env.update(
                (name, expand(value, "envFile"))
                for name, value in values.items()
                if value is not None
            )
        env.update((name, expand(value, "env")) for name, value in server.env.items())
        cwd = workspace_root if server.cwd is None else resolve_path(server.cwd, "cwd")
    else:
        candidates = [(name, expand(value, "headers")) for name, value in server.header_values]
        candidates.extend(
            (name, env_value(variable, "env_http_headers"))
            for name, variable in server.env_http_headers.items()
        )
        if server.bearer_token_env_var is not None:
            token = env_value(server.bearer_token_env_var, "bearer_token_env_var")
            candidates.append(("authorization", f"Bearer {token}"))
        for name, value in candidates:
            key = name.lower()
            if key in headers and headers[key] != value:
                raise _config_error(server.server_id, key, "有效 header 值冲突")
            headers[key] = value
    return MCPResolvedServer(
        server_id=server.server_id,
        transport=server.transport,
        command=expand(server.command, "command") if server.command is not None else None,
        args=tuple(expand(arg, "args") for arg in server.args),
        cwd=cwd,
        env=env,
        url=expand(server.url, "url") if server.url is not None else None,
        headers=headers,
        required=server.required,
        trust_annotations=server.trust_annotations,
        startup_timeout_sec=server.startup_timeout_sec,
        tool_timeout_sec=server.tool_timeout_sec,
        enabled_tools=server.enabled_tools,
        disabled_tools=server.disabled_tools,
    )


__all__ = ["load_mcp_config", "resolve_server_config"]
