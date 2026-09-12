"""MCP 配置边界模型和已解析的进程内数据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..exceptions import IrisConfigError

type MCPTransport = Literal["stdio", "streamable-http", "sse"]


@dataclass(frozen=True, slots=True)
class MCPDiagnostic:
    """配置、连接或目录处理中可定位的诊断，不包含有效凭据。"""

    server_id: str
    stage: Literal["config", "connect", "catalog"]
    code: str
    message: str
    field: str | None = None
    wire_name: str | None = None


class MCPServerConfig(BaseModel):
    """在外部文件边界校验一次的归一 server 声明。"""

    server_id: str
    source_dir: Path
    transport: MCPTransport
    command: str | None = Field(default=None, min_length=1)
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    env_vars: tuple[str, ...] = ()
    env_file: str | None = None
    url: str | None = Field(default=None, min_length=1)
    header_values: tuple[tuple[str, str], ...] = ()
    env_http_headers: dict[str, str] = Field(default_factory=dict)
    bearer_token_env_var: str | None = None
    required: bool = Field(default=True, strict=True)
    trust_annotations: bool = Field(default=False, strict=True)
    startup_timeout_sec: float = Field(default=30, gt=0, allow_inf_nan=False, strict=True)
    tool_timeout_sec: float = Field(default=30, gt=0, allow_inf_nan=False, strict=True)
    enabled_tools: tuple[str, ...] | None = None
    disabled_tools: tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerConfig:
        """在声明边界拒绝互斥或不适用的 transport 字段。"""
        stdio_fields = {"command", "args", "cwd", "env", "env_vars", "env_file"}
        http_fields = {"url", "header_values", "env_http_headers", "bearer_token_env_var"}
        forbidden = http_fields if self.transport == "stdio" else stdio_fields
        conflict = forbidden & self.model_fields_set
        if conflict:
            field = sorted(conflict)[0]
            raise IrisConfigError("字段不适用于该 transport", field=field)
        if self.transport == "stdio" and self.command is None:
            raise IrisConfigError("STDIO 缺少 command", field="command")
        if self.transport != "stdio" and self.url is None:
            raise IrisConfigError("HTTP/SSE 缺少 url", field="url")
        return self


@dataclass(frozen=True, slots=True)
class MCPResolvedServer:
    """准备时求值后的 SDK 输入与有效策略，仅驻留内存。"""

    server_id: str
    transport: MCPTransport
    command: str | None
    args: tuple[str, ...]
    cwd: Path | None
    env: dict[str, str]
    url: str | None
    headers: dict[str, str]
    required: bool
    trust_annotations: bool
    startup_timeout_sec: float
    tool_timeout_sec: float
    enabled_tools: tuple[str, ...] | None
    disabled_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """启用且声明有效的 servers，以及被跳过的 optional 诊断。"""

    servers: tuple[MCPServerConfig, ...]
    diagnostics: tuple[MCPDiagnostic, ...] = ()


__all__ = ["MCPConfig", "MCPDiagnostic", "MCPResolvedServer", "MCPServerConfig", "MCPTransport"]
