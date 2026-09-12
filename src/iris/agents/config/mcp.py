"""Agent 引用 MCP 文件的声明与本地策略。"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class MCPServerOverride(BaseModel):
    """按原始 server 名覆盖 Iris 的准备与工具策略。"""

    required: bool | None = Field(default=None, strict=True)
    trust_annotations: bool | None = Field(default=None, strict=True)
    startup_timeout_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    tool_timeout_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)

    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentMCPConfig(BaseModel):
    """引用单个 MCP 文件；读取和环境求值由后续装配与准备负责。"""

    path: Path
    overrides: dict[str, MCPServerOverride] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("path")
    @classmethod
    def _resolve_path(cls, value: Path, info: ValidationInfo) -> Path:
        """沿用 Agent YAML 的相对路径基准。"""
        config_path: Path | None = (info.context or {}).get("config_path")
        if config_path is not None and not value.is_absolute():
            return (config_path.parent / value).resolve()
        return value


__all__ = ["AgentMCPConfig", "MCPServerOverride"]
