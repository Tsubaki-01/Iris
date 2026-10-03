"""Agent 对独立 Decision 配置文件的引用。"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class AgentDecisionConfig(BaseModel):
    """按声明它的 Agent YAML 解析相对路径，不在模型内读取文件。"""

    path: Path

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("path")
    @classmethod
    def _resolve_path(cls, value: Path, info: ValidationInfo) -> Path:
        """在 YAML 输入边界解析一次文件相对路径。"""
        config_path: Path | None = (info.context or {}).get("config_path")
        if config_path is not None and not value.is_absolute():
            return (config_path.parent / value).resolve()
        return value


__all__ = ["AgentDecisionConfig"]
