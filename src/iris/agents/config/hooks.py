"""Hooks 与工具 Middleware 的声明式配置边界。"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...hooks.models import HookEventName

type _NonemptyText = Annotated[str, Field(pattern=r"\S")]


class PythonHookHandlerConfig(BaseModel):
    """由同步工厂构造 Python 异步处理器的声明。"""

    type: Literal["python"]
    factory: _NonemptyText
    options: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True, extra="forbid")


class CommandHookHandlerConfig(BaseModel):
    """在当前 Agent 命令环境中执行的固定脚本声明。"""

    type: Literal["command"]
    command: _NonemptyText
    model_config = ConfigDict(frozen=True, extra="forbid")


type HookHandlerConfig = Annotated[
    PythonHookHandlerConfig | CommandHookHandlerConfig, Field(discriminator="type")
]


class HookConfig(BaseModel):
    """有序事件处理器；tools 使用实际工具调用名而非 builtin 配置键。"""

    name: _NonemptyText
    event: HookEventName
    handler: HookHandlerConfig
    tools: Annotated[tuple[_NonemptyText, ...], Field(min_length=1)] | None = None
    timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_tool_filter(self) -> Self:
        """仅工具事件接受非空的工具名过滤。"""
        if self.tools is not None and self.event not in {"tool.before", "tool.after"}:
            raise ValueError("hooks.tools 只能用于工具事件")
        return self


class ToolMiddlewareConfig(BaseModel):
    """由同步工厂构造一个 ToolMiddleware 实例的声明。"""

    factory: _NonemptyText
    options: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True, extra="forbid")


class MiddlewareConfig(BaseModel):
    """只开放普通工具包装链的 Middleware 配置。"""

    tools: tuple[ToolMiddlewareConfig, ...] = ()
    model_config = ConfigDict(frozen=True, extra="forbid")


__all__ = [
    "CommandHookHandlerConfig",
    "HookConfig",
    "HookHandlerConfig",
    "MiddlewareConfig",
    "PythonHookHandlerConfig",
    "ToolMiddlewareConfig",
]
