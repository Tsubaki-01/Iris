"""工具 middleware 扩展点。"""

from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolExecutionContext, ToolResult


class ToolMiddleware:
    """工具执行生命周期 middleware 基类。

    所有钩子默认不改变执行流程。调用方仍可传入只实现部分钩子的旧式对象，
    executor 会按存在的钩子动态调用，以保持阶段 2 middleware 兼容。
    """

    async def before_call(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> None:
        """工具调用前执行。"""

    async def after_call(
        self,
        tool: BaseTool,
        result: ToolResult,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """工具执行后处理结果。"""
        return result

    async def on_error(
        self,
        tool: BaseTool,
        error: Exception,
        context: ToolExecutionContext,
    ) -> ToolResult | None:
        """工具执行错误时可返回替代结果。"""
        return None

    async def after_execute(
        self,
        result: ToolResult,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """兼容阶段 2 的执行后处理钩子。"""
        return result
