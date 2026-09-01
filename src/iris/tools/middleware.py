"""工具 middleware 扩展点。"""

from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolExecutionContext, ToolResult


class ToolMiddleware:
    """工具执行生命周期 middleware 基类。

    所有 async 钩子默认不改变执行流程；自定义 middleware 直接继承并覆盖所需钩子。
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
