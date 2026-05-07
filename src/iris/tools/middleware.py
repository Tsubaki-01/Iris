"""工具 middleware 扩展点。"""

from __future__ import annotations

from typing import Protocol

from .base import ToolExecutionContext, ToolResult


class ToolMiddleware(Protocol):
    """阶段 2 仅定义最小 middleware 协议。"""

    async def after_execute(
        self,
        result: ToolResult,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """工具执行后处理结果。"""
        ...
