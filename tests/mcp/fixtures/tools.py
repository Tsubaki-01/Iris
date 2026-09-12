"""MCP adapter 测试的连接替身与工具构造。"""

from dataclasses import replace
from typing import Any, cast

from mcp import types

from iris.mcp.catalog import build_catalog
from iris.mcp.connection import MCPConnection
from iris.mcp.models import MCPResolvedServer
from iris.mcp.tools import MCPTool
from iris.tools import (
    BaseTool,
    PermissionDecision,
    PermissionEffect,
    PermissionPolicy,
    ToolExecutionContext,
)


class ResultConnection:
    """记录 adapter 的实际调用，并返回已解析的 SDK 结果。"""

    def __init__(
        self, result: types.CallToolResult | None = None, error: Exception | None = None
    ) -> None:
        self.result = (
            result
            if result is not None
            else types.CallToolResult(content=[types.TextContent(text="ok")])
        )
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, wire_name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """只模拟连接边界，不替代 adapter/executor。"""
        self.calls.append((wire_name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


class AllowTools(PermissionPolicy):
    """显式允许测试中的 MCP 写工具，覆盖 claim 后的真实链路。"""

    def check(
        self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
    ) -> PermissionDecision:
        """返回固定的测试策略。"""
        return PermissionDecision(effect=PermissionEffect.ALLOW)

    def fingerprint_payload(self) -> dict[str, object]:
        """声明稳定测试策略。"""
        return {"type": "test-allow"}


def make_tool(
    config: MCPResolvedServer,
    *,
    trust: bool = True,
    result: types.CallToolResult | None = None,
    error: Exception | None = None,
    schema: dict[str, Any] | None = None,
) -> tuple[MCPTool, ResultConnection]:
    """从真实 catalog 构建供 executor 使用的 MCPTool。"""
    catalog, diagnostics = build_catalog(
        replace(config, trust_annotations=trust),
        [
            types.Tool(
                name="Wire.Name",
                input_schema=schema or {"type": "object"},
                annotations=types.ToolAnnotations(read_only_hint=True),
            )
        ],
    )
    assert not diagnostics
    connection = ResultConnection(result, error)
    return MCPTool(catalog[0], cast(MCPConnection, connection)), connection
