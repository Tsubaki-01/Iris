"""通过官方 SDK 提供可控的本地协议服务，供各阶段复用。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError


class ServerScenario:
    """可观察的工具调用与取消，不依赖真实外部服务。"""

    def __init__(self, *, legacy: bool = False, log_path: Path | None = None) -> None:
        self.log_path = log_path
        self.calls: list[dict[str, Any]] = []
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.server = Server(
            "iris-test", version="1", on_list_tools=self.list_tools, on_call_tool=self.call_tool
        )
        if legacy:
            self.server.add_request_handler(
                "server/discover", types.RequestParams, self.reject_discover
            )

    def record(self, **event: Any) -> None:
        """记录 fixture 的可观察事件。"""
        self.calls.append(event)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")

    async def reject_discover(
        self, ctx: ServerRequestContext, params: types.RequestParams
    ) -> types.DiscoverResult:
        """让 SDK 通过公开 METHOD_NOT_FOUND 响应进入旧版握手。"""
        raise MCPError(types.METHOD_NOT_FOUND, "legacy fixture")

    async def list_tools(
        self, ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        """暴露覆盖输出与协议分支的固定目录。"""
        self.record(event="list", protocol=ctx.protocol_version)
        names = (
            "echo",
            "invalid",
            "null",
            "business_error",
            "rpc_error",
            "slow",
            "state",
            "interactive",
            "loop",
            "disconnect",
        )
        output_schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=name,
                    input_schema={"type": "object"},
                    output_schema=({"type": "null"} if name == "null" else output_schema)
                    if name in {"echo", "invalid", "null", "business_error"}
                    else None,
                    annotations=types.ToolAnnotations(read_only_hint=True),
                )
                for name in names
                if name != "null" or ctx.protocol_version == "2026-07-28"
            ]
        )

    async def call_tool(
        self, ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult | types.InputRequiredResult:
        """返回真实 wire 结果或由 SDK 传递的 RPC 错误。"""
        name = params.name
        self.record(
            event="call",
            name=name,
            state=params.request_state,
            arguments=params.arguments,
            protocol=ctx.protocol_version,
        )
        self.entered.set()
        if name == "rpc_error":
            raise MCPError(-32001, "fixture RPC error")
        if name == "disconnect":
            # 独立 STDIO 测试进程模拟服务端处理后断线，不影响 pytest 进程。
            os._exit(0)
        if name == "slow":
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                self.record(event="cancelled", name=name)
                raise
        if name == "state" and params.request_state is None:
            return types.InputRequiredResult(request_state="opaque-state")
        if name == "loop":
            return types.InputRequiredResult(request_state="again")
        if name == "interactive":
            return types.InputRequiredResult(input_requests={"roots": types.ListRootsRequest()})
        if name == "business_error":
            return types.CallToolResult(
                content=[types.TextContent(text="business failed")], is_error=True
            )
        if name == "invalid":
            return types.CallToolResult(content=[], structured_content={"value": "wrong"})
        if name == "null":
            return types.CallToolResult(content=[], structured_content=None)
        value = (params.arguments or {}).get("value", 7)
        return types.CallToolResult(
            content=[types.TextContent(text=f"value={value}")], structured_content={"value": value}
        )


async def serve_stdio(*, legacy: bool) -> None:
    """以真实 STDIO transport 服务至 client 关闭。"""
    log_path = os.environ.get("IRIS_MCP_TEST_LOG")
    scenario = ServerScenario(legacy=legacy, log_path=Path(log_path) if log_path else None)
    try:
        async with stdio_server() as (read, write):
            await scenario.server.run(read, write, scenario.server.create_initialization_options())
    finally:
        scenario.record(event="closed")


if __name__ == "__main__":
    asyncio.run(serve_stdio(legacy="--legacy" in sys.argv))
