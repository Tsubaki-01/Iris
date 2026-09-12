"""官方 MCP SDK 的单 server 接入与资源生命周期。"""

from __future__ import annotations

import asyncio
import math
from contextlib import AsyncExitStack
from typing import Any

import httpx2
from mcp import Client, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from ..exceptions import IrisMCPCallError, IrisMCPError, IrisMCPToolError
from .models import MCPResolvedServer


class MCPConnection:
    """在同一个长期 task 中开闭 SDK 上下文，调用由调用方直接 await。

    manager 顺序调用 open/list_tools 后才发布工具；准备幂等由 manager 负责。
    """

    def __init__(self, config: MCPResolvedServer) -> None:
        self.config = config
        self._owner: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None]
        self._close_requested = asyncio.Event()
        self._call_lock = asyncio.Lock()
        self._client: Client
        self._protocol_version: str

    @property
    def protocol_version(self) -> str:
        """返回准备阶段实际协商的协议版本。"""
        return self._protocol_version

    async def open(self) -> None:
        """创建资源 owner 并等待协商；外层取消会等待本次打开的资源收尾。"""
        self._ready = asyncio.get_running_loop().create_future()
        self._owner = asyncio.create_task(self._serve(), name=f"iris-mcp:{self.config.server_id}")
        try:
            await self._ready
        except asyncio.CancelledError:
            self._owner.cancel()
            await asyncio.gather(self._owner, return_exceptions=True)
            self._owner = None
            raise

    async def _serve(self) -> None:
        """SDK AnyIO 上下文只能由进入它的 task 退出。"""
        try:
            async with AsyncExitStack() as stack:
                config = self.config
                if config.transport == "stdio":
                    target = StdioServerParameters(
                        command=config.command,
                        args=list(config.args),
                        env=config.env,
                        cwd=config.cwd,
                    )
                elif config.transport == "streamable-http":
                    http_client = await stack.enter_async_context(
                        httpx2.AsyncClient(
                            headers=config.headers,
                            timeout=httpx2.Timeout(config.tool_timeout_sec),
                        )
                    )
                    target = streamable_http_client(config.url, http_client=http_client)
                else:
                    target = sse_client(
                        config.url,
                        headers=config.headers,
                        timeout=config.startup_timeout_sec,
                        # SSE 流跨调用复用；空闲等待不消耗单次工具调用的期限。
                        sse_read_timeout=math.inf,
                    )
                self._client = await stack.enter_async_context(
                    Client(target, mode="auto", cache=None)
                )
                if self._client.server_capabilities.tools is None:
                    raise IrisMCPError("服务未声明 Tools 能力", server_id=config.server_id)
                self._protocol_version = self._client.protocol_version
                self._ready.set_result(None)
                await self._close_requested.wait()
        except Exception as error:
            failure = IrisMCPError("MCP 连接资源失败", server_id=self.config.server_id)
            if not self._ready.done():
                self._ready.set_exception(failure)
            else:
                raise failure from error

    async def list_tools(self) -> tuple[types.Tool, ...]:
        """收集全部工具页；cursor 仅以 None 为结束标记。

        Raises:
            IrisMCPError: 任一页发现失败，不返回部分目录。
        """
        tools: list[types.Tool] = []
        params: types.PaginatedRequestParams | None = None
        try:
            while True:
                page = await self._client.session.list_tools(params=params)
                tools.extend(page.tools)
                if page.next_cursor is None:
                    return tuple(tools)
                params = types.PaginatedRequestParams(cursor=page.next_cursor)
        except Exception as error:
            raise IrisMCPError("MCP 工具发现失败", server_id=self.config.server_id) from error

    async def call_tool(self, wire_name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """在同一调用期限内串行执行工具及最多八轮 state continuation。

        Args:
            wire_name: 已发布工具的原始协议名称。
            arguments: executor 已校验的工具参数。

        Returns:
            SDK 已解析并完成成功输出校验的结果。

        Raises:
            IrisMCPCallError: SDK MCPError，未取得可用结果。
            IrisMCPToolError: 已知结果错误或首版不支持的交互。
            TimeoutError: Iris 调用期限耗尽，包含锁等待时间。
        """
        async with asyncio.timeout(self.config.tool_timeout_sec), self._call_lock:
            state: str | None = None
            for _ in range(8):
                try:
                    result = await self._client.session.call_tool(
                        wire_name,
                        arguments,
                        allow_input_required=True,
                        input_responses=None,
                        request_state=state,
                    )
                except MCPError as error:
                    raise IrisMCPCallError(
                        "MCP 调用未取得可用结果",
                        server_id=self.config.server_id,
                        wire_name=wire_name,
                        sdk_code=error.code,
                    ) from None
                except RuntimeError:
                    # 2.2.0 的 ready、已发现工具与 allow_input_required 边界限定此错误来源。
                    raise IrisMCPToolError(
                        "MCP 成功结果不符合输出 schema",
                        code="MCP_RESULT_INVALID",
                        server_id=self.config.server_id,
                        wire_name=wire_name,
                    ) from None
                if isinstance(result, types.CallToolResult):
                    return result
                if result.input_requests:
                    raise IrisMCPToolError(
                        "首版不支持交互式 MCP 请求",
                        code="MCP_INTERACTION_UNSUPPORTED",
                        server_id=self.config.server_id,
                        wire_name=wire_name,
                    )
                state = result.request_state
            raise IrisMCPToolError(
                "MCP continuation 超过八次往返",
                code="MCP_CONTINUATION_LIMIT",
                server_id=self.config.server_id,
                wire_name=wire_name,
            )

    async def aclose(self) -> None:
        """通知 owner 关闭并等待退出；重复关闭不重复退出上下文。"""
        self._close_requested.set()
        if self._owner is not None:
            await asyncio.shield(self._owner)


__all__ = ["MCPConnection"]
