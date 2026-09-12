"""单连接资源 ownership、分页、串行和取消契约。"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from mcp import types
from mcp.shared.exceptions import MCPError

from iris.exceptions import IrisMCPCallError, IrisMCPError, IrisMCPToolError
from iris.mcp.connection import MCPConnection
from iris.mcp.models import MCPResolvedServer


def server_config(tmp_path: Path, timeout: float = 1) -> MCPResolvedServer:
    """连接测试的已解析输入。"""
    return MCPResolvedServer(
        "s", "stdio", "python", (), tmp_path, {}, None, {}, True, False, 1, timeout, None, ()
    )


class ControlledClient:
    """仅替换 SDK 边界，记录真实 Connection 发起的请求与上下文。"""

    def __init__(self) -> None:
        self.session = self
        self.protocol_version = "2026-07-28"
        self.server_capabilities = types.ServerCapabilities(tools=types.ToolsCapability())
        self.tasks: list[asyncio.Task | None] = []
        self.cursors: list[str | None] = []
        self.calls: list[tuple[str, dict, str | None]] = []
        self.responses: list[Any] = [types.CallToolResult(content=[])]
        self.block = asyncio.Event()
        self.entered = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.hold = False
        self.open_error = False
        self.close_error = False

    async def __aenter__(self) -> "ControlledClient":
        self.tasks.append(asyncio.current_task())
        if self.open_error:
            raise OSError("connection failed")
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.tasks.append(asyncio.current_task())
        if self.close_error:
            raise OSError("close failed")

    async def list_tools(
        self, *, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        cursor = None if params is None else params.cursor
        self.cursors.append(cursor)
        return types.ListToolsResult(
            tools=[types.Tool(name=f"page{len(self.cursors)}", input_schema={"type": "object"})],
            next_cursor="" if cursor is None else None,
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        *,
        allow_input_required: bool,
        input_responses: None,
        request_state: str | None,
    ) -> Any:
        assert allow_input_required and input_responses is None
        self.calls.append((name, arguments, request_state))
        self.entered.set()
        try:
            if self.hold:
                await self.block.wait()
            result = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self.cleaned.set()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ControlledClient:
    """让每次测试独立控制连接边界。"""
    result = ControlledClient()

    def create(target: Any, *, mode: str, cache: None) -> ControlledClient:
        assert mode == "auto" and cache is None
        return result

    monkeypatch.setattr("iris.mcp.connection.Client", create)
    return result


@pytest.mark.asyncio
async def test_owner_and_opaque_pagination(tmp_path: Path, client: ControlledClient) -> None:
    connection = MCPConnection(server_config(tmp_path))
    assert client.tasks == []
    await connection.open()
    tools = await connection.list_tools()
    assert [tool.name for tool in tools] == ["page1", "page2"]
    assert client.cursors == [None, ""]
    assert connection.protocol_version == "2026-07-28"
    await connection.aclose()
    await connection.aclose()
    assert len(client.tasks) == 2 and client.tasks[0] is client.tasks[1]
    assert client.tasks[0] is not asyncio.current_task()


@pytest.mark.asyncio
async def test_open_failure_reaches_waiter(tmp_path: Path, client: ControlledClient) -> None:
    client.open_error = True
    connection = MCPConnection(server_config(tmp_path))
    with pytest.raises(IrisMCPError):
        await asyncio.wait_for(connection.open(), 1)
    await connection.aclose()


@pytest.mark.asyncio
async def test_cancel_during_open_finishes_owner_cleanup(
    tmp_path: Path,
    client: ControlledClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def opening(self: ControlledClient) -> ControlledClient:
        self.entered.set()
        try:
            await self.block.wait()
            return self
        finally:
            self.cleaned.set()

    monkeypatch.setattr(ControlledClient, "__aenter__", opening)
    connection = MCPConnection(server_config(tmp_path))
    task = asyncio.create_task(connection.open())
    await client.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.cleaned.is_set()
    await connection.aclose()


@pytest.mark.asyncio
async def test_listing_failure_never_returns_partial_catalog(
    tmp_path: Path,
    client: ControlledClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing = client.list_tools

    async def fail_second(*, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        if params is not None:
            raise MCPError(-32000, "page failure")
        return await listing(params=params)

    monkeypatch.setattr(client, "list_tools", fail_second)
    connection = MCPConnection(server_config(tmp_path))
    await connection.open()
    with pytest.raises(IrisMCPError):
        await connection.list_tools()
    await connection.aclose()
    assert len(client.tasks) == 2


@pytest.mark.asyncio
async def test_continuation_shares_deadline_and_cancellation_stops_next_round(
    tmp_path: Path,
    client: ControlledClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def continuation(name: str, arguments: dict, **kwargs: Any) -> types.InputRequiredResult:
        client.calls.append((name, arguments, kwargs["request_state"]))
        await asyncio.sleep(0.035)
        return types.InputRequiredResult(request_state="next")

    monkeypatch.setattr(client, "call_tool", continuation)
    connection = MCPConnection(server_config(tmp_path, timeout=0.06))
    await connection.open()
    with pytest.raises(TimeoutError):
        await connection.call_tool("state", {})
    # Windows 调度可能在首轮就耗尽预算；关键是共享期限终止，而不是跑满八轮。
    assert 1 <= len(client.calls) < 8
    await connection.aclose()


@pytest.mark.asyncio
async def test_close_failure_reaches_host(tmp_path: Path, client: ControlledClient) -> None:
    connection = MCPConnection(server_config(tmp_path))
    await connection.open()
    client.close_error = True
    with pytest.raises(IrisMCPError):
        await connection.aclose()


@pytest.mark.asyncio
async def test_continuation_and_error_categories(tmp_path: Path, client: ControlledClient) -> None:
    connection = MCPConnection(server_config(tmp_path))
    await connection.open()
    try:
        client.responses = [
            types.InputRequiredResult(request_state="opaque"),
            types.CallToolResult(content=[]),
        ]
        await connection.call_tool("Wire.Name", {"x": 1})
        assert client.calls == [("Wire.Name", {"x": 1}, None), ("Wire.Name", {"x": 1}, "opaque")]
        for exception, expected in [
            (MCPError(-32000, "no response"), IrisMCPCallError),
            (RuntimeError("invalid output"), IrisMCPToolError),
        ]:
            client.responses = [exception]
            with pytest.raises(expected):
                await connection.call_tool("Wire.Name", {})
        client.responses = [types.InputRequiredResult(request_state="again")]
        count = len(client.calls)
        with pytest.raises(IrisMCPToolError, match="continuation|continuation|往返") as error:
            await connection.call_tool("Wire.Name", {})
        assert error.value.code == "MCP_CONTINUATION_LIMIT"
        assert len(client.calls) - count == 8
    finally:
        await connection.aclose()


@pytest.mark.asyncio
async def test_call_cancellation_keeps_connection_and_releases_lock(
    tmp_path: Path,
    client: ControlledClient,
) -> None:
    connection = MCPConnection(server_config(tmp_path))
    await connection.open()
    client.hold = True
    first = asyncio.create_task(connection.call_tool("one", {}))
    await client.entered.wait()
    queued = asyncio.create_task(connection.call_tool("two", {}))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert client.cleaned.is_set() and len(client.tasks) == 1
    assert [call[0] for call in client.calls] == ["one"]
    client.hold = False
    await connection.call_tool("three", {})
    await connection.aclose()


@pytest.mark.asyncio
async def test_lock_wait_uses_call_deadline(tmp_path: Path, client: ControlledClient) -> None:
    connection = MCPConnection(server_config(tmp_path, timeout=0.08))
    await connection.open()
    client.hold = True
    first = asyncio.create_task(connection.call_tool("one", {}))
    await client.entered.wait()
    with pytest.raises(TimeoutError):
        await connection.call_tool("two", {})
    with pytest.raises(TimeoutError):
        await first
    await connection.aclose()


@pytest.mark.asyncio
async def test_transport_target_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: ControlledClient
) -> None:
    from contextlib import asynccontextmanager

    calls: list[tuple[str, dict]] = []

    @asynccontextmanager
    async def unused_transport(url: str, **kwargs: Any) -> AsyncIterator[None]:
        yield None

    def transport(url: str, **kwargs: Any) -> AbstractAsyncContextManager[None]:
        calls.append((url, kwargs))
        return unused_transport(url, **kwargs)

    monkeypatch.setattr("iris.mcp.connection.sse_client", transport)
    config = replace(
        server_config(tmp_path),
        transport="sse",
        command=None,
        cwd=None,
        url="https://example.test/sse",
        headers={"authorization": "Bearer test"},
    )
    connection = MCPConnection(config)
    await connection.open()
    await connection.aclose()
    assert calls[0][0] == "https://example.test/sse"
    assert calls[0][1]["headers"] == {"authorization": "Bearer test"}
