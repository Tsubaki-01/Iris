"""MCP 本地服务测试共享 fixture。"""

import asyncio
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from iris.mcp.connection import MCPConnection
from iris.mcp.models import MCPResolvedServer

from .fixtures.servers import ServerScenario


@pytest.fixture
def stdio_config(tmp_path: Path) -> MCPResolvedServer:
    """使用当前 uv 环境启动官方 SDK fixture server。"""
    script = Path(__file__).parent / "fixtures" / "servers.py"
    return MCPResolvedServer(
        "fixture",
        "stdio",
        sys.executable,
        (str(script),),
        tmp_path,
        {"IRIS_MCP_TEST_LOG": str(tmp_path / "server.jsonl")},
        None,
        {},
        True,
        False,
        8,
        3,
        None,
        (),
    )


@pytest_asyncio.fixture
async def stdio_connection(stdio_config: MCPResolvedServer) -> AsyncIterator[MCPConnection]:
    """完整准备并在每次测试后关闭真实 STDIO 连接。"""
    connection = MCPConnection(stdio_config)
    try:
        async with asyncio.timeout(10):
            await connection.open()
            await connection.list_tools()
        yield connection
    finally:
        await connection.aclose()


@asynccontextmanager
async def serve_http(scenario: ServerScenario, *, sse: bool = False) -> AsyncIterator[str]:
    """复用官方服务与本机 HTTP listener 的生命周期。"""
    if sse:
        transport = SseServerTransport("/messages/")

        class SSEEndpoint:
            """使用 ASGI Route 保留消息 endpoint 的根路径。"""

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                async with transport.connect_sse(scope, receive, send) as streams:
                    await scenario.server.run(
                        *streams, scenario.server.create_initialization_options()
                    )

        app = Starlette(
            routes=[
                Route("/sse", endpoint=SSEEndpoint()),
                Mount("/messages/", app=transport.handle_post_message),
            ]
        )
    else:
        app = scenario.server.streamable_http_app()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{port}" + ("/sse" if sse else "/mcp")
        finally:
            server.should_exit = True
            await task


@pytest_asyncio.fixture
async def legacy_http(tmp_path: Path) -> AsyncIterator[tuple[MCPResolvedServer, ServerScenario]]:
    """在本机端口运行仅通过旧版握手建立连接的 HTTP fixture。"""
    scenario = ServerScenario(legacy=True)
    async with serve_http(scenario) as url:
        yield (
            MCPResolvedServer(
                "legacy",
                "streamable-http",
                None,
                (),
                None,
                {},
                url,
                {},
                True,
                False,
                5,
                2,
                None,
                (),
            ),
            scenario,
        )


@pytest_asyncio.fixture
async def modern_sse() -> AsyncIterator[tuple[MCPResolvedServer, ServerScenario]]:
    """现代协议在显式 SSE transport 上运行，供空闲复用回归使用。"""
    scenario = ServerScenario()
    async with serve_http(scenario, sse=True) as url:
        yield (
            MCPResolvedServer(
                "sse", "sse", None, (), None, {}, url, {}, True, False, 5, 0.3, None, ()
            ),
            scenario,
        )
