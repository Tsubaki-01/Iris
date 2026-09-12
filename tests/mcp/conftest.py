"""MCP 本地服务测试共享 fixture。"""

import asyncio
import socket
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn

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


@pytest_asyncio.fixture
async def legacy_http(tmp_path: Path) -> AsyncIterator[tuple[MCPResolvedServer, ServerScenario]]:
    """在本机端口运行仅通过旧版握手建立连接的 HTTP fixture。"""
    scenario = ServerScenario(legacy=True)
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
            config = MCPResolvedServer(
                "legacy",
                "streamable-http",
                None,
                (),
                None,
                {},
                f"http://127.0.0.1:{port}/mcp",
                {},
                True,
                False,
                5,
                2,
                None,
                (),
            )
            yield config, scenario
        finally:
            server.should_exit = True
            await task
