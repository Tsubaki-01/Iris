"""锁定官方 SDK 2.2 的实际协商、输出验证和取消边界。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from mcp import Client

from iris.exceptions import IrisMCPCallError, IrisMCPToolError
from iris.mcp.connection import MCPConnection
from iris.mcp.models import MCPResolvedServer

from .fixtures.servers import ServerScenario


@pytest.mark.asyncio
async def test_real_modern_stdio_and_outputs(stdio_connection: MCPConnection) -> None:
    assert stdio_connection.protocol_version == "2026-07-28"
    result = await stdio_connection.call_tool("echo", {"value": 42})
    assert result.structured_content == {"value": 42}
    for name in ("invalid", "null"):
        with pytest.raises(IrisMCPToolError) as error:
            await stdio_connection.call_tool(name, {})
        assert error.value.code == "MCP_RESULT_INVALID"
    result = await stdio_connection.call_tool("business_error", {})
    assert result.is_error and result.content[0].text == "business failed"


@pytest.mark.asyncio
async def test_real_continuations(stdio_connection: MCPConnection) -> None:
    result = await stdio_connection.call_tool("state", {})
    assert result.structured_content == {"value": 7}
    for name, code in (
        ("loop", "MCP_CONTINUATION_LIMIT"),
        ("interactive", "MCP_INTERACTION_UNSUPPORTED"),
    ):
        with pytest.raises(IrisMCPToolError) as error:
            await stdio_connection.call_tool(name, {})
        assert error.value.code == code


@pytest.mark.asyncio
async def test_real_rpc_error_is_uncertain(stdio_connection: MCPConnection) -> None:
    with pytest.raises(IrisMCPCallError):
        await stdio_connection.call_tool("rpc_error", {})


@pytest.mark.asyncio
async def test_real_disconnect_is_uncertain_without_replay(stdio_config: MCPResolvedServer) -> None:
    connection = MCPConnection(stdio_config)
    await connection.open()
    await connection.list_tools()
    try:
        with pytest.raises(IrisMCPCallError):
            await connection.call_tool("disconnect", {})
    finally:
        await connection.aclose()
    events = [
        json.loads(line)
        for line in Path(stdio_config.env["IRIS_MCP_TEST_LOG"]).read_text().splitlines()
    ]
    assert sum(event.get("name") == "disconnect" for event in events) == 1


@pytest.mark.asyncio
async def test_real_legacy_http(legacy_http: tuple[MCPResolvedServer, ServerScenario]) -> None:
    config, scenario = legacy_http
    connection = MCPConnection(config)
    try:
        await connection.open()
        assert connection.protocol_version == "2025-11-25"
        await connection.list_tools()
        result = await connection.call_tool("echo", {"value": 9})
        assert result.structured_content == {"value": 9}
    finally:
        await connection.aclose()
    assert [event["name"] for event in scenario.calls if event["event"] == "call"] == ["echo"]


@pytest.mark.asyncio
async def test_real_sdk_cancellation_keeps_stdio_usable(
    stdio_connection: MCPConnection, tmp_path: Path
) -> None:
    task = asyncio.create_task(stdio_connection.call_tool("slow", {}))
    async with asyncio.timeout(3):
        while '"name": "slow"' not in (tmp_path / "server.jsonl").read_text():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await stdio_connection.call_tool("echo", {"value": 2})
    assert result.structured_content == {"value": 2}


@pytest.mark.asyncio
async def test_sdk_read_timeout_is_distinct_from_iris_deadline(
    stdio_config: MCPResolvedServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def short_sdk_timeout(target, *, mode: str, cache: None) -> Client:
        return Client(target, mode=mode, cache=cache, read_timeout_seconds=0.15)

    monkeypatch.setattr("iris.mcp.connection.Client", short_sdk_timeout)
    connection = MCPConnection(replace(stdio_config, tool_timeout_sec=2))
    await connection.open()
    await connection.list_tools()
    try:
        with pytest.raises(IrisMCPCallError):
            await connection.call_tool("slow", {})
    finally:
        await connection.aclose()
