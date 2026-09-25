"""MCP 多服务准备、一次发布与资源清理。"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcp import types

from iris.exceptions import IrisConfigError, IrisMCPError, IrisToolValidationError
from iris.mcp.config import load_mcp_config
from iris.mcp.manager import MCPManager
from iris.mcp.models import MCPResolvedServer
from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolRegistry

from .fixtures.tools import AllowTools
from .test_connection import ControlledClient


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """只替换外部连接边界；manager、config、catalog 和 registry 保持真实。"""
    log: list[str] = []

    class Connection:
        protocol_version = "2026-07-28"

        def __init__(self, config: MCPResolvedServer) -> None:
            self.config = config

        async def open(self) -> None:
            log.append(f"open:{self.config.server_id}")
            if self.config.command == "fail":
                raise IrisMCPError("fixture unavailable")
            if self.config.command == "slow":
                await asyncio.Event().wait()

        async def list_tools(self) -> tuple[types.Tool, ...]:
            log.append(f"list:{self.config.server_id}")
            if self.config.command == "list-fail":
                raise IrisMCPError("listing failed")
            if self.config.command == "empty":
                return ()
            return (types.Tool(name="echo", input_schema={"type": "object"}),)

        async def aclose(self) -> None:
            log.append(f"close:{self.config.server_id}")
            if self.config.command == "close-fail":
                raise IrisMCPError("close failed")

    monkeypatch.setattr("iris.mcp.manager.MCPConnection", Connection)
    return log


def manager_for(tmp_path: Path, servers: dict[str, Any], registry: ToolRegistry) -> MCPManager:
    """从文件入口准备真实配置。"""
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": servers}))
    return MCPManager(
        load_mcp_config(path, overrides={}), registry=registry, workspace_root=tmp_path
    )


@pytest.mark.asyncio
async def test_prepare_is_ordered_idempotent_and_publishes_to_existing_view(
    tmp_path: Path,
    events: list[str],
) -> None:
    registry = ToolRegistry()
    view = registry.view()
    manager = manager_for(tmp_path, {"z": {"command": "ok"}, "a": {"command": "ok"}}, registry)
    assert manager.snapshot is None and not events
    snapshots = await asyncio.gather(manager.prepare(), manager.prepare())
    assert [item.config.server_id for item in snapshots[0].servers] == ["a", "z"]
    assert [tool.name for tool in view.active_tools] == ["mcp__a__echo", "mcp__z__echo"]
    await manager.prepare()
    assert events == ["open:a", "list:a", "open:z", "list:z"]
    await manager.aclose()
    await manager.aclose()
    assert events.count("close:a") == events.count("close:z") == 1
    with pytest.raises(IrisMCPError):
        await manager.prepare()


@pytest.mark.asyncio
async def test_deferred_catalog_still_prepares_complete_connection(
    tmp_path: Path, events: list[str]
) -> None:
    registry = ToolRegistry()
    original = manager_for(tmp_path, {"external": {"command": "ok"}}, registry)
    manager = MCPManager(
        original.config, registry=registry, workspace_root=tmp_path, defer_tools=True
    )
    snapshot = await manager.prepare()
    assert len(snapshot.servers) == 1
    assert events == ["open:external", "list:external"]
    assert registry.view().active_tools == []
    assert registry.view().available_tools[0].definition.deferred is True
    assert registry.search_deferred("echo")[0].name == "mcp__external__echo"
    await manager.aclose()
    assert events[-1] == "close:external"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["fail", "list-fail"])
async def test_required_failure_preserves_registry_and_closes_every_connection(
    tmp_path: Path,
    events: list[str],
    command: str,
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "local", name="local", description="local")
    manager = manager_for(tmp_path, {"a": {"command": "ok"}, "b": {"command": command}}, registry)
    with pytest.raises(IrisMCPError):
        await manager.prepare()
    assert [tool.name for tool in registry.view().active_tools] == ["local"]
    assert events.count("close:a") == events.count("close:b") == 1
    assert manager.snapshot is None
    with pytest.raises(IrisMCPError):
        await manager.prepare()


@pytest.mark.asyncio
@pytest.mark.parametrize("only_optional", [True, False])
async def test_optional_failure_is_diagnostic_and_not_retried(
    tmp_path: Path, events: list[str], only_optional: bool
) -> None:
    registry = ToolRegistry()
    servers = {"bad": {"command": "fail", "required": False}}
    if not only_optional:
        servers["good"] = {"command": "ok"}
    manager = manager_for(tmp_path, servers, registry)
    snapshot = await manager.prepare()
    assert [s.config.server_id for s in snapshot.servers] == ([] if only_optional else ["good"])
    assert snapshot.diagnostics[0].server_id == "bad"
    await manager.prepare()
    assert events.count("open:bad") == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_missing_environment_and_empty_catalog(
    tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_MCP_TEST_MISSING", raising=False)
    registry = ToolRegistry()
    manager = manager_for(
        tmp_path,
        {
            "missing": {"command": "${IRIS_MCP_TEST_MISSING}", "required": False},
            "empty": {"command": "empty"},
        },
        registry,
    )
    snapshot = await manager.prepare()
    assert len(snapshot.servers) == 1 and snapshot.servers[0].tools == ()
    assert snapshot.diagnostics[0].stage == "config"
    assert not registry.view().active_tools
    await manager.aclose()
    required = manager_for(tmp_path, {"missing": {"command": "${IRIS_MCP_TEST_MISSING}"}}, registry)
    with pytest.raises(IrisConfigError):
        await required.prepare()


@pytest.mark.asyncio
async def test_startup_deadline_closes_failed_connection(tmp_path: Path, events: list[str]) -> None:
    manager = manager_for(
        tmp_path, {"slow": {"command": "slow", "startup_timeout_sec": 0.04}}, ToolRegistry()
    )
    with pytest.raises(IrisMCPError):
        await manager.prepare()
    assert events == ["open:slow", "close:slow"]


@pytest.mark.asyncio
async def test_registry_collision_closes_all_resources_without_partial_tools(
    tmp_path: Path, events: list[str]
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "local", name="mcp__b__echo", description="local")
    manager = manager_for(tmp_path, {"a": {"command": "ok"}, "b": {"command": "ok"}}, registry)
    with pytest.raises(IrisToolValidationError):
        await manager.prepare()
    assert [tool.name for tool in registry.view().active_tools] == ["mcp__b__echo"]
    assert events.count("close:a") == events.count("close:b") == 1


@pytest.mark.asyncio
async def test_cancellation_keeps_original_error_and_attempts_all_closes(
    tmp_path: Path, events: list[str]
) -> None:
    manager = manager_for(
        tmp_path, {"a": {"command": "close-fail"}, "b": {"command": "slow"}}, ToolRegistry()
    )
    task = asyncio.create_task(manager.prepare())
    async with asyncio.timeout(1):
        while "open:b" not in events:
            await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events.count("close:a") == events.count("close:b") == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_explicit_close_reports_error_after_attempting_all(
    tmp_path: Path, events: list[str]
) -> None:
    manager = manager_for(
        tmp_path, {"a": {"command": "ok"}, "b": {"command": "close-fail"}}, ToolRegistry()
    )
    await manager.prepare()
    with pytest.raises(IrisMCPError):
        await manager.aclose()
    assert events.count("close:a") == events.count("close:b") == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_real_manager_discovers_publishes_calls_and_closes_stdio(
    tmp_path: Path,
    stdio_config: MCPResolvedServer,
) -> None:
    """真实 SDK 连接经 manager/registry/executor 完成同一条工具调用。"""
    registry = ToolRegistry()
    manager = manager_for(
        tmp_path,
        {
            "fixture": {
                "command": stdio_config.command,
                "args": list(stdio_config.args),
                "env": stdio_config.env,
            }
        },
        registry,
    )
    try:
        snapshot = await manager.prepare()
        assert snapshot.servers[0].protocol_version == "2026-07-28"
        result = await ToolExecutor(registry, permission_policy=AllowTools()).execute_one(
            ToolUseBlock(id="real", name="mcp__fixture__echo", input={"value": 12}),
            ToolExecutionContext(workspace_root=tmp_path, session_id="session"),
        )
        assert not result.is_error
        assert json.loads(result.artifact.path.read_text())["structuredContent"] == {"value": 12}
    finally:
        await manager.aclose()
    events = [json.loads(line) for line in (tmp_path / "server.jsonl").read_text().splitlines()]
    assert [event["name"] for event in events if event["event"] == "call"] == ["echo"]
    assert events[-1]["event"] == "closed"


@pytest.mark.asyncio
async def test_repeated_prepare_cancel_waits_all_sdk_connection_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第二次取消不能打断失败清理，提前丢失第一条已准备连接。"""
    listing = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()
    log: list[str] = []

    class Client(ControlledClient):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def list_tools(
            self, *, params: types.PaginatedRequestParams | None
        ) -> types.ListToolsResult:
            if self.name == "b":
                listing.set()
                await asyncio.Event().wait()
            return types.ListToolsResult(tools=[])

        async def __aexit__(self, *args: Any) -> None:
            log.append(f"closing:{self.name}")
            if self.name == "b":
                closing.set()
                await release.wait()
            await super().__aexit__(*args)
            log.append(f"closed:{self.name}")

    def client(target: Any, *, mode: str, cache: None) -> Client:
        return Client(target.command)

    monkeypatch.setattr("iris.mcp.connection.Client", client)
    registry = ToolRegistry()
    manager = manager_for(tmp_path, {"a": {"command": "a"}, "b": {"command": "b"}}, registry)
    task = asyncio.create_task(manager.prepare())
    await listing.wait()
    task.cancel()
    await closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    try:
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert log == ["closing:b", "closed:b", "closing:a", "closed:a"]
    assert manager.snapshot is None and not registry.view().active_tools
