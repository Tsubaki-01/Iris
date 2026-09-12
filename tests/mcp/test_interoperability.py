"""真实 runner → executor → MCP SDK → 本地服务的五种互操作路径。"""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from iris.agents import AgentConfig
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason, ToolCallPhase
from iris.mcp.models import MCPResolvedServer
from iris.message import ToolUseBlock

from ..harness.fakes import StaticProvider, text_response, tool_response
from .conftest import serve_http
from .fixtures.servers import ServerScenario
from .fixtures.tools import AllowTools


def runner_for(
    tmp_path: Path, server: dict[str, Any], name: str = "echo", *, trust: bool = True
) -> AgentRunner:
    """从真实文件配置构造 runner，只替换 LLM provider。"""
    path = tmp_path / "interop.json"
    path.write_text(json.dumps({"mcpServers": {"fixture": server}}))
    config = AgentConfig.model_validate(
        {
            "name": "interop",
            "model": "openai/test",
            "system": "test",
            "permissions": {"workspace": str(tmp_path)},
            "mcp": {"path": str(path), "overrides": {"fixture": {"trust_annotations": trust}}},
        }
    )
    return AgentRunner.from_config(
        config,
        permission_policy=AllowTools(),
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(id="call", name=f"mcp__fixture__{name}", input={"value": 13})
            ),
            text_response(),
            tool_response(ToolUseBlock(id="next", name="mcp__fixture__echo", input={"value": 17})),
            text_response(),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True], ids=["I01-modern-stdio", "I02-legacy-stdio"])
async def test_stdio_runner_interoperability(
    tmp_path: Path, stdio_config: MCPResolvedServer, legacy: bool
) -> None:
    runner = runner_for(
        tmp_path,
        {
            "command": stdio_config.command,
            "args": [*stdio_config.args, *(["--legacy"] if legacy else [])],
            "env": stdio_config.env,
        },
    )
    try:
        result = await runner.start(AgentRunRequest(input="call", run_id="interop"))
        assert result.run.stop_reason is RunStopReason.COMPLETED
        record = runner.list_tool_calls("interop")[0]
        assert record.phase is ToolCallPhase.COMMITTED
        assert json.loads(record.result.artifact.path.read_text())["structuredContent"] == {
            "value": 13
        }
        assert runner.runtime.environment.mcp_manager.snapshot.servers[0].protocol_version == (
            "2025-11-25" if legacy else "2026-07-28"
        )
    finally:
        await runner.aclose()
    events = [
        json.loads(line)
        for line in Path(stdio_config.env["IRIS_MCP_TEST_LOG"]).read_text().splitlines()
    ]
    assert events[-1]["event"] == "closed"
    assert [event["name"] for event in events if event["event"] == "call"] == ["echo"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy,sse",
    [(False, False), (True, False), (True, True)],
    ids=["I03-modern-http", "I04-legacy-http", "I05-legacy-sse"],
)
async def test_http_runner_interoperability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool, sse: bool
) -> None:
    scenario = ServerScenario(legacy=legacy)
    clients: list[httpx2.AsyncClient] = []
    client_type = httpx2.AsyncClient

    class RecordingClient(client_type):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setattr(httpx2, "AsyncClient", RecordingClient)
    monkeypatch.setenv("IRIS_INTEROP_TOKEN", "synthetic-token")
    async with serve_http(scenario, sse=sse) as url:
        runner = runner_for(
            tmp_path,
            {
                "type": "sse" if sse else "http",
                "url": url,
                "headers": {"x-fixture-static": "static", "x-fixture-env": "${IRIS_INTEROP_TOKEN}"},
            },
        )
        try:
            result = await runner.start(AgentRunRequest(input="call", run_id="interop"))
            assert result.run.stop_reason is RunStopReason.COMPLETED
            assert runner.list_tool_calls("interop")[0].phase is ToolCallPhase.COMMITTED
            assert runner.runtime.environment.mcp_manager.snapshot.servers[0].protocol_version == (
                "2025-11-25" if legacy else "2026-07-28"
            )
        finally:
            await runner.aclose()
        assert clients and all(client.is_closed for client in clients)
        http_events = [event for event in scenario.calls if event["event"] == "http"]
        assert any(
            event["static"] == "static" and event["env"] == "synthetic-token"
            for event in http_events
        )
    assert [event["name"] for event in scenario.calls if event["event"] == "call"] == ["echo"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,trust,code",
    [
        ("invalid", True, "MCP_RESULT_INVALID"),
        ("business_error", True, "MCP_TOOL_ERROR"),
        ("rpc_error", True, "MCP_CALL_FAILED"),
        ("rpc_error", False, "TOOL_OUTCOME_UNKNOWN"),
        ("disconnect", False, "TOOL_OUTCOME_UNKNOWN"),
        ("state", True, None),
        ("interactive", True, "MCP_INTERACTION_UNSUPPORTED"),
        ("loop", True, "MCP_CONTINUATION_LIMIT"),
    ],
)
async def test_sdk_results_reach_durable_settlement(
    tmp_path: Path, stdio_config: MCPResolvedServer, name: str, trust: bool, code: str | None
) -> None:
    runner = runner_for(
        tmp_path,
        {"command": stdio_config.command, "args": list(stdio_config.args), "env": stdio_config.env},
        name,
        trust=trust,
    )
    try:
        result = await runner.start(AgentRunRequest(input="call", run_id="result"))
        record = runner.list_tool_calls("result")[0]
        if code == "TOOL_OUTCOME_UNKNOWN":
            assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
            assert record.phase is ToolCallPhase.OUTCOME_UNKNOWN
            assert await runner.recover("result") == result
        else:
            assert record.phase is ToolCallPhase.COMMITTED
            assert (record.result.error.code if record.result.error else None) == code
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_public_cancel_of_real_sdk_request_keeps_root_reusable(
    tmp_path: Path, stdio_config: MCPResolvedServer
) -> None:
    runner = runner_for(
        tmp_path,
        {"command": stdio_config.command, "args": list(stdio_config.args), "env": stdio_config.env},
        "slow",
    )
    log = Path(stdio_config.env["IRIS_MCP_TEST_LOG"])
    runner.runtime.environment.provider = StaticProvider(
        tool_response(ToolUseBlock(id="call", name="mcp__fixture__slow", input={})),
        tool_response(ToolUseBlock(id="next", name="mcp__fixture__echo", input={})),
        text_response(),
    )
    task = asyncio.create_task(runner.start(AgentRunRequest(input="slow", run_id="cancel")))
    try:
        async with asyncio.timeout(8):
            while not log.exists() or '"name": "slow"' not in log.read_text():
                await asyncio.sleep(0.01)
        result = await runner.cancel("cancel", settlement_timeout=2)
        assert result == await task
        assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
        assert runner.list_tool_calls("cancel")[0].phase is ToolCallPhase.OUTCOME_UNKNOWN
        snapshot = runner.store.load_checkpoint("cancel")
        assert (
            await runner.start(AgentRunRequest(input="echo", run_id="fresh"))
        ).run.stop_reason is RunStopReason.COMPLETED
        assert runner.store.load_checkpoint("cancel") == snapshot
        assert runner.get_result("cancel") == result
    finally:
        await task
        await runner.aclose()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(event["event"] == "cancelled" for event in events)
    assert [event["name"] for event in events if event["event"] == "call"] == ["slow", "echo"]
