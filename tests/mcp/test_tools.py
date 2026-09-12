"""MCP adapter 经现有 executor 与 runner 的执行契约。"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from mcp import types

from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisMCPCallError,
    IrisMCPOutcomeUnknownError,
)
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolCallPhase,
    ToolErrorPolicy,
)
from iris.mcp.models import MCPResolvedServer
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    BaseTool,
    DefaultPermissionPolicy,
    PermissionEffect,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)

from ..harness.fakes import StaticProvider, build_runtime, text_response, tool_response
from ..runtime.test_tool_effect_guard import MutableCancellationSignal
from .fixtures.tools import AllowTools, make_tool


@pytest.mark.asyncio
async def test_input_validated_once_before_wire_call(
    stdio_config: MCPResolvedServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool, connection = make_tool(
        stdio_config,
        schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    )
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)
    invalid = await executor.execute_one(
        ToolUseBlock(id="bad", name=tool.name, input={"x": "1"}), context
    )
    assert invalid.error.code == "VALIDATION_ERROR" and not connection.calls
    prepared = executor.prepare_many(
        [ToolUseBlock(id="ok", name=tool.name, input={"x": 1})], context
    ).calls[0]

    def revalidation_forbidden(params: dict) -> dict:
        raise AssertionError("prepared input must not be validated twice")

    monkeypatch.setattr(tool, "validate_input", revalidation_forbidden)
    result = await executor.execute_prepared(prepared, context)
    assert result.model_content == "ok"
    assert connection.calls == [("Wire.Name", {"x": 1})]


@pytest.mark.parametrize("trust", [True, False])
def test_local_trust_uses_current_base_tool_signatures(
    stdio_config: MCPResolvedServer, tmp_path: Path, trust: bool
) -> None:
    tool, _ = make_tool(stdio_config, trust=trust)
    assert tool.is_read_only({}) is trust
    assert tool.is_concurrency_safe({}) is False
    effect = (
        DefaultPermissionPolicy()
        .check(tool, {}, ToolExecutionContext(workspace_root=tmp_path))
        .effect
    )
    assert effect is (PermissionEffect.ALLOW if trust else PermissionEffect.REQUIRE_HUMAN)


@pytest.mark.asyncio
async def test_unknown_bypasses_both_executor_error_handlers(
    stdio_config: MCPResolvedServer, tmp_path: Path
) -> None:
    class SwallowErrors(ToolMiddleware):
        async def on_error(
            self, tool: BaseTool, error: Exception, context: ToolExecutionContext
        ) -> ToolResult | None:
            """任何调用都表明 unknown 误入普通异常 hook。"""
            raise AssertionError("unknown must bypass middleware conversion")

    tool, connection = make_tool(stdio_config, trust=False, error=IrisMCPCallError("no result"))
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry, permission_policy=AllowTools(), middleware=[SwallowErrors()])
    with pytest.raises(IrisMCPOutcomeUnknownError):
        await executor.execute_one(
            ToolUseBlock(id="call", name=tool.name, input={}),
            ToolExecutionContext(workspace_root=tmp_path),
        )
    assert len(connection.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("trust", "deadline"), [(False, False), (False, True), (True, True)])
async def test_runtime_settles_claim_as_unknown_without_replay(
    stdio_config: MCPResolvedServer, tmp_path: Path, trust: bool, deadline: bool
) -> None:
    tool, connection = make_tool(
        stdio_config,
        trust=trust,
        error=TimeoutError() if deadline else IrisMCPCallError("no result"),
    )
    registry = ToolRegistry()
    registry.register(tool)
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            permission_policy=AllowTools(),
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="call", name=tool.name, input={}))
            ),
        ),
        store=store,
    )
    result = await runner.start(AgentRunRequest(input="call", run_id="unknown"))
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert store.list_tool_calls("unknown")[0].phase is ToolCallPhase.OUTCOME_UNKNOWN
    assert await runner.recover("unknown") == result
    assert len(connection.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [True, False])
async def test_trusted_sdk_error_obeys_tool_error_policy(
    stdio_config: MCPResolvedServer, tmp_path: Path, stop: bool
) -> None:
    tool, connection = make_tool(stdio_config, error=IrisMCPCallError("no result"))
    registry = ToolRegistry()
    registry.register(tool)
    store = InMemoryLifecycleStore()
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="call", name=tool.name, input={})), text_response()
    )
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider), store=store
    )
    result = await runner.start(
        AgentRunRequest(input="call", run_id="known"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(
                tool_error_policy=ToolErrorPolicy.STOP if stop else ToolErrorPolicy.RETURN_TO_MODEL
            )
        ),
    )
    assert result.run.stop_reason is (RunStopReason.FAILED if stop else RunStopReason.COMPLETED)
    record = store.list_tool_calls("known")[0]
    assert record.phase is ToolCallPhase.COMMITTED
    assert record.result.error.code == "MCP_CALL_FAILED"
    assert len(connection.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("returns_result", [True, False])
async def test_executor_signal_cancels_mcp_body_and_waits_for_cleanup(
    stdio_config: MCPResolvedServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returns_result: bool,
) -> None:
    tool, connection = make_tool(stdio_config)
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def waiting(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            if returns_result:
                return connection.result
            raise

    monkeypatch.setattr(connection, "call_tool", waiting)
    signal = MutableCancellationSignal()
    registry = ToolRegistry()
    registry.register(tool)
    task = asyncio.create_task(
        ToolExecutor(registry).execute_one(
            ToolUseBlock(id="call", name=tool.name, input={}),
            ToolExecutionContext(workspace_root=tmp_path, cancellation=signal),
        )
    )
    await entered.wait()
    signal.requested = True
    await asyncio.wait_for(cleaning.wait(), 1)
    assert not task.done()
    release.set()
    if returns_result:
        assert (await task).model_content == "ok"
    else:
        with pytest.raises(IrisCancellationRequestedError):
            await task
