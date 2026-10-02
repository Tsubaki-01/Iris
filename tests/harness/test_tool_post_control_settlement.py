"""工具已知结果后的取消与清理失败保持原有结算意图。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.exceptions import IrisCommandCleanupError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import (
    BaseTool,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)
from iris.tools.middleware import ToolCall, ToolNext

from .fakes import FrozenClock, StaticProvider, build_runtime, text_response, tool_response
from .test_command_settlement import ControlledService, bind_service


class KnownTool(BaseTool):
    """保存当前调用槽供故障注入，真实 body 只执行一次。"""

    def __init__(self) -> None:
        self.definition = ToolDefinition(
            name="known",
            description="已知结果工具",
            input_schema={"type": "object"},
            capabilities={ToolCapability.READ},
        )
        self.calls = 0
        self.context: ToolExecutionContext | None = None

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """产生已知结果，不由中间件制造 body 事实。"""
        self.calls += 1
        self.context = context
        return ToolResult(
            tool_use_id=context.call_id, tool_name=self.name, content=[TextBlock(text="known")]
        )


class PostCleanupFailure(ToolMiddleware):
    """在真实 body 之后等待取消，并模拟停止排空失败。"""

    def __init__(self, tool: KnownTool, service: ControlledService) -> None:
        self.tool = tool
        self.service = service
        self.entered = asyncio.Event()
        self.error = IrisCommandCleanupError("post cleanup pending")

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        """把取消和独立命令清理事实同时交回框架。"""
        await call_next()
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert self.tool.context is not None
            self.tool.context.command_stop_slot.receipt = self.service.receipt
            self.tool.context.command_stop_slot.cleanup_error = self.error
            raise
        raise AssertionError("post must be interrupted")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cancel", "deadline", "sdk"])
@pytest.mark.parametrize("sqlite", [False, True])
async def test_known_result_control_cleanup_retry_preserves_intent(
    tmp_path: Path, source: str, sqlite: bool
) -> None:
    """真实 Runner/Store 路径先保存结果，再报告并重试原意图的清理。"""
    clock = FrozenClock()
    tool = KnownTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="call", name="known", input={})), text_response("resumed")
    )
    runtime = build_runtime(tmp_path, registry=registry, provider=provider)
    store = SQLiteStore(tmp_path / "lifecycle.db") if sqlite else InMemoryLifecycleStore()
    runner = AgentRunner(runtime=runtime, store=store, clock=clock)
    service = ControlledService()
    # 首次错误由后处理注入；显式重试时命令服务可以完成排空。
    service.release()
    bind_service(runner, service)
    middleware = PostCleanupFailure(tool, service)
    runtime.environment.tool_bridge.tool_executor.middleware = [middleware]
    limits = RunLimits(deadline_at=clock.now() + timedelta(seconds=30))
    options = AgentRunOptions(limits=limits) if source == "deadline" else AgentRunOptions()
    task = asyncio.create_task(
        runner.start(AgentRunRequest(input="go", run_id="post"), options=options)
    )
    await asyncio.wait_for(middleware.entered.wait(), 2)
    if source == "sdk":
        task.cancel()
    elif source == "deadline":
        clock.advance(seconds=31)
        await runner._command_deadline_due("post")
    else:
        runner.request_cancel("post")
    with pytest.raises(IrisCommandCleanupError, match="post cleanup pending"):
        await asyncio.wait_for(task, 2)

    assert runner.get_run("post").phase is RunPhase.ACTIVE
    pending = runner._command_lifecycle.pending["post"]
    expected = {
        "cancel": RunStopReason.CANCELLED,
        "deadline": RunStopReason.DEADLINE_EXCEEDED,
        "sdk": None,
    }[source]
    assert pending.stop_reason is expected
    assert pending.receipt is service.receipt
    assert store.load_tool_call("post", "call").result.model_content == "known"
    assert tool.calls == 1
    assert len(provider.requests) == 1

    result = await runner._command_lifecycle.join(pending)
    assert "post" not in runner._command_lifecycle.pending
    if expected is None:
        assert result is None
        assert runner.get_run("post").phase is RunPhase.ACTIVE
        recovered = await runner.recover(
            "post", expected_activation_id=runner.get_run("post").current_activation_id
        )
        assert recovered.run.stop_reason is RunStopReason.COMPLETED
    else:
        assert result is not None
        assert result.run.stop_reason is expected
        assert (await runner.recover("post")) == result
    assert tool.calls == 1
    assert len(provider.requests) == (2 if source == "sdk" else 1)
    assert store.load_tool_call("post", "call").result.model_content == "known"
    await runner.aclose()
