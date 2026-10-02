"""工具已知结果后的取消与清理失败保持原有结算意图。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.command import CommandStopReceipt, CommandStopSlot
from iris.exceptions import IrisCommandCleanupError, IrisToolOutcomeUnknownError
from iris.harness import AgentRunner
from iris.hooks import HookEvent, HookRegistration, ToolAfterResult
from iris.hooks._dispatch_types import CommandHookRegistration, HookControl, HookInvocationOutcome
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import (
    BaseTool,
    CancellationSignal,
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


class AfterCleanupFailure:
    """命令 after 在中断时交接真实 body 以外的资源清理事实。"""

    def __init__(self, service: ControlledService) -> None:
        self.service = service
        self.entered = asyncio.Event()
        self.error = IrisCommandCleanupError("post cleanup pending")

    async def __call__(
        self, event: HookEvent, *, cancellation: CancellationSignal | None
    ) -> HookInvocationOutcome:
        """模拟 adapter 对停止失败的类型化交接，不改写已知工具结果。"""
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            return HookInvocationOutcome(
                control=HookControl(
                    "task_cancelled",
                    error,
                    stop_slot=CommandStopSlot(
                        receipt=self.service.receipt, cleanup_error=self.error
                    ),
                    call_id="hook-cleanup",
                )
            )
        raise AssertionError("post must be interrupted")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cancel", "deadline", "sdk"])
@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize("extension", ["middleware", "after"])
async def test_known_result_control_cleanup_retry_preserves_intent(
    tmp_path: Path, source: str, sqlite: bool, extension: str
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
    executor = runtime.environment.tool_bridge.tool_executor
    remaining_calls: list[str] = []
    if extension == "middleware":
        post: PostCleanupFailure | AfterCleanupFailure = PostCleanupFailure(tool, service)
        executor.middleware = [post]
    else:
        post = AfterCleanupFailure(service)

        async def feedback(event: HookEvent) -> ToolAfterResult:
            """取消之前已收集的反馈必须随真实工具结果提交。"""
            return ToolAfterResult(feedback="first feedback")

        async def unexpected(event: HookEvent) -> None:
            remaining_calls.append(event.event)

        dispatcher = HookDispatcher(
            [
                HookRegistration(event="tool.after", name="feedback", handler=feedback),
                CommandHookRegistration("tool.after", "cleanup", post),
                HookRegistration(event="tool.after", name="unexpected", handler=unexpected),
            ]
        )
        runtime.environment = replace(runtime.environment, hook_dispatcher=dispatcher)
    limits = RunLimits(deadline_at=clock.now() + timedelta(seconds=30))
    options = AgentRunOptions(limits=limits) if source == "deadline" else AgentRunOptions()
    task = asyncio.create_task(
        runner.start(AgentRunRequest(input="go", run_id="post"), options=options)
    )
    await asyncio.wait_for(post.entered.wait(), 2)
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
    saved = store.load_tool_call("post", "call").result
    assert saved.content == [TextBlock(text="known")]
    assert saved.hook_feedback == (("first feedback",) if extension == "after" else ())
    assert remaining_calls == []
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
    assert store.load_tool_call("post", "call").result == saved
    await runner.aclose()


class HookStopService(ControlledService):
    """已有收据和主动停止使用同一个可控排空失败。"""

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        """模拟 after 未知脚本结果所需的命令资源收口。"""
        if self.fail:
            raise IrisCommandCleanupError("hook drain pending")
        await super().wait_drained(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_receipt", [False, True])
@pytest.mark.parametrize("drain_fails", [False, True])
async def test_after_unknown_keeps_known_body_and_does_not_replay(
    tmp_path: Path, has_receipt: bool, drain_fails: bool
) -> None:
    """附加脚本 unknown 只收口其资源；已知 body 和先前反馈始终可回读。"""
    tool = KnownTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="call", name="known", input={})), text_response("next")
    )
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runtime = build_runtime(tmp_path, registry=registry, provider=provider)
    runner = AgentRunner(runtime=runtime, store=store)
    service = HookStopService()
    service.release()
    service.fail = drain_fails
    bind_service(runner, service)
    seen: list[str] = []

    async def first(event: HookEvent) -> ToolAfterResult:
        seen.append("first")
        return ToolAfterResult(feedback="first feedback")

    async def unknown(
        event: HookEvent, *, cancellation: CancellationSignal | None
    ) -> HookInvocationOutcome:
        seen.append("unknown")
        raise IrisToolOutcomeUnknownError(
            "lost script result", stop_receipt=service.receipt if has_receipt else None
        )

    async def later(event: HookEvent) -> None:
        seen.append("later")

    runtime.environment = replace(
        runtime.environment,
        hook_dispatcher=HookDispatcher(
            [
                HookRegistration(event="tool.after", name="first", handler=first),
                CommandHookRegistration("tool.after", "unknown", unknown),
                HookRegistration(event="tool.after", name="later", handler=later),
            ]
        ),
    )
    request = AgentRunRequest(input="go", run_id="unknown-after")
    try:
        if drain_fails:
            with pytest.raises(IrisCommandCleanupError):
                await runner.start(request)
            assert runner.get_run("unknown-after").phase is RunPhase.ACTIVE
            pending = runner._command_lifecycle.pending["unknown-after"]
            assert pending.stop_reason is RunStopReason.FAILED
            assert len(provider.requests) == 1
            service.fail = False
            result = await runner.recover("unknown-after")
            assert result.run.stop_reason is RunStopReason.FAILED
        else:
            result = await runner.start(request)
            assert result.run.stop_reason is RunStopReason.COMPLETED
            assert len(provider.requests) == 2
        assert seen == ["first", "unknown"]
        assert tool.calls == 1
        record = SQLiteStore(store.path).load_tool_call("unknown-after", "call")
        assert record is not None and record.result is not None
        assert not record.result.is_error
        assert record.result.content == [TextBlock(text="known")]
        assert record.result.hook_feedback == ("first feedback",)
        assert (await runner.recover("unknown-after")) == result
        assert seen == ["first", "unknown"] and tool.calls == 1
    finally:
        service.fail = False
        await runner.aclose()
