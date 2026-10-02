"""已知工具结果后的控制必须先提交，再按原停止意图收尾。"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    FakeRuntimeSteeringPort,
    MutableCancellationSignal,
    start_activation,
)
from pydantic import BaseModel

from iris.command import CommandStopReceipt
from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisCommandCleanupError,
    IrisToolOutcomeUnknownError,
)
from iris.lifecycle import RuntimeExecutionOptions
from iris.message import TextBlock, ToolUseBlock
from iris.runtime import RuntimeActivationOutcome
from iris.runtime import runtime as runtime_module
from iris.tools import (
    BaseTool,
    ToolCall,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
)
from iris.tools.base import ToolTimeoutOwner

from .test_execute import _runtime, _tool_batch_response


class PostAction(ToolMiddleware):
    """在真实 body 已返回后执行当前测试的动作。"""

    def __init__(self, action: Callable[[ToolCall], Awaitable[None]]) -> None:
        self.action = action

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        result = await call_next()
        await self.action(call)
        return result


class KnownTool(BaseTool):
    """记录每个 call 的实际执行，并可附带待交接的命令清理事实。"""

    definition = ToolDefinition(
        name="known",
        description="返回确定结果",
        input_schema={"type": "object"},
        capabilities={ToolCapability.READ},
    )

    def __init__(self, cleanup: IrisCommandCleanupError | None = None) -> None:
        self.calls: list[str] = []
        self.cleanup = cleanup
        self.receipt = CommandStopReceipt("service", "stop")

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        self.calls.append(context.call_id)
        if self.cleanup is not None:
            context.command_stop_slot.cleanup_error = self.cleanup
            context.command_stop_slot.receipt = self.receipt
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[TextBlock(text=f"body-{context.call_id}")],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["cancel", "deadline", "timeout", "cleanup"])
async def test_known_result_keeps_control_intent_with_cleanup(tmp_path: Path, control: str) -> None:
    cleanup = IrisCommandCleanupError("cleanup pending")
    tool = KnownTool(cleanup)
    registry = ToolRegistry()
    registry.register(tool)
    provider = FakeProvider([_tool_batch_response([ToolUseBlock(id="a", name="known", input={})])])
    activation = start_activation(
        options=RuntimeExecutionOptions(tool_timeout_seconds=0.03 if control == "timeout" else None)
    )
    commits = FakeRuntimeCommitPort(activation)

    async def post(call: ToolCall) -> None:
        if control == "deadline":
            commits.deadline = 0
            raise IrisCancellationRequestedError("deadline interrupted post")
        if control == "cancel":
            raise IrisCancellationRequestedError("post cancelled")
        if control == "timeout":
            await asyncio.Event().wait()

    runtime = _runtime(
        provider=provider, tmp_path=tmp_path, registry=registry, middleware=[PostAction(post)]
    )
    steering = FakeRuntimeSteeringPort([])
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal(), steering=steering
    )

    assert tool.calls == ["a"], (result.error, [item.result for item in commits.tool_commits])
    assert (
        result.outcome
        is {
            "cancel": RuntimeActivationOutcome.CANCELLED,
            "deadline": RuntimeActivationOutcome.DEADLINE_EXCEEDED,
            "timeout": RuntimeActivationOutcome.FAILED,
            "cleanup": RuntimeActivationOutcome.FAILED,
        }[control]
    )
    assert result.cleanup_error is cleanup
    assert result.stop_receipt is tool.receipt
    assert result.stop_call_id == "a"
    assert "cleanup_error" not in result.model_dump(mode="json")
    assert [commit.result.model_content for commit in commits.tool_commits] == ["body-a"]
    assert tool.calls == ["a"]
    assert len(provider.requests) == 1
    assert not steering.events
    assert runtime.environment.command_stop_slots[(activation.run_id, "a")].cleanup_error is cleanup


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_cancel_or_timeout_during_post_commits_body_once(
    tmp_path: Path, timeout: bool
) -> None:
    entered = asyncio.Event()

    async def post(call: ToolCall) -> None:
        entered.set()
        await asyncio.Event().wait()

    tool = KnownTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = FakeProvider([_tool_batch_response([ToolUseBlock(id="a", name="known", input={})])])
    runtime = _runtime(
        provider=provider, tmp_path=tmp_path, registry=registry, middleware=[PostAction(post)]
    )
    activation = start_activation(
        options=RuntimeExecutionOptions(tool_timeout_seconds=0.03 if timeout else None)
    )
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort([])
    execution = asyncio.create_task(
        runtime.execute(
            activation, commits=commits, cancellation=MutableCancellationSignal(), steering=steering
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    if timeout:
        result = await asyncio.wait_for(execution, 1)
        assert result.outcome is RuntimeActivationOutcome.FAILED
        assert result.error is not None and result.error.code == "TOOL_TIMEOUT"
    else:
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
    assert [commit.result.model_content for commit in commits.tool_commits] == ["body-a"]
    assert tool.calls == ["a"]
    assert not steering.events
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_parallel_post_control_drains_known_siblings_without_sdk_cancellation(
    tmp_path: Path,
) -> None:
    entered: set[str] = set()
    together = asyncio.Event()
    drained: set[str] = set()

    async def post(call: ToolCall) -> None:
        entered.add(call.tool_use_id)
        if len(entered) == 3:
            together.set()
        await together.wait()
        if call.tool_use_id == "b":
            raise IrisCancellationRequestedError("first real cause")
        try:
            await asyncio.Event().wait()
        finally:
            drained.add(call.tool_use_id)

    tool = KnownTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = FakeProvider(
        [_tool_batch_response([ToolUseBlock(id=name, name="known", input={}) for name in "abc"])]
    )
    runtime = _runtime(
        provider=provider, tmp_path=tmp_path, registry=registry, middleware=[PostAction(post)]
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort([])
    result = await asyncio.wait_for(
        runtime.execute(
            activation, commits=commits, cancellation=MutableCancellationSignal(), steering=steering
        ),
        1,
    )
    assert result.outcome is RuntimeActivationOutcome.CANCELLED
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == list("abc")
    assert drained == {"a", "c"}
    assert not steering.events
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_pending", [False, True])
async def test_parallel_unknown_body_does_not_commit_later_known_result(
    tmp_path: Path, cleanup_pending: bool
) -> None:
    later_done = asyncio.Event()
    receipt = CommandStopReceipt("service", "unknown-stop")
    cleanup = IrisCommandCleanupError("unknown body cleanup pending")

    class UnknownRead(KnownTool):
        """首条未取得结果，后条已返回；控制异常直接来自工具协议。"""

        async def arun(
            self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            if context.call_id == "a1":
                await later_done.wait()
                if cleanup_pending:
                    raise cleanup
                raise IrisToolOutcomeUnknownError("first body is unknown", stop_receipt=receipt)
            later_done.set()
            return await super().arun(params, context)

    registry = ToolRegistry()
    registry.register(UnknownRead())
    provider = FakeProvider(
        [_tool_batch_response([ToolUseBlock(id=f"a{i}", name="known", input={}) for i in (1, 2)])]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert not commits.tool_commits, [item.result.error for item in commits.tool_commits]
    assert result.outcome is RuntimeActivationOutcome.OUTCOME_UNKNOWN
    assert result.cleanup_error is (cleanup if cleanup_pending else None)
    assert result.stop_receipt is (None if cleanup_pending else receipt)
    assert result.stop_call_id == "a1"
    assert result.cursor.next_tool_index == 0
    assert set(commits.claims) == {"a1", "a2"}
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_parallel_cleanup_on_known_result_stops_and_keeps_slots(tmp_path: Path) -> None:
    cleanup = IrisCommandCleanupError("known results still need cleanup")
    tool = KnownTool(cleanup)
    registry = ToolRegistry()
    registry.register(tool)
    provider = FakeProvider(
        [_tool_batch_response([ToolUseBlock(id=name, name="known", input={}) for name in "ab"])]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort([])
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal(), steering=steering
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.cleanup_error is cleanup
    assert result.stop_receipt is tool.receipt
    assert result.stop_call_id == "a"
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == list("ab")
    assert set(runtime.environment.command_stop_slots) == {
        (activation.run_id, name) for name in "ab"
    }
    assert not steering.events
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_parallel_same_completion_batch_uses_first_ordinal_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    together = asyncio.Event()
    completed = 0
    execute = runtime_module._execute_tool_with_timeout

    async def complete_together(
        operation: Awaitable[ToolResult], timeout: float | None, owner: ToolTimeoutOwner
    ) -> runtime_module._ToolCompletion:
        nonlocal completed
        result = await execute(operation, timeout, owner)
        completed += 1
        if completed == 2:
            together.set()
        await together.wait()
        return result

    monkeypatch.setattr(runtime_module, "_execute_tool_with_timeout", complete_together)

    async def post(call: ToolCall) -> None:
        if call.tool_use_id == "a":
            raise IrisCancellationRequestedError("ordinal a")
        raise asyncio.CancelledError("ordinal b")

    registry = ToolRegistry()
    registry.register(KnownTool())
    provider = FakeProvider(
        [_tool_batch_response([ToolUseBlock(id=name, name="known", input={}) for name in "ab"])]
    )
    runtime = _runtime(
        provider=provider, tmp_path=tmp_path, registry=registry, middleware=[PostAction(post)]
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.CANCELLED
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == list("ab")
    assert len(provider.requests) == 1
