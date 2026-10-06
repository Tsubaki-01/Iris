"""工具包装链的单次 continuation、真实结果和控制交接。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from iris.command.models import CommandStatus, CommandStopReceipt, CommandStopSlot
from iris.exceptions import IrisCancellationRequestedError, IrisToolExecutionError
from iris.message import TextBlock, ToolUseBlock
from iris.observability.service import Observability
from iris.tools import (
    BaseTool,
    CircuitBreaker,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
)
from iris.tools._execution_control import ToolExecutionControlSlot


def _result(text: str) -> ToolResult:
    return ToolResult(tool_use_id="foreign", tool_name="foreign", content=[TextBlock(text=text)])


class _Wrapper(ToolMiddleware):
    def __init__(self, callback: Callable[[ToolCall, ToolNext], Awaitable[ToolResult]]) -> None:
        self.callback = callback

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        return await self.callback(call, call_next)


class _Body(BaseTool):
    definition = ToolDefinition(
        name="body", description="测试工具", input_schema={"type": "object", "properties": {}}
    )

    def __init__(self, callback: Callable[[ToolExecutionContext], Awaitable[ToolResult]]) -> None:
        self.callback = callback

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        return await self.callback(context)


def _executor(
    body: Callable[[ToolExecutionContext], Awaitable[ToolResult]],
    *wrappers: Callable[[ToolCall, ToolNext], Awaitable[ToolResult]],
    breaker: CircuitBreaker | None = None,
    observability: Observability | None = None,
) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(_Body(body))
    return ToolExecutor(
        registry,
        middleware=[_Wrapper(w) for w in wrappers],
        circuit_breaker=breaker,
        observability=observability,
    )


async def _execute(executor: ToolExecutor, context: ToolExecutionContext) -> ToolResult:
    return await executor.execute_one(ToolUseBlock(id="current", name="body", input={}), context)


@pytest.mark.asyncio
async def test_onion_order_and_identity(tmp_path: Path) -> None:
    order: list[str] = []

    async def body(context: ToolExecutionContext) -> ToolResult:
        order.append("body")
        return _result("body")

    def layer(name: str) -> Callable[[ToolCall, ToolNext], Awaitable[ToolResult]]:
        async def wrap(call: ToolCall, call_next: ToolNext) -> ToolResult:
            order.append(f"{name} before")
            result = await call_next()
            order.append(f"{name} after")
            return result

        return wrap

    result = await _execute(
        _executor(body, layer("A"), layer("B")), ToolExecutionContext(workspace_root=tmp_path)
    )
    assert order == ["A before", "B before", "body", "B after", "A after"]
    assert (result.tool_use_id, result.tool_name) == ("current", "body")


@pytest.mark.asyncio
async def test_short_circuit_has_no_body_or_breaker_count(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability

    async def body(context: ToolExecutionContext) -> ToolResult:
        pytest.fail("短路不进入body")

    async def short(call: ToolCall, call_next: ToolNext) -> ToolResult:
        return _result("cached")

    breaker = CircuitBreaker(failure_threshold=1)
    result = await _execute(
        _executor(body, short, breaker=breaker, observability=observation),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.model_content == "cached"
    assert breaker._states == {}
    [span] = exporter.get_finished_spans()
    assert span.name == "execute_tool body"
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes["gen_ai.tool.call.id"] == "current"
    assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {
        "parts": [{"type": "text", "content": "cached"}]
    }


@pytest.mark.asyncio
async def test_duplicate_and_late_continuation_cannot_reexecute(tmp_path: Path) -> None:
    calls = 0
    saved: ToolNext | None = None

    async def body(context: ToolExecutionContext) -> ToolResult:
        nonlocal calls
        calls += 1
        return _result("first")

    async def twice(call: ToolCall, call_next: ToolNext) -> ToolResult:
        nonlocal saved
        saved = call_next
        await call_next()
        return await call_next()

    result = await _execute(_executor(body, twice), ToolExecutionContext(workspace_root=tmp_path))
    assert result.model_content == "first"
    assert saved is not None
    with pytest.raises(IrisToolExecutionError):
        await saved()
    assert calls == 1


@pytest.mark.asyncio
async def test_started_continuation_is_drained_and_early_result_ignored(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    pending: asyncio.Task[ToolResult] | None = None

    async def body(context: ToolExecutionContext) -> ToolResult:
        entered.set()
        await release.wait()
        return _result("real")

    async def early(call: ToolCall, call_next: ToolNext) -> ToolResult:
        nonlocal pending
        pending = asyncio.create_task(call_next())
        await entered.wait()
        return _result("fake")

    task = asyncio.create_task(
        _execute(_executor(body, early), ToolExecutionContext(workspace_root=tmp_path))
    )
    await entered.wait()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    result = await task
    assert result.model_content == "real"
    assert pending is not None and pending.done()


@pytest.mark.asyncio
async def test_early_continuation_drain_forwards_first_cancellation(tmp_path: Path) -> None:
    entered = asyncio.Event()
    returned = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_interrupted = False

    async def body(context: ToolExecutionContext) -> ToolResult:
        nonlocal cleanup_interrupted
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            try:
                await release_cleanup.wait()
            except asyncio.CancelledError:
                cleanup_interrupted = True
                raise
        return _result("unreachable")

    async def early(call: ToolCall, call_next: ToolNext) -> ToolResult:
        asyncio.create_task(call_next())
        await entered.wait()
        returned.set()
        return _result("fake")

    task = asyncio.create_task(
        _execute(_executor(body, early), ToolExecutionContext(workspace_root=tmp_path))
    )
    await returned.wait()
    await asyncio.sleep(0)
    task.cancel()
    try:
        await asyncio.wait_for(cleanup_entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not cleanup_interrupted
    finally:
        release_cleanup.set()
        outcome = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome[0], asyncio.CancelledError)


@pytest.mark.asyncio
async def test_duplicate_continuation_cannot_hide_violation(tmp_path: Path) -> None:
    async def body(context: ToolExecutionContext) -> ToolResult:
        return _result("first")

    async def duplicate(call: ToolCall, call_next: ToolNext) -> ToolResult:
        await call_next()
        try:
            await call_next()
        except IrisToolExecutionError:
            return _result("fake")
        pytest.fail("第二次continuation应失败")

    result = await _execute(
        _executor(body, duplicate), ToolExecutionContext(workspace_root=tmp_path)
    )
    assert result.model_content == "first"


@pytest.mark.asyncio
async def test_body_failure_recovery_keeps_real_breaker_failure(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability

    async def body(context: ToolExecutionContext) -> ToolResult:
        raise IrisToolExecutionError("body failed")

    async def recover(call: ToolCall, call_next: ToolNext) -> ToolResult:
        try:
            return await call_next()
        except IrisToolExecutionError:
            return _result("recovered")

    breaker = CircuitBreaker(failure_threshold=1)
    executor = _executor(body, recover, breaker=breaker, observability=observation)
    context = ToolExecutionContext(workspace_root=tmp_path)
    assert (await _execute(executor, context)).model_content == "recovered"
    assert (await _execute(executor, context)).error.code == "CIRCUIT_OPEN"
    recovered, blocked = exporter.get_finished_spans()
    assert recovered.status.status_code is StatusCode.UNSET
    assert blocked.status.status_code is StatusCode.ERROR
    assert "recovered" in recovered.attributes["gen_ai.tool.call.result"]
    assert "CIRCUIT_OPEN" in blocked.attributes["gen_ai.tool.call.result"]


@pytest.mark.asyncio
async def test_post_failure_keeps_downstream_result(tmp_path: Path) -> None:
    async def body(context: ToolExecutionContext) -> ToolResult:
        return _result("real")

    async def fail(call: ToolCall, call_next: ToolNext) -> ToolResult:
        await call_next()
        _result("replacement never returned")
        raise RuntimeError("post failed")

    result = await _execute(_executor(body, fail), ToolExecutionContext(workspace_root=tmp_path))
    assert result.model_content == "real"


@pytest.mark.asyncio
@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("owned", [False, True])
async def test_control_cannot_be_swallowed(tmp_path: Path, known: bool, owned: bool) -> None:
    original = IrisCancellationRequestedError("stop")

    async def body(context: ToolExecutionContext) -> ToolResult:
        if not known:
            raise original
        return _result("known")

    async def interrupt(call: ToolCall, call_next: ToolNext) -> ToolResult:
        await call_next()
        raise original

    async def swallow(call: ToolCall, call_next: ToolNext) -> ToolResult:
        try:
            return await call_next()
        except Exception:
            return _result("fake")

    slot = ToolExecutionControlSlot() if owned else None
    context = ToolExecutionContext(workspace_root=tmp_path, execution_control=slot)
    if known and owned:
        result = await _execute(_executor(body, swallow, interrupt), context)
        assert result.model_content == "known"
        assert slot.control.error is original
    else:
        with pytest.raises(IrisCancellationRequestedError) as caught:
            await _execute(_executor(body, swallow, interrupt), context)
        assert caught.value is original


class _Signal:
    requested = False

    def raise_if_requested(self) -> None:
        if self.requested:
            raise IrisCancellationRequestedError("signal")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["signal", "task"])
async def test_post_wait_cancellation_returns_known_result(tmp_path: Path, source: str) -> None:
    signal = _Signal()
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def body(context: ToolExecutionContext) -> ToolResult:
        return _result("known")

    async def wait(call: ToolCall, call_next: ToolNext) -> ToolResult:
        await call_next()
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return _result("unreachable")

    slot = ToolExecutionControlSlot()
    context = ToolExecutionContext(
        workspace_root=tmp_path, cancellation=signal, execution_control=slot
    )
    task = asyncio.create_task(_execute(_executor(body, wait), context))
    await entered.wait()
    if source == "signal":
        signal.requested = True
    else:
        task.cancel()
    result = await asyncio.wait_for(task, 1)
    assert cleaned.is_set()
    assert result.model_content == "known"
    error_type = IrisCancellationRequestedError if source == "signal" else asyncio.CancelledError
    assert isinstance(slot.control.error, error_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_batch_calls_get_separate_slots(tmp_path: Path, parallel: bool) -> None:
    contexts: list[ToolExecutionContext] = []

    async def body(context: ToolExecutionContext) -> ToolResult:
        contexts.append(context)
        assert context.command_stop_slot.status is None
        assert context.execution_control is None
        context.command_stop_slot.status = CommandStatus.ENVIRONMENT_INTERRUPTED
        context.command_stop_slot.receipt = CommandStopReceipt("service", context.call_id)
        return _result(context.call_id)

    executor = _executor(body)
    executor.registry.get("body").is_concurrency_safe = lambda params: parallel
    base = ToolExecutionContext(
        workspace_root=tmp_path, execution_control=ToolExecutionControlSlot()
    )
    await executor.execute_many(
        [ToolUseBlock(id=str(i), name="body", input={}) for i in range(2)], base
    )
    assert contexts[0].command_stop_slot is not contexts[1].command_stop_slot
    assert base.command_stop_slot.status is None
    assert base.model_copy(deep=True).execution_control is base.execution_control


@pytest.mark.asyncio
async def test_parallel_spans_end_when_each_tool_finishes(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability
    slow_started, release_slow, fast_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def body(context: ToolExecutionContext) -> ToolResult:
        if context.call_id == "slow":
            slow_started.set()
            await release_slow.wait()
        else:
            await slow_started.wait()
            fast_finished.set()
        return _result(context.call_id)

    executor = _executor(body, observability=observation)
    executor.registry.get("body").is_concurrency_safe = lambda params: True
    with observation.scope("parent") as parent:
        task = asyncio.create_task(
            executor.execute_many(
                [ToolUseBlock(id=name, name="body", input={}) for name in ("slow", "fast")],
                ToolExecutionContext(workspace_root=tmp_path),
            )
        )
        try:
            await asyncio.wait_for(fast_finished.wait(), 5)
            # final artifact 归一化沿既有异步IO完成，而不是只等 body 返回。
            async with asyncio.timeout(5):
                while not exporter.get_finished_spans():
                    await asyncio.sleep(0)
            [fast] = exporter.get_finished_spans()
            assert fast.attributes["gen_ai.tool.call.id"] == "fast"
            assert not task.done()
        finally:
            release_slow.set()
            results = await task
    fast, slow, parent_span = exporter.get_finished_spans()
    assert [result.tool_use_id for result in results] == ["slow", "fast"]
    assert fast.end_time <= slow.end_time
    assert fast.parent == slow.parent == parent.get_span_context()
    assert parent_span.context == parent.get_span_context()


def test_middleware_requires_current_abstract_method() -> None:
    class OldMiddleware(ToolMiddleware):
        pass

    with pytest.raises(TypeError, match="wrap_tool_call"):
        OldMiddleware()


@pytest.mark.asyncio
async def test_call_arguments_snapshot_cannot_change_body_input(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability

    class ArgumentTool(_Body):
        async def arun(
            self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            assert params == {"nested": {"value": "original"}}
            return _result("original")

    async def unused(context: ToolExecutionContext) -> ToolResult:
        pytest.fail("fixture override应被调用")

    async def mutate(call: ToolCall, call_next: ToolNext) -> ToolResult:
        call.arguments["nested"]["value"] = "replacement"
        return await call_next()

    registry = ToolRegistry()
    registry.register(ArgumentTool(unused))
    executor = ToolExecutor(registry, middleware=[_Wrapper(mutate)], observability=observation)
    result = await executor.execute_one(
        ToolUseBlock(id="current", name="body", input={"nested": {"value": "original"}}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.model_content == "original"
    [span] = exporter.get_finished_spans()
    assert json.loads(span.attributes["gen_ai.tool.call.arguments"]) == {
        "nested": {"value": "original"}
    }


def test_command_success_does_not_erase_pending_receipt() -> None:
    receipt = CommandStopReceipt("service", "first")
    slot = CommandStopSlot()
    slot.record(status=CommandStatus.ENVIRONMENT_INTERRUPTED, receipt=receipt)
    slot.record(receipt=None)
    assert slot.receipt is receipt
    assert slot.status is CommandStatus.ENVIRONMENT_INTERRUPTED
