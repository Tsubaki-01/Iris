"""AgentRunner durable cancellation 与 settlement observation 测试。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from iris.exceptions import IrisRunObservationTimeoutError
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunRequest,
    RunEventKind,
    RunPhase,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    BaseTool,
    CallableExecutionMode,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)

from .fakes import (
    BlockingProvider,
    StaticProvider,
    build_runtime,
    tool_batch_response,
    tool_response,
)


@pytest.mark.asyncio
async def test_active_cancel_persists_first_reason_and_interrupts_provider(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="等待", run_id="run-cancel-provider"))
    )
    await provider.started.wait()

    result = await runner.cancel(
        "run-cancel-provider",
        reason="用户停止",
        settlement_timeout=1,
    )
    repeated = runner.request_cancel("run-cancel-provider", reason="后来原因")

    assert result == await running
    assert result.run.stop_reason is RunStopReason.CANCELLED
    assert repeated.cancellation_reason == "用户停止"
    assert [event.kind for event in store.list_events("run-cancel-provider")].count(
        RunEventKind.CANCELLATION_REQUESTED
    ) == 1


@pytest.mark.asyncio
async def test_waiting_cancel_closes_interaction_and_releases_lane(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_function(
        lambda: "写入",
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write-1", name="write", input={}))
            ),
        ),
        store=store,
    )
    waiting = await runner.start(AgentRunRequest(input="写入", run_id="run-wait-cancel"))
    assert waiting.pending_interaction is not None

    result = await runner.cancel("run-wait-cancel", settlement_timeout=1)
    closed = store.load_interaction(waiting.pending_interaction.interaction_id)

    assert result.run.stop_reason is RunStopReason.CANCELLED
    assert closed is not None and closed.status.value == "closed"
    assert result.run.pending_interaction_id is None
    [tool_result] = store.load_session("default").messages[-1].tool_results
    [tool_call] = store.list_tool_calls("run-wait-cancel")
    assert tool_result.tool_use_id == "write-1"
    assert tool_result.metadata["error"]["code"] == "TOOL_NOT_STARTED"
    assert tool_call.phase is ToolCallPhase.PREPARED
    assert await runner.recover("run-wait-cancel") == result
    next_result = await runner.start(
        AgentRunRequest(input="继续", session_id="default", run_id="run-after-cancel")
    )
    assert next_result.run.phase is RunPhase.TERMINAL


@pytest.mark.asyncio
async def test_remote_cancel_timeout_only_observes_and_adds_no_timeout_fact(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    owner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    observer = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    running = asyncio.create_task(
        owner.start(AgentRunRequest(input="等待", run_id="run-remote-cancel"))
    )
    await provider.started.wait()

    with pytest.raises(IrisRunObservationTimeoutError):
        await observer.cancel("run-remote-cancel", settlement_timeout=0.01)
    events_after_timeout = store.list_events("run-remote-cancel")
    assert events_after_timeout[-1].kind is RunEventKind.CANCELLATION_REQUESTED
    assert store.load_result("run-remote-cancel") is None

    provider.release.set()
    result = await running
    assert result.run.stop_reason is RunStopReason.CANCELLED


@pytest.mark.asyncio
async def test_tool_result_commits_before_cancelled_settlement(tmp_path: Path) -> None:
    """结果与取消同轮就绪时，先提交真实结果再结算取消。"""
    started = asyncio.Event()
    release = asyncio.Event()

    class DelayedResultTool(BaseTool):
        definition = ToolDefinition(
            name="delayed",
            description="延迟返回",
            input_schema={"type": "object", "properties": {}},
        )

        async def arun(
            self,
            params: BaseModel | dict[str, object],
            context: ToolExecutionContext,
        ) -> ToolResult:
            del params
            started.set()
            await release.wait()
            result = ToolResult(
                tool_use_id=context.call_id,
                tool_name=context.tool_name,
                content=[TextBlock(text="effect-complete")],
            )
            # 中间不 await，保证 waiter 同时观察到 body 完成和 signal。
            snapshot = runner.request_cancel("run-result-before-cancel")
            assert snapshot.phase is RunPhase.ACTIVE
            return result

    registry = ToolRegistry()
    registry.register(DelayedResultTool())
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="delayed-1", name="delayed", input={}))
            ),
        ),
        store=store,
    )
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="执行", run_id="run-result-before-cancel"))
    )
    await started.wait()

    release.set()
    result = await running

    assert result.run.stop_reason is RunStopReason.CANCELLED
    [record] = store.list_tool_calls("run-result-before-cancel")
    assert record.phase is ToolCallPhase.COMMITTED
    assert record.result is not None and record.result.model_content == "effect-complete"


@pytest.mark.asyncio
async def test_public_cancel_after_claim_settles_outcome_unknown(tmp_path: Path) -> None:
    started = asyncio.Event()

    class CooperativeClaimedTool(BaseTool):
        definition = ToolDefinition(
            name="cooperative_claimed",
            description="claim 后协作取消",
            input_schema={"type": "object", "properties": {}},
        )

        async def arun(
            self,
            params: BaseModel | dict[str, object],
            context: ToolExecutionContext,
        ) -> ToolResult:
            del params
            assert context.cancellation is not None
            started.set()
            while not context.cancellation.requested:
                await asyncio.sleep(0)
            context.cancellation.raise_if_requested()
            raise AssertionError("取消后不应继续")

    registry = ToolRegistry()
    registry.register(CooperativeClaimedTool())
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(id="cooperative-1", name="cooperative_claimed", input={})
                )
            ),
        ),
        store=store,
    )
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="执行", run_id="run-claimed-cancel"))
    )
    await started.wait()

    result = await runner.cancel("run-claimed-cancel", settlement_timeout=1)

    assert result == await running
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert result.error is not None and result.error.code == "TOOL_OUTCOME_UNKNOWN"
    [record] = store.list_tool_calls("run-claimed-cancel")
    assert record.phase is ToolCallPhase.OUTCOME_UNKNOWN
    assert RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN in {
        event.kind for event in store.list_events("run-claimed-cancel")
    }


@pytest.mark.asyncio
async def test_parallel_claims_cancel_atomically_settle_outcome_unknown(
    tmp_path: Path,
) -> None:
    """并发 body 都已 claim 后取消时，全部 unresolved claim 必须一起关闭。"""
    started: set[str] = set()
    both_started = asyncio.Event()

    class CooperativeParallelTool(BaseTool):
        definition = ToolDefinition(
            name="cooperative_parallel",
            description="并发 claim 后协作取消",
            input_schema={
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
        )

        async def arun(
            self,
            params: BaseModel | dict[str, object],
            context: ToolExecutionContext,
        ) -> ToolResult:
            del params
            assert context.cancellation is not None
            started.add(context.call_id)
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            while not context.cancellation.requested:
                await asyncio.sleep(0)
            context.cancellation.raise_if_requested()
            raise AssertionError("取消后不应继续")

    registry = ToolRegistry()
    registry.register(CooperativeParallelTool())
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_batch_response(
                    ToolUseBlock(
                        id="parallel-1",
                        name="cooperative_parallel",
                        input={"index": 1},
                    ),
                    ToolUseBlock(
                        id="parallel-2",
                        name="cooperative_parallel",
                        input={"index": 2},
                    ),
                )
            ),
        ),
        store=store,
    )
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="并发取消", run_id="run-parallel-cancel"))
    )
    await asyncio.wait_for(both_started.wait(), timeout=1)

    result = await runner.cancel("run-parallel-cancel", settlement_timeout=1)

    assert result == await running
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    records = store.list_tool_calls("run-parallel-cancel")
    assert {record.tool_call_id for record in records} == {"parallel-1", "parallel-2"}
    assert all(record.phase is ToolCallPhase.OUTCOME_UNKNOWN for record in records)
    events = store.list_events("run-parallel-cancel")
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert {
        event.correlation_id
        for event in events
        if event.kind is RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN
    } == {"parallel-1", "parallel-2"}


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["thread", "async_to_thread"])
async def test_thread_callable_cancel_after_claim_discards_late_result(
    tmp_path: Path,
    placement: str,
) -> None:
    """thread worker 不能强停；取消后应 fail closed，晚到返回不得提交。"""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_effect() -> str:
        started.set()
        try:
            release.wait(timeout=2)
            return "late-effect-complete"
        finally:
            finished.set()

    registry = ToolRegistry()
    if placement == "thread":
        registry.register_function(
            blocking_effect,
            description="线程阻塞工具",
            execution_mode=CallableExecutionMode.THREAD,
        )
    else:

        async def async_effect() -> str:
            """工具内部的 to_thread 也由统一 body waiter 响应取消。"""
            return await asyncio.to_thread(blocking_effect)

        registry.register_function(
            async_effect, name="blocking_effect", description="异步工具中的线程等待"
        )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="thread-1", name="blocking_effect", input={}))
            ),
        ),
        store=store,
    )
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="执行", run_id="run-thread-cancel"))
    )

    try:
        assert await asyncio.to_thread(started.wait, 1)
        result = await runner.cancel("run-thread-cancel", settlement_timeout=1)
        assert result == await running
        assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
        assert result.error is not None and result.error.code == "TOOL_OUTCOME_UNKNOWN"
        [record] = store.list_tool_calls("run-thread-cancel")
        assert record.phase is ToolCallPhase.OUTCOME_UNKNOWN
        assert not finished.is_set()

        events_before_release = store.list_events("run-thread-cancel")
        session_before_release = store.load_session("default")
        checkpoint_before_release = store.load_checkpoint("run-thread-cancel")
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        for _ in range(3):
            await asyncio.sleep(0)

        assert store.list_events("run-thread-cancel") == events_before_release
        assert store.load_session("default") == session_before_release
        assert store.load_checkpoint("run-thread-cancel") == checkpoint_before_release
        assert store.load_result("run-thread-cancel") == result
    finally:
        release.set()
        if not running.done():
            await running


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_kind", ["callable", "base_tool"])
@pytest.mark.parametrize("returns_on_cancel", [False, True])
async def test_public_cancel_interrupts_async_body_and_preserves_known_result(
    tmp_path: Path, tool_kind: str, returns_on_cancel: bool
) -> None:
    """不读取 Iris signal 的普通异步 body 也响应公开取消。"""
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def delayed() -> str:
        """仅响应 Python task cancellation，或在清理后返回确定结果。"""
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            if not returns_on_cancel:
                raise
        finally:
            finished.set()
        return "known-effect"

    class DelayedTool(BaseTool):
        """与 callable 共用同一异步执行体。"""

        definition = ToolDefinition(
            name="delayed",
            description="受控异步工具",
            input_schema={"type": "object", "properties": {}},
        )

        async def arun(
            self, params: BaseModel | dict[str, object], context: ToolExecutionContext
        ) -> ToolResult:
            """返回实际工具结果，不读取取消信号。"""
            return ToolResult(
                tool_use_id=context.call_id,
                tool_name=context.tool_name,
                content=[TextBlock(text=await delayed())],
            )

    registry = ToolRegistry()
    if tool_kind == "callable":
        registry.register_function(delayed, description="受控异步工具")
    else:
        registry.register(DelayedTool())
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="delayed-1", name="delayed", input={}))
            ),
        ),
        store=store,
    )
    running = asyncio.create_task(runner.start(AgentRunRequest(input="执行", run_id="async")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert store.load_tool_call("async", "delayed-1").phase is ToolCallPhase.CLAIMED
        result = await runner.cancel("async", settlement_timeout=0.25)
        assert finished.is_set()
        assert result == await running
        record = store.load_tool_call("async", "delayed-1")
        if returns_on_cancel:
            assert result.run.stop_reason is RunStopReason.CANCELLED
            assert record.phase is ToolCallPhase.COMMITTED
            assert record.result.model_content == "known-effect"
        else:
            assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
            assert record.phase is ToolCallPhase.OUTCOME_UNKNOWN
    finally:
        release.set()
        await asyncio.wait_for(running, timeout=1)


@pytest.mark.asyncio
async def test_cancel_observation_timeout_preserves_pending_body_cleanup(tmp_path: Path) -> None:
    """观察时限不能提前结算仍在执行的取消清理。"""
    entered = asyncio.Event()
    release_body = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def delayed() -> str:
        """清理由测试放行，普通业务等待不读取 Iris signal。"""
        entered.set()
        try:
            await release_body.wait()
            return "released"
        finally:
            cleanup_entered.set()
            await release_cleanup.wait()

    registry = ToolRegistry()
    registry.register_function(delayed, description="受控取消清理")
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="delayed-1", name="delayed", input={}))
            ),
        ),
        store=store,
    )
    running = asyncio.create_task(runner.start(AgentRunRequest(input="执行", run_id="cleanup")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        with pytest.raises(IrisRunObservationTimeoutError):
            await runner.cancel("cleanup", settlement_timeout=0.05)
        assert cleanup_entered.is_set()
        assert store.load_result("cleanup") is None
        assert store.load_run("cleanup").phase is RunPhase.ACTIVE
        assert store.load_tool_call("cleanup", "delayed-1").phase is ToolCallPhase.CLAIMED
    finally:
        release_body.set()
        release_cleanup.set()
        result = await asyncio.wait_for(running, timeout=1)
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_public_cancel_keeps_known_prefix_in_async_tool_window(tmp_path: Path) -> None:
    """普通异步并发窗口取消时，保留已知前缀且不越过未完成调用。"""
    started: set[int] = set()
    known: set[int] = set()
    all_started = asyncio.Event()
    known_ready = asyncio.Event()
    release = asyncio.Event()

    async def read_value(index: int) -> str:
        """第一、三条完成，中间一条保持等待直到框架取消。"""
        started.add(index)
        if len(started) == 3:
            all_started.set()
        await all_started.wait()
        if index == 2:
            await release.wait()
        else:
            known.add(index)
            if known == {1, 3}:
                known_ready.set()
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="并发异步读取")
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_batch_response(
                    *(
                        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
                        for index in (1, 2, 3)
                    )
                )
            ),
        ),
        store=store,
    )
    running = asyncio.create_task(runner.start(AgentRunRequest(input="读取", run_id="window")))
    try:
        await asyncio.wait_for(known_ready.wait(), timeout=1)
        result = await runner.cancel("window", settlement_timeout=0.25)
        assert result == await running
        assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
        records = store.list_tool_calls("window")
        assert [record.phase for record in records] == [
            ToolCallPhase.COMMITTED,
            ToolCallPhase.OUTCOME_UNKNOWN,
            ToolCallPhase.OUTCOME_UNKNOWN,
        ]
        assert records[0].result.model_content == "value-1"
    finally:
        release.set()
        await asyncio.wait_for(running, timeout=1)
