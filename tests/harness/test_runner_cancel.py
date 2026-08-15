"""AgentRunner durable cancellation 与 settlement observation 测试。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from iris.exceptions import IrisRunObservationTimeoutError, IrisRunStateError
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunRequest,
    RunEvent,
    RunEventKind,
    RunPhase,
    RunResult,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    BaseTool,
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
async def test_managed_active_cancel_relays_request_and_terminal_live(
    tmp_path: Path,
) -> None:
    """Active managed run 的 cancellation 与 terminal mutation 均同步 relay。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    activation_started = asyncio.Event()
    relayed: list[RunEvent] = []
    running = asyncio.create_task(
        runner._start_managed(
            AgentRunRequest(input="等待", run_id="run-managed-cancel"),
            durable_event_callback=relayed.append,
            activation_started=activation_started,
        )
    )

    await asyncio.wait_for(activation_started.wait(), timeout=1)
    snapshot = runner.request_cancel("run-managed-cancel", reason="用户停止")
    provider.release.set()
    result = await running

    assert snapshot.cancellation_reason == "用户停止"
    assert result.run.stop_reason is RunStopReason.CANCELLED
    assert RunEventKind.CANCELLATION_REQUESTED in {event.kind for event in relayed}
    assert relayed[-1].kind is RunEventKind.RUN_TERMINAL
    assert relayed == store.list_events("run-managed-cancel")


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
            return ToolResult(
                tool_use_id=context.call_id,
                tool_name=context.tool_name,
                content=[TextBlock(text="effect-complete")],
            )

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

    snapshot = runner.request_cancel("run-result-before-cancel")
    assert snapshot.phase is RunPhase.ACTIVE
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
async def test_non_cooperative_sync_tool_delays_cancel_settlement(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_effect() -> str:
        started.set()
        release.wait(timeout=2)
        return "effect-complete"

    registry = ToolRegistry()
    registry.register_function(blocking_effect, description="同步阻塞工具")
    store = InMemoryLifecycleStore()
    owner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="blocking-1", name="blocking_effect", input={}))
            ),
        ),
        store=store,
    )
    observer = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    outcomes: list[object] = []

    def run_owner() -> None:
        try:
            outcomes.append(
                asyncio.run(
                    owner.start(AgentRunRequest(input="执行", run_id="run-sync-blocking-cancel"))
                )
            )
        except BaseException as exc:  # pragma: no cover - 失败时由主线程断言暴露
            outcomes.append(exc)

    thread = threading.Thread(target=run_owner)
    thread.start()
    assert await asyncio.to_thread(started.wait, 1)

    with pytest.raises(IrisRunObservationTimeoutError):
        await observer.cancel("run-sync-blocking-cancel", settlement_timeout=0.02)
    assert store.load_result("run-sync-blocking-cancel") is None

    release.set()
    await asyncio.to_thread(thread.join, 2)
    assert not thread.is_alive()
    [result] = outcomes
    assert not isinstance(result, BaseException)
    assert isinstance(result, RunResult)
    assert result.run.stop_reason is RunStopReason.CANCELLED


def test_cancel_validates_reason_and_observation_timeout(tmp_path: Path) -> None:
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())

    with pytest.raises(IrisRunStateError, match="reason"):
        runner.request_cancel("missing", reason=" ")
