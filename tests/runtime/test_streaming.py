"""Runtime live streaming 与 durable effect 顺序测试。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    FakeStreamingProvider,
    MutableCancellationSignal,
    build_runtime,
    start_activation,
)
from pydantic import BaseModel

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisRunPersistenceError
from iris.message import (
    LLMRequest,
    LLMResponse,
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockRef,
    ModelBlockStarted,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ProviderStreamError,
    TextBlock,
    ToolUseBlock,
)
from iris.runtime import (
    AgentRuntime,
    RuntimeActivationOutcome,
    RuntimeCursor,
    RuntimeEventSink,
    RuntimeProvider,
    RuntimeStreamEvent,
    RuntimeToolCall,
    RuntimeToolResultCommit,
    ToolCallClaim,
)
from iris.tools import (
    BaseTool,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="streaming-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
        permissions={"workspace": ".", "writes": "allow"},
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id=f"response-{text}",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=4,
        output_tokens=2,
        total_tokens=6,
    )


def _tool_response(*calls: ToolUseBlock) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-tools",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要调用工具。"), *calls],
        finish_reason="tool_calls",
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
    )


def _stream_events(
    response: LLMResponse,
    *,
    stream_id: str,
    fragment_arguments: bool = False,
) -> list[ModelStreamEvent]:
    scope = ModelStreamScope(
        model_stream_id=stream_id,
        provider=response.provider,
        model=response.model,
        attempt=1,
    )
    occurred_at = datetime.now(UTC)
    sequence = 1
    events: list[ModelStreamEvent] = [
        ModelResponseStarted(
            scope=scope,
            sequence=sequence,
            occurred_at=occurred_at,
            response_id=response.id,
        )
    ]
    for index, content in enumerate(response.content):
        sequence += 1
        if isinstance(content, TextBlock):
            block = ModelBlockRef(index=index, block_id=f"text-{index}", kind="text")
            events.append(
                ModelBlockStarted(
                    scope=scope,
                    sequence=sequence,
                    occurred_at=occurred_at,
                    block=block,
                )
            )
            sequence += 1
            events.append(
                ModelBlockDelta(
                    scope=scope,
                    sequence=sequence,
                    occurred_at=occurred_at,
                    block=block,
                    channel="text",
                    delta=content.text,
                    snapshot=content.text,
                )
            )
            sequence += 1
            events.append(
                ModelBlockCompleted(
                    scope=scope,
                    sequence=sequence,
                    occurred_at=occurred_at,
                    block=block,
                )
            )
            continue
        assert isinstance(content, ToolUseBlock)
        block = ModelBlockRef(
            index=index,
            block_id=f"tool-{index}",
            kind="tool_call",
            tool_call_id=content.id,
        )
        events.append(
            ModelBlockStarted(
                scope=scope,
                sequence=sequence,
                occurred_at=occurred_at,
                block=block,
            )
        )
        sequence += 1
        events.append(
            ModelBlockDelta(
                scope=scope,
                sequence=sequence,
                occurred_at=occurred_at,
                block=block,
                channel="tool_name",
                delta=content.name,
                snapshot=content.name,
            )
        )
        arguments = json.dumps(content.input, ensure_ascii=False)
        fragments = [arguments]
        if fragment_arguments:
            midpoint = len(arguments) // 2
            fragments = [arguments[:midpoint], arguments[midpoint:]]
        snapshot = ""
        for fragment in fragments:
            sequence += 1
            snapshot += fragment
            events.append(
                ModelBlockDelta(
                    scope=scope,
                    sequence=sequence,
                    occurred_at=occurred_at,
                    block=block,
                    channel="tool_arguments",
                    delta=fragment,
                    snapshot=snapshot,
                )
            )
        sequence += 1
        events.append(
            ModelBlockCompleted(
                scope=scope,
                sequence=sequence,
                occurred_at=occurred_at,
                block=block,
            )
        )
    sequence += 1
    events.append(
        ModelResponseCompleted(
            scope=scope,
            sequence=sequence,
            occurred_at=occurred_at,
            response=response,
            semantic_output_emitted=True,
        )
    )
    return events


class RecordingSink(RuntimeEventSink):
    """同步记录 runtime live event，并允许在 emit 时观察状态。"""

    def __init__(
        self,
        callback: Callable[[RuntimeStreamEvent], None] | None = None,
    ) -> None:
        self.events: list[RuntimeStreamEvent] = []
        self._callback = callback

    def emit(self, event: RuntimeStreamEvent) -> None:
        """记录一条事件。"""
        self.events.append(event)
        if self._callback is not None:
            self._callback(event)


class _ClosableModelStream(AsyncIterator[ModelStreamEvent]):
    """记录 Runtime 是否关闭 typed provider iterator。"""

    def __init__(self, events: Sequence[ModelStreamEvent]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> _ClosableModelStream:
        """返回当前 iterator。"""
        return self

    async def __anext__(self) -> ModelStreamEvent:
        """返回下一条 typed model event。"""
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        """记录 Runtime 已释放 provider iterator。"""
        self.closed = True


class _ClosableStreamingProvider(FakeProvider):
    """返回可观察关闭状态的 typed stream provider。"""

    def __init__(self, events: Sequence[ModelStreamEvent]) -> None:
        super().__init__([])
        self.iterator = _ClosableModelStream(events)

    def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """返回唯一受控 stream。"""
        del request
        return self.iterator


def _runtime(
    provider: RuntimeProvider,
    tmp_path: Path,
    *,
    registry: ToolRegistry | None = None,
) -> AgentRuntime:
    resolved_registry = registry or ToolRegistry()
    return build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        tool_registry=resolved_registry,
        tool_view=resolved_registry.view(),
        workspace_root=tmp_path,
    )


@pytest.mark.asyncio
async def test_execute_without_sink_keeps_complete_only_path(tmp_path: Path) -> None:
    provider = FakeProvider([_text_response("完成")])
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await _runtime(provider, tmp_path).execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 1
    assert provider.requests[0].stream is False
    assert len(commits.model_commits) == 1


@pytest.mark.asyncio
async def test_sink_rejects_complete_only_provider_without_fallback(tmp_path: Path) -> None:
    provider = FakeProvider([_text_response("不能回退")])
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    sink = RecordingSink()

    result = await _runtime(provider, tmp_path).execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        stream_sink=sink,
    )

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "PROVIDER_STREAM_ERROR"
    assert result.error.source == "provider"
    assert provider.requests == []
    assert commits.model_commits == []
    assert [event.kind for event in sink.events] == ["model.step.started"]


@pytest.mark.asyncio
async def test_streaming_success_emits_before_single_model_commit(tmp_path: Path) -> None:
    response = _text_response("分段完成")
    provider = FakeStreamingProvider([_stream_events(response, stream_id="stream-1")])
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    sink = RecordingSink()

    result = await _runtime(provider, tmp_path).execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        stream_sink=sink,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert provider.requests == []
    assert len(provider.stream_requests) == 1
    assert provider.stream_requests[0].stream is True
    assert len(commits.model_commits) == 1
    assert commits.model_commits[0].assistant_message.model_dump(exclude={"timestamp"}) == (
        response.to_msg().model_dump(exclude={"timestamp"})
    )
    assert sink.events[0].kind == "model.step.started"
    model_events = [event.model_event for event in sink.events[1:]]
    assert [event.kind for event in model_events if event is not None] == [
        event.kind for event in _stream_events(response, stream_id="expected")
    ]


@pytest.mark.asyncio
async def test_streaming_terminal_closes_provider_iterator(tmp_path: Path) -> None:
    """Runtime 读取 terminal 后仍关闭 provider typed iterator。"""
    response = _text_response("完成")
    provider = _ClosableStreamingProvider(_stream_events(response, stream_id="closable-stream"))
    activation = start_activation()

    result = await _runtime(provider, tmp_path).execute(
        activation,
        commits=FakeRuntimeCommitPort(activation),
        cancellation=MutableCancellationSignal(),
        stream_sink=RecordingSink(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert provider.iterator.closed


@pytest.mark.asyncio
async def test_stream_failure_cancel_and_eof_never_commit(tmp_path: Path) -> None:
    response = _text_response("部分输出")
    partial = _stream_events(response, stream_id="failed")[:3]
    started = partial[0]
    assert isinstance(started, ModelResponseStarted)
    failed = ModelResponseFailed(
        scope=started.scope,
        sequence=4,
        occurred_at=datetime.now(UTC),
        error=ProviderStreamError(
            code="PROVIDER_STREAM_ERROR",
            message="上游失败",
            retryable=True,
        ),
        semantic_output_emitted=True,
    )
    cancelled = ModelResponseCancelled(
        scope=started.scope,
        sequence=4,
        occurred_at=datetime.now(UTC),
        semantic_output_emitted=True,
    )

    for events, expected_code in (
        ([*partial, failed], "PROVIDER_STREAM_ERROR"),
        ([*partial, cancelled], "PROVIDER_STREAM_ERROR"),
        (partial, "PROVIDER_STREAM_INTERRUPTED"),
    ):
        provider = FakeStreamingProvider([events])
        activation = start_activation()
        commits = FakeRuntimeCommitPort(activation)

        result = await _runtime(provider, tmp_path).execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            stream_sink=RecordingSink(),
        )

        assert result.outcome is RuntimeActivationOutcome.FAILED
        assert result.error is not None and result.error.code == expected_code
        assert commits.model_commits == []


@pytest.mark.asyncio
async def test_stream_local_cancellation_propagates(tmp_path: Path) -> None:
    class CancellingProvider(FakeProvider):
        async def stream(
            self,
            request: LLMRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            del request
            raise asyncio.CancelledError
            yield

    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    with pytest.raises(asyncio.CancelledError):
        await _runtime(CancellingProvider([]), tmp_path).execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            stream_sink=RecordingSink(),
        )

    assert commits.model_commits == []


@pytest.mark.asyncio
async def test_fragmented_tool_arguments_do_not_preflight_before_terminal(
    tmp_path: Path,
) -> None:
    validation_calls: list[str] = []
    effects: list[str] = []

    class CountingEchoTool(BaseTool):
        definition = ToolDefinition(
            name="echo",
            description="记录参数验证与 effect 顺序",
            input_schema={"type": "object"},
        )

        def validate_input(self, params: dict[str, Any]) -> dict[str, Any]:
            value = str(params["value"])
            validation_calls.append(value)
            return {"value": value}

        async def arun(
            self,
            params: BaseModel | dict[str, Any],
            context: ToolExecutionContext,
        ) -> ToolResult:
            del context
            assert isinstance(params, dict)
            value = str(params["value"])
            effects.append(value)
            return ToolResult(content=[TextBlock(text=value)])

    registry = ToolRegistry()
    registry.register(CountingEchoTool())
    tool_response = _tool_response(
        ToolUseBlock(id="echo-1", name="echo", input={"value": "完整参数"})
    )
    final_response = _text_response("完成")
    provider = FakeStreamingProvider(
        [
            _stream_events(
                tool_response,
                stream_id="tool-stream",
                fragment_arguments=True,
            ),
            _stream_events(final_response, stream_id="final-stream"),
        ]
    )

    def assert_no_early_tool_work(event: RuntimeStreamEvent) -> None:
        if (
            event.kind == "model.event"
            and event.model_event is not None
            and event.model_event.scope.model_stream_id == "tool-stream"
        ):
            assert validation_calls == []
            assert effects == []

    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    result = await _runtime(provider, tmp_path, registry=registry).execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        stream_sink=RecordingSink(assert_no_early_tool_work),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert validation_calls == ["完整参数"]
    assert effects == ["完整参数"]


@pytest.mark.asyncio
async def test_tool_live_events_follow_claim_body_and_commit_order(tmp_path: Path) -> None:
    timeline: list[str] = []

    def effect() -> str:
        timeline.append("body")
        return "effect"

    registry = ToolRegistry()
    registry.register_function(
        effect,
        description="执行 effect",
    )
    provider = FakeStreamingProvider(
        [
            _stream_events(
                _tool_response(ToolUseBlock(id="effect-1", name="effect", input={})),
                stream_id="tool-stream",
            ),
            _stream_events(_text_response("完成"), stream_id="final-stream"),
        ]
    )
    activation = start_activation()

    class OrderingPort(FakeRuntimeCommitPort):
        def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
            claim = super().claim_tool_call(call)
            timeline.append("claim")
            return claim

        def commit_tool_result(self, commit: RuntimeToolResultCommit) -> RuntimeCursor:
            cursor = super().commit_tool_result(commit)
            timeline.append("commit")
            return cursor

    def record_tool_event(event: RuntimeStreamEvent) -> None:
        if event.kind.startswith("tool."):
            timeline.append(event.kind)

    result = await _runtime(provider, tmp_path, registry=registry).execute(
        activation,
        commits=OrderingPort(activation),
        cancellation=MutableCancellationSignal(),
        stream_sink=RecordingSink(record_tool_event),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert timeline[:6] == [
        "tool.preparing",
        "claim",
        "tool.started",
        "body",
        "commit",
        "tool.completed",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_effects", "present", "absent"),
    [
        ("claim_tool_call", [], [], ["tool.started", "tool.completed"]),
        ("commit_tool_result", ["effect"], ["tool.started"], ["tool.completed"]),
    ],
)
async def test_tool_live_events_require_successful_claim_and_commit(
    tmp_path: Path,
    failure: str,
    expected_effects: list[str],
    present: list[str],
    absent: list[str],
) -> None:
    effects: list[str] = []

    def effect() -> str:
        effects.append("effect")
        return "effect"

    registry = ToolRegistry()
    registry.register_function(effect, description="执行 effect")
    provider = FakeStreamingProvider(
        [
            _stream_events(
                _tool_response(ToolUseBlock(id="effect-1", name="effect", input={})),
                stream_id="tool-stream",
            )
        ]
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, fail_at=failure)
    sink = RecordingSink()

    with pytest.raises(IrisRunPersistenceError):
        await _runtime(provider, tmp_path, registry=registry).execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            stream_sink=sink,
        )

    kinds = [event.kind for event in sink.events]
    assert effects == expected_effects
    assert all(kind in kinds for kind in present)
    assert all(kind not in kinds for kind in absent)


@pytest.mark.asyncio
async def test_cancel_after_claim_emits_started_but_not_body_or_completed(
    tmp_path: Path,
) -> None:
    effects: list[str] = []
    signal = MutableCancellationSignal()

    def effect() -> str:
        effects.append("effect")
        return "effect"

    registry = ToolRegistry()
    registry.register_function(effect, description="执行 effect")
    provider = FakeStreamingProvider(
        [
            _stream_events(
                _tool_response(ToolUseBlock(id="effect-1", name="effect", input={})),
                stream_id="tool-stream",
            )
        ]
    )
    activation = start_activation()

    class ClaimCancellingPort(FakeRuntimeCommitPort):
        def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
            claim = super().claim_tool_call(call)
            signal.requested = True
            return claim

    sink = RecordingSink()
    result = await _runtime(provider, tmp_path, registry=registry).execute(
        activation,
        commits=ClaimCancellingPort(activation),
        cancellation=signal,
        stream_sink=sink,
    )

    kinds = [event.kind for event in sink.events]
    assert result.outcome is RuntimeActivationOutcome.OUTCOME_UNKNOWN
    assert effects == []
    assert "tool.started" in kinds
    assert "tool.completed" not in kinds


@pytest.mark.asyncio
async def test_parallel_tool_bodies_can_finish_out_of_order_but_commit_in_order(
    tmp_path: Path,
) -> None:
    body_order: list[str] = []
    commit_order: list[str] = []
    second_done = asyncio.Event()

    async def first() -> str:
        await second_done.wait()
        body_order.append("first")
        return "first"

    async def second() -> str:
        body_order.append("second")
        second_done.set()
        return "second"

    registry = ToolRegistry()
    registry.register_function(first, description="第一个", concurrency_safe=True)
    registry.register_function(second, description="第二个", concurrency_safe=True)
    provider = FakeStreamingProvider(
        [
            _stream_events(
                _tool_response(
                    ToolUseBlock(id="first-1", name="first", input={}),
                    ToolUseBlock(id="second-1", name="second", input={}),
                ),
                stream_id="tool-stream",
            ),
            _stream_events(_text_response("完成"), stream_id="final-stream"),
        ]
    )
    activation = start_activation()

    class OrderingPort(FakeRuntimeCommitPort):
        def commit_tool_result(self, commit: RuntimeToolResultCommit) -> RuntimeCursor:
            cursor = super().commit_tool_result(commit)
            commit_order.append(commit.tool_call.tool_name)
            return cursor

    completed_order: list[str] = []

    def record_completed(event: RuntimeStreamEvent) -> None:
        if event.kind == "tool.completed":
            assert event.tool_name is not None
            completed_order.append(event.tool_name)

    result = await _runtime(provider, tmp_path, registry=registry).execute(
        activation,
        commits=OrderingPort(activation),
        cancellation=MutableCancellationSignal(),
        stream_sink=RecordingSink(record_completed),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert body_order == ["second", "first"]
    assert commit_order == ["first", "second"]
    assert completed_order == ["first", "second"]


@pytest.mark.asyncio
async def test_sink_error_propagates_without_model_commit(tmp_path: Path) -> None:
    class SinkError(RuntimeError):
        pass

    class FailingSink(RecordingSink):
        def emit(self, event: RuntimeStreamEvent) -> None:
            if event.kind == "model.event":
                raise SinkError("publisher failed")
            super().emit(event)

    response = _text_response("不应提交")
    provider = FakeStreamingProvider([_stream_events(response, stream_id="stream-1")])
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    with pytest.raises(SinkError, match="publisher failed"):
        await _runtime(provider, tmp_path).execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            stream_sink=FailingSink(),
        )

    assert commits.model_commits == []


@pytest.mark.asyncio
async def test_tool_started_sink_error_is_not_tool_body_failure(tmp_path: Path) -> None:
    class SinkError(RuntimeError):
        pass

    class FailingSink(RecordingSink):
        def emit(self, event: RuntimeStreamEvent) -> None:
            if event.kind == "tool.started":
                raise SinkError("tool publisher failed")
            super().emit(event)

    effects: list[str] = []

    def effect() -> str:
        effects.append("effect")
        return "effect"

    registry = ToolRegistry()
    registry.register_function(effect, description="执行 effect")
    provider = FakeStreamingProvider(
        [
            _stream_events(
                _tool_response(ToolUseBlock(id="effect-1", name="effect", input={})),
                stream_id="tool-stream",
            )
        ]
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    with pytest.raises(SinkError, match="tool publisher failed"):
        await _runtime(provider, tmp_path, registry=registry).execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            stream_sink=FailingSink(),
        )

    assert effects == []
    assert len(commits.claims) == 1
    assert commits.tool_commits == []
