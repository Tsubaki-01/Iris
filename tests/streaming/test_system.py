"""Streaming 全链路与 durable recovery 系统回归测试。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import litellm
import pytest
from pydantic import BaseModel

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.harness import AgentRunner, SessionManager, SubmissionEvent
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import (
    AgentRunRequest,
    LifecycleStore,
    RunEvent,
    RunPhase,
    RunStopReason,
)
from iris.message import LLMRequest, LLMResponse, ModelStreamEvent, TextBlock
from iris.providers import ProviderClient
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.streaming import (
    CancelAccepted,
    CancelCommand,
    DurableRunCursor,
    DurableSyncItem,
    GatewayStreamItem,
    GatewaySubscription,
    LiveCursor,
    LiveEnvelope,
    LiveStreamBroker,
    ReplayGap,
    ResumeAccepted,
    ResumeCommand,
    SSEAdapter,
    StreamingGateway,
    SubscribeCommand,
    SubscriptionTerminal,
    SyncAccepted,
    SyncCommand,
)
from iris.tools import (
    BaseTool,
    DefaultPermissionPolicy,
    PermissionDecision,
    ToolArtifact,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)
from tests.runtime.fakes import build_runtime


class _ControlledRawStream(AsyncIterator[dict[str, Any]]):
    """按指定 pull index 等待测试放行的确定性 raw stream。"""

    def __init__(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        gated_indexes: Sequence[int] = (),
    ) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self.gates = {index: asyncio.Event() for index in gated_indexes}
        self.waiting = {index: asyncio.Event() for index in gated_indexes}
        self.pull_count = 0
        self.cancelled = False
        self.cancelled_event = asyncio.Event()
        self.closed = False

    def __aiter__(self) -> _ControlledRawStream:
        """返回当前 raw iterator。"""
        return self

    async def __anext__(self) -> dict[str, Any]:
        """在受控同步点后返回下一个 Chat Completion chunk。"""
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        index = self._index
        self.pull_count += 1
        gate = self.gates.get(index)
        if gate is not None:
            self.waiting[index].set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                self.cancelled_event.set()
                raise
        chunk = self._chunks[index]
        self._index += 1
        return chunk

    async def aclose(self) -> None:
        """记录 provider client 已关闭 raw iterator。"""
        self.closed = True


class _FakeChatBackend:
    """替代 LiteLLM 网络调用并按顺序返回受控 raw streams。"""

    def __init__(self, streams: Sequence[_ControlledRawStream]) -> None:
        self._streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> _ControlledRawStream:
        """记录 Chat Completion 参数并返回下一条脚本。"""
        self.calls.append(kwargs)
        return self._streams.pop(0)


class _RecordingStreamingProvider:
    """记录 typed 请求并委托真实 ProviderClient streaming contract。"""

    def __init__(self) -> None:
        self._client = ProviderClient(provider="openai", api_key="test-key")
        self.complete_requests: list[LLMRequest] = []
        self.stream_requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录意外 complete 调用并使系统测试立即失败。"""
        self.complete_requests.append(request)
        raise AssertionError("live publisher 路径不得回退到 complete()")

    def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """记录请求并返回真实 ProviderClient typed stream。"""
        self.stream_requests.append(request)
        return self._client.stream(request)


class _RecordingPermissionPolicy(DefaultPermissionPolicy):
    """记录每次 preflight 与 effect 前的权限刷新。"""

    def __init__(self) -> None:
        super().__init__(write_mode="confirm")
        self.checks: list[tuple[str, dict[str, Any]]] = []

    def check(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PermissionDecision:
        """记录调用后委托默认三态权限策略。"""
        self.checks.append((tool.name, dict(params)))
        return super().check(tool, params, context)


class _SensitiveWriteTool(BaseTool):
    """返回含本地字段的写工具，用于验证远端投影。"""

    def __init__(self, workspace_root: Path) -> None:
        self.definition = ToolDefinition(
            name="write_secret",
            description="写入测试值",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            capabilities={ToolCapability.WRITE},
        )
        self._artifact_path = (workspace_root / "private-result.txt").resolve()
        self.calls: list[str] = []

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """记录 effect 并返回包含敏感本地字段的成功结果。"""
        del context
        values = params.model_dump() if isinstance(params, BaseModel) else params
        value = str(values["value"])
        self.calls.append(value)
        return ToolResult(
            tool_use_id="write-1",
            tool_name=self.name,
            content=[TextBlock(text=f"已写入 {value}")],
            data={"secret_data": value},
            artifact=ToolArtifact(
                path=self._artifact_path,
                mime_type="text/plain",
                size_bytes=8,
                preview="安全预览",
            ),
            stats={"private_latency": 1},
            metadata={"private_trace": "trace-secret"},
        )


def _text_chunks(*parts: str, finish: bool = True) -> list[dict[str, Any]]:
    """构造可由真实 accumulator 消费的 Chat Completion chunks。"""
    chunks = [
        {
            "id": "response-system",
            "model": "fake-model",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": part}}],
        }
        for part in parts
    ]
    if finish:
        chunks.extend(
            [
                {
                    "id": "response-system",
                    "model": "fake-model",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {
                    "id": "response-system",
                    "model": "fake-model",
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    },
                },
            ]
        )
    return chunks


def _tool_chunks(
    calls: Sequence[tuple[str, str, str]],
    *,
    fragmented: bool = False,
) -> list[dict[str, Any]]:
    """构造一个或多个 function tool calls 的 raw chunks。"""
    chunks: list[dict[str, Any]] = []
    for index, (call_id, name, arguments) in enumerate(calls):
        name_parts = (name[: len(name) // 2], name[len(name) // 2 :]) if fragmented else (name,)
        argument_parts = (
            (arguments[: len(arguments) // 2], arguments[len(arguments) // 2 :])
            if fragmented
            else (arguments,)
        )
        part_count = max(len(name_parts), len(argument_parts))
        for part_index in range(part_count):
            function: dict[str, str] = {}
            if part_index < len(name_parts) and name_parts[part_index]:
                function["name"] = name_parts[part_index]
            if part_index < len(argument_parts) and argument_parts[part_index]:
                function["arguments"] = argument_parts[part_index]
            chunks.append(
                {
                    "id": "response-tools",
                    "model": "fake-model",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call_id if part_index == 0 else None,
                                        "type": "function",
                                        "function": function,
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
    chunks.extend(
        [
            {
                "id": "response-tools",
                "model": "fake-model",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "response-tools",
                "model": "fake-model",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 4,
                    "total_tokens": 9,
                },
            },
        ]
    )
    return chunks


def _build_system(
    tmp_path: Path,
    provider: _RecordingStreamingProvider,
    *,
    store: LifecycleStore | None = None,
    registry: ToolRegistry | None = None,
    permission_policy: DefaultPermissionPolicy | None = None,
    subscription_capacity: int = 256,
) -> tuple[
    AgentRunner,
    SessionManager,
    LiveStreamBroker,
    StreamingGateway,
    LifecycleStore,
]:
    """装配共用 broker 的真实 runtime、runner、manager 与 gateway。"""
    resolved_store = store or InMemoryLifecycleStore()
    resolved_registry = registry or ToolRegistry()
    broker = LiveStreamBroker(
        replay_capacity_per_scope=512,
        subscription_capacity=subscription_capacity,
    )
    runtime = build_runtime(
        agent_config=AgentConfig(
            name="streaming-system-agent",
            model={"provider": "openai", "name": "fake-model"},
            system="你是本地助手。",
            permissions={"workspace": ".", "writes": "confirm"},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
        ),
        provider=provider,
        tool_registry=resolved_registry,
        tool_view=resolved_registry.view(),
        workspace_root=tmp_path,
        permission_policy=permission_policy,
    )
    runner = AgentRunner(runtime=runtime, store=resolved_store, live_publisher=broker)
    manager = SessionManager(
        runner,
        "session-system",
        submission_publisher=broker,
    )
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=broker,
        session_id="session-system",
        durable_page_size=3,
        allow_tool_arguments=True,
    )
    return runner, manager, broker, gateway, resolved_store


async def _take_until(
    subscription: GatewaySubscription,
    predicate: Callable[[GatewayStreamItem], bool],
) -> list[GatewayStreamItem]:
    """读取有限事件直到命中确定性终点。"""
    items: list[GatewayStreamItem] = []
    for _ in range(256):
        item = await asyncio.wait_for(anext(subscription), timeout=1)
        items.append(item)
        if predicate(item):
            return items
    raise AssertionError("subscription 未在有限事件内到达预期终点")


async def _take_manager_event(
    stream: AsyncIterator[RunEvent | SubmissionEvent],
    predicate: Callable[[RunEvent | SubmissionEvent], bool],
) -> RunEvent | SubmissionEvent:
    """从 manager 唯一 mixed stream 读取到指定事实。"""
    for _ in range(256):
        item = await asyncio.wait_for(anext(stream), timeout=1)
        if predicate(item):
            return item
    raise AssertionError("manager stream 未在有限事件内到达预期事实")


def _is_live_kind(kind: str) -> Callable[[GatewayStreamItem], bool]:
    """构造匹配 live envelope kind 的断言谓词。"""
    return lambda item: isinstance(item, LiveEnvelope) and item.kind == kind


def _live_envelopes(items: Sequence[GatewayStreamItem]) -> list[LiveEnvelope]:
    """从 gateway items 中提取 live envelopes。"""
    return [item for item in items if isinstance(item, LiveEnvelope)]


@pytest.mark.asyncio
async def test_text_streaming_commits_only_complete_response_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial 只走 live plane，完成响应只提交一次并保留 usage。"""
    raw = _ControlledRawStream(_text_chunks("分段", "完成"), gated_indexes=(0, 2))
    backend = _FakeChatBackend([raw])
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    runner, manager, broker, gateway, store = _build_system(tmp_path, provider)
    session_subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-session",
            scope="session",
            scope_id="session-system",
        )
    )
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(
                input="开始流式回答",
                run_id="run-text",
                session_id="session-system",
            )
        )
    )
    await asyncio.wait_for(raw.waiting[0].wait(), timeout=1)
    run_subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-run",
            scope="run",
            scope_id="run-text",
        )
    )

    raw.gates[0].set()
    await asyncio.wait_for(raw.waiting[2].wait(), timeout=1)
    partial_items = await _take_until(
        run_subscription,
        lambda item: (
            isinstance(item, LiveEnvelope)
            and item.kind == "model.block.delta"
            and item.payload.get("snapshot") == "分段完成"
        ),
    )
    preterminal = runner.get_session("session-system")
    assert runner.get_result("run-text") is None
    assert all(message.role.value != "assistant" for message in preterminal.messages)
    checkpoint = store.load_checkpoint("run-text")
    assert checkpoint is not None and checkpoint.model_steps_committed == 0

    raw.gates[2].set()
    result = await task
    run_items = partial_items + await _take_until(
        run_subscription,
        _is_live_kind("run.terminal"),
    )
    session_items = await _take_until(
        session_subscription,
        _is_live_kind("run.terminal"),
    )

    assert result.run.phase is RunPhase.TERMINAL
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.run.usage.model_steps_committed == 1
    assert result.run.usage.input_tokens == 7
    assert result.run.usage.output_tokens == 3
    assert result.run.usage.total_tokens == 10
    assert result.assistant_message is not None
    assert result.assistant_message.text == "分段完成"
    assert result.assistant_message.metadata["finish_reason"] == "stop"
    assert provider.complete_requests == []
    assert len(provider.stream_requests) == 1
    assert provider.stream_requests[0].stream is True
    assert backend.calls[0]["stream_options"] == {"include_usage": True}
    assert raw.closed

    for items in (run_items, session_items):
        sequences = [item.live_sequence for item in _live_envelopes(items)]
        assert sequences == sorted(sequences)
        assert len(sequences) == len(set(sequences))
        assert any(
            item.kind == "model.block.delta" and item.payload.get("snapshot") == "分段完成"
            for item in _live_envelopes(items)
        )

    durable_events = runner.list_events("run-text")
    replay = gateway.subscribe(
        SubscribeCommand(
            request_id="replay-run",
            scope="run",
            scope_id="run-text",
            cursor=LiveCursor(
                stream_epoch=broker.current_epoch(),
                scope="run",
                scope_id="run-text",
                after_live_sequence=0,
            ),
        )
    )
    replay_items = await _take_until(
        replay,
        lambda item: (
            isinstance(item, LiveEnvelope) and item.durable_sequence == durable_events[-1].sequence
        ),
    )
    assert [
        item.durable_sequence
        for item in _live_envelopes(replay_items)
        if item.durable_sequence is not None
    ] == [event.sequence for event in durable_events]

    await replay.aclose()
    await run_subscription.aclose()
    await session_subscription.aclose()
    await manager.close()
    broker.close()


@pytest.mark.asyncio
async def test_midstream_eof_fails_without_durable_partial_or_model_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """语义 partial 后 EOF 形成安全失败，且不触发 durable model/tool 事实。"""
    raw = _ControlledRawStream(
        _text_chunks("仅在live可见", finish=False),
        gated_indexes=(0,),
    )
    backend = _FakeChatBackend([raw])
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    runner, manager, broker, gateway, store = _build_system(tmp_path, provider)
    session_subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-failure",
            scope="session",
            scope_id="session-system",
        )
    )
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(
                input="触发中途失败",
                run_id="run-failure",
                session_id="session-system",
            )
        )
    )
    await asyncio.wait_for(raw.waiting[0].wait(), timeout=1)
    raw.gates[0].set()
    result = await task
    items = await _take_until(session_subscription, _is_live_kind("run.terminal"))

    assert result.run.phase is RunPhase.TERMINAL
    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None and result.error.source == "provider"
    assert result.run.usage.model_steps_committed == 0
    assert result.run.usage.tool_calls_committed == 0
    assert runner.list_tool_calls("run-failure") == []
    assert any(
        item.kind == "model.block.delta" and item.payload.get("snapshot") == "仅在live可见"
        for item in _live_envelopes(items)
    )
    failed = next(item for item in _live_envelopes(items) if item.kind == "model.response.failed")
    assert failed.payload["error"] == {
        "code": "PROVIDER_STREAM_INTERRUPTED",
        "message": "provider stream在合法终态前结束",
        "retryable": False,
    }
    durable_payload = json.dumps(
        {
            "session": runner.get_session("session-system").model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "events": [
                event.model_dump(mode="json") for event in runner.list_events("run-failure")
            ],
            "checkpoint": (
                checkpoint.model_dump(mode="json")
                if (checkpoint := store.load_checkpoint("run-failure")) is not None
                else None
            ),
        },
        ensure_ascii=False,
    )
    assert "仅在live可见" not in durable_payload
    assert raw.closed

    await session_subscription.aclose()
    await manager.close()
    broker.close()


@pytest.mark.asyncio
async def test_fragmented_tool_hitl_resume_preserves_effect_guards_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 tool terminal 后才预检，typed resume 后才 claim 与执行。"""
    tool_stream = _ControlledRawStream(
        _tool_chunks(
            [("write-1", "write_secret", '{"value":"x"}')],
            fragmented=True,
        ),
        gated_indexes=(0, 2),
    )
    final_stream = _ControlledRawStream(_text_chunks("工具已完成"))
    backend = _FakeChatBackend([tool_stream, final_stream])
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    policy = _RecordingPermissionPolicy()
    tool = _SensitiveWriteTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    runner, manager, broker, gateway, _ = _build_system(
        tmp_path,
        provider,
        registry=registry,
        permission_policy=policy,
    )
    receipt = await manager.submit("写入值")
    await asyncio.wait_for(tool_stream.waiting[0].wait(), timeout=1)
    subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-tool",
            scope="run",
            scope_id=receipt.run_id,
        )
    )

    tool_stream.gates[0].set()
    await asyncio.wait_for(tool_stream.waiting[2].wait(), timeout=1)
    partial_items = await _take_until(
        subscription,
        lambda item: (
            isinstance(item, LiveEnvelope)
            and item.kind == "model.block.delta"
            and item.payload.get("channel") == "tool_arguments"
            and item.payload.get("snapshot") == '{"value":"x"}'
        ),
    )
    assert tool.calls == []
    assert policy.checks == []
    assert runner.list_tool_calls(receipt.run_id) == []

    tool_stream.gates[2].set()
    waiting_items = await _take_until(
        subscription,
        _is_live_kind("interaction.suspended"),
    )
    waiting = runner.get_result(receipt.run_id)
    assert waiting is not None and waiting.run.phase is RunPhase.WAITING
    assert waiting.pending_interaction is not None
    assert tool.calls == []
    assert policy.checks == [("write_secret", {"value": "x"})]
    prepared = runner.list_tool_calls(receipt.run_id)
    assert len(prepared) == 1 and prepared[0].phase.value == "prepared"

    command_result = await gateway.handle(
        ResumeCommand(
            request_id="resume-tool",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    )
    assert isinstance(command_result, ResumeAccepted)
    assert command_result.result.run.stop_reason is RunStopReason.COMPLETED
    assert tool.calls == ["x"]
    assert policy.checks == [
        ("write_secret", {"value": "x"}),
        ("write_secret", {"value": "x"}),
        ("write_secret", {"value": "x"}),
    ]
    completed_items = await _take_until(subscription, _is_live_kind("run.terminal"))
    envelopes = _live_envelopes(partial_items + waiting_items + completed_items)
    ordered_kinds = [item.kind for item in envelopes]
    assert ordered_kinds.index("tool.preparing") < ordered_kinds.index("interaction.suspended")
    assert ordered_kinds.index("tool.started") < ordered_kinds.index("tool.completed")
    assert ordered_kinds.index("tool_call.claimed") < ordered_kinds.index("tool.started")
    assert ordered_kinds.index("tool.started") < ordered_kinds.index("tool_call.committed")

    direct = runner.list_tool_calls(receipt.run_id)[0]
    assert direct.result is not None and direct.result.artifact is not None
    remote = gateway.durable_sync([DurableRunCursor(run_id=receipt.run_id, after_sequence=0)]).runs[
        0
    ]
    remote_call = remote.tool_calls[0]
    assert remote_call.result is not None
    assert remote_call.result.data == {}
    assert remote_call.result.artifact is None
    assert remote_call.result.stats == {}
    assert remote_call.result.metadata == {}
    remote_json = remote.model_dump_json()
    assert str(tool._artifact_path) not in remote_json
    assert "trace-secret" not in remote_json
    assert "private_latency" not in remote_json

    await subscription.aclose()
    await manager.close()
    broker.close()


@pytest.mark.asyncio
async def test_parallel_tool_bodies_finish_out_of_order_but_commit_by_ordinal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """并发 body 可乱序结束，durable result 与 history 仍按模型 ordinal。"""
    first_release = asyncio.Event()
    second_release = asyncio.Event()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    second_finished = asyncio.Event()
    body_order: list[str] = []

    async def first(value: str) -> str:
        """等待测试放行第一条只读工具。"""
        first_started.set()
        await first_release.wait()
        body_order.append(f"first:{value}")
        return "first-result"

    async def second(value: str) -> str:
        """等待测试放行第二条只读工具。"""
        second_started.set()
        await second_release.wait()
        body_order.append(f"second:{value}")
        second_finished.set()
        return "second-result"

    registry = ToolRegistry()
    registry.register_function(first, description="第一条", concurrency_safe=True)
    registry.register_function(second, description="第二条", concurrency_safe=True)
    backend = _FakeChatBackend(
        [
            _ControlledRawStream(
                _tool_chunks(
                    [
                        ("first-1", "first", '{"value":"a"}'),
                        ("second-1", "second", '{"value":"b"}'),
                    ]
                )
            ),
            _ControlledRawStream(_text_chunks("并发完成")),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    runner, manager, broker, gateway, _ = _build_system(
        tmp_path,
        provider,
        registry=registry,
    )
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(
                input="并发读取",
                run_id="run-parallel",
                session_id="session-system",
            )
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await asyncio.wait_for(second_started.wait(), timeout=1)
    subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-parallel",
            scope="run",
            scope_id="run-parallel",
            cursor=LiveCursor(
                stream_epoch=broker.current_epoch(),
                scope="run",
                scope_id="run-parallel",
                after_live_sequence=0,
            ),
        )
    )

    second_release.set()
    await asyncio.wait_for(second_finished.wait(), timeout=1)
    assert not task.done()
    first_release.set()
    result = await task
    items = await _take_until(subscription, _is_live_kind("run.terminal"))

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert body_order == ["second:b", "first:a"]
    records = runner.list_tool_calls("run-parallel")
    assert [(record.ordinal, record.tool_call_id) for record in records] == [
        (1, "first-1"),
        (2, "second-1"),
    ]
    completed = [
        item.payload["tool_call_id"]
        for item in _live_envelopes(items)
        if item.kind == "tool.completed"
    ]
    assert completed == ["first-1", "second-1"]
    tool_result_ids = [
        block.tool_use_id
        for message in runner.get_session("session-system").messages
        for block in message.tool_results
    ]
    assert tool_result_ids == ["first-1", "second-1"]

    await subscription.aclose()
    await manager.close()
    broker.close()


@pytest.mark.asyncio
async def test_slow_consumer_isolated_while_normal_consumer_and_run_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """慢订阅仅收到 gap/terminal，正常订阅与 run 均继续完成。"""
    raw = _ControlledRawStream(
        _text_chunks(*("片段" for _ in range(12))),
        gated_indexes=(12,),
    )
    backend = _FakeChatBackend([raw])
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    runner, manager, broker, gateway, _ = _build_system(
        tmp_path,
        provider,
        subscription_capacity=7,
    )
    slow = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-slow",
            scope="session",
            scope_id="session-system",
        )
    )
    normal = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-normal",
            scope="session",
            scope_id="session-system",
        )
    )
    run_task = asyncio.create_task(
        runner.start(
            AgentRunRequest(
                input="测试慢消费者",
                run_id="run-slow",
                session_id="session-system",
            )
        )
    )
    await asyncio.wait_for(raw.waiting[12].wait(), timeout=1)
    partial_items = await _take_until(
        normal,
        lambda item: (
            isinstance(item, LiveEnvelope)
            and item.kind == "model.block.delta"
            and item.payload.get("snapshot") == "片段" * 12
        ),
    )
    raw.gates[12].set()
    result = await run_task
    normal_items = partial_items + await _take_until(
        normal,
        _is_live_kind("run.terminal"),
    )
    slow_items = await _take_until(
        slow,
        lambda item: isinstance(item, SubscriptionTerminal),
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert any(item.kind == "run.terminal" for item in _live_envelopes(normal_items))
    gaps = [item for item in slow_items if isinstance(item, ReplayGap)]
    terminals = [item for item in slow_items if isinstance(item, SubscriptionTerminal)]
    assert len(gaps) == 1 and gaps[0].reason == "slow_consumer"
    assert len(terminals) == 1 and terminals[0].reason == "slow_consumer"
    assert all(len(ring) <= 512 for ring in broker._rings.values())
    assert slow._subscription._data_count <= 7
    assert normal._subscription._data_count <= 7

    await slow.aclose()
    await normal.aclose()
    await manager.close()
    broker.close()


@pytest.mark.asyncio
async def test_transport_disconnect_keeps_run_active_and_cancel_command_stops_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE 断线只关闭订阅；下一 run 只有显式命令才形成 durable cancel。"""
    disconnect_raw = _ControlledRawStream(
        _text_chunks("断线后仍完成"),
        gated_indexes=(0, 1),
    )
    cancel_raw = _ControlledRawStream(
        _text_chunks("不会完成"),
        gated_indexes=(0,),
    )
    backend = _FakeChatBackend([disconnect_raw, cancel_raw])
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    runner, manager, broker, gateway, _ = _build_system(tmp_path, provider)
    manager_events = manager.events()

    first = await manager.submit("保持运行")
    await asyncio.wait_for(disconnect_raw.waiting[0].wait(), timeout=1)
    subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-disconnect",
            scope="run",
            scope_id=first.run_id,
        )
    )
    sse = SSEAdapter(heartbeat_interval_s=10).stream(subscription)
    disconnect_raw.gates[0].set()
    await asyncio.wait_for(disconnect_raw.waiting[1].wait(), timeout=1)
    frame = await asyncio.wait_for(anext(sse), timeout=1)
    assert frame.startswith(b"id: ")
    await sse.aclose()

    active = runner.get_run(first.run_id)
    assert active.phase is RunPhase.ACTIVE
    assert active.cancellation_requested_at is None
    assert not disconnect_raw.cancelled
    assert subscription._closed

    disconnect_raw.gates[1].set()
    await _take_manager_event(
        manager_events,
        lambda item: (
            isinstance(item, RunEvent)
            and item.run_id == first.run_id
            and item.kind.value == "run.terminal"
        ),
    )
    first_result = runner.get_result(first.run_id)
    assert first_result is not None
    assert first_result.run.stop_reason is RunStopReason.COMPLETED

    second = await manager.submit("显式取消")
    await asyncio.wait_for(cancel_raw.waiting[0].wait(), timeout=1)
    cancel_receipt = await gateway.handle(
        CancelCommand(request_id="cancel-run", reason="用户显式取消")
    )
    assert isinstance(cancel_receipt, CancelAccepted)
    assert cancel_receipt.run.run_id == second.run_id
    await asyncio.wait_for(cancel_raw.cancelled_event.wait(), timeout=1)
    await _take_manager_event(
        manager_events,
        lambda item: (
            isinstance(item, RunEvent)
            and item.run_id == second.run_id
            and item.kind.value == "run.terminal"
        ),
    )
    second_result = runner.get_result(second.run_id)
    assert second_result is not None
    assert second_result.run.stop_reason is RunStopReason.CANCELLED
    assert second_result.run.cancellation_reason == "用户显式取消"

    await manager_events.aclose()
    await manager.close()
    broker.close()


@pytest.mark.asyncio
async def test_busy_submission_remains_on_manager_stream_and_broker_side_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway 不消费 manager stream，busy submission 在两条既有通道各出现一次。"""
    first_raw = _ControlledRawStream(_text_chunks("第一轮"), gated_indexes=(0,))
    second_raw = _ControlledRawStream(_text_chunks("第二轮"))
    backend = _FakeChatBackend([first_raw, second_raw])
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    _, manager, broker, gateway, _ = _build_system(tmp_path, provider)
    manager_events = manager.events()
    subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-side-channel",
            scope="session",
            scope_id="session-system",
        )
    )

    await manager.submit("第一轮")
    await asyncio.wait_for(first_raw.waiting[0].wait(), timeout=1)
    pending = await manager.submit("第二轮", mode="follow_up")
    original = await _take_manager_event(
        manager_events,
        lambda item: (
            isinstance(item, SubmissionEvent) and item.submission_id == pending.submission_id
        ),
    )
    projected_items = await _take_until(
        subscription,
        lambda item: (
            isinstance(item, LiveEnvelope)
            and item.kind == "submission.pending"
            and item.payload.get("submission_id") == pending.submission_id
        ),
    )

    assert isinstance(original, SubmissionEvent)
    assert original.state == "pending"
    assert pending.state == "pending"
    projected = [
        item
        for item in _live_envelopes(projected_items)
        if item.payload.get("submission_id") == pending.submission_id
    ]
    assert len(projected) == 1
    ring_matches = [
        stored.envelope
        for stored in broker._rings[("session", "session-system")]
        if stored.envelope.payload.get("submission_id") == pending.submission_id
    ]
    assert len(ring_matches) == 1

    first_raw.gates[0].set()
    await _take_manager_event(
        manager_events,
        lambda item: (
            isinstance(item, RunEvent)
            and item.run_id == pending.run_id
            and item.kind.value == "run.terminal"
        ),
    )
    await subscription.aclose()
    await manager_events.aclose()
    await manager.close()
    broker.close()


@pytest.mark.asyncio
async def test_sqlite_restart_uses_new_epoch_and_per_run_durable_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart 后 live cursor 明示 epoch gap，durable facts 仅按 known run cursor 恢复。"""
    success_raw = _ControlledRawStream(_text_chunks("SQLite", "完成"))
    failed_raw = _ControlledRawStream(_text_chunks("sqlite-failure-live-only", finish=False))
    waiting_raw = _ControlledRawStream(
        _tool_chunks([("write-1", "write_secret", '{"value":"persisted"}')])
    )
    backend = _FakeChatBackend([success_raw, failed_raw, waiting_raw])
    monkeypatch.setattr(litellm, "acompletion", backend)
    provider = _RecordingStreamingProvider()
    policy = _RecordingPermissionPolicy()
    tool = _SensitiveWriteTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    database = tmp_path / "streaming-lifecycle.db"
    first_store = SQLiteStore(database)
    first_runner, first_manager, first_broker, _, _ = _build_system(
        tmp_path,
        provider,
        store=first_store,
        registry=registry,
        permission_policy=policy,
    )
    completed = await first_runner.start(
        AgentRunRequest(
            input="SQLite 成功",
            run_id="run-sqlite-complete",
            session_id="session-system",
        )
    )
    failed = await first_runner.start(
        AgentRunRequest(
            input="SQLite 失败",
            run_id="run-sqlite-failed",
            session_id="session-system",
        )
    )
    waiting = await first_runner.start(
        AgentRunRequest(
            input="SQLite 等待",
            run_id="run-sqlite-waiting",
            session_id="session-system",
        )
    )
    assert completed.run.stop_reason is RunStopReason.COMPLETED
    assert failed.run.stop_reason is RunStopReason.FAILED
    assert waiting.run.phase is RunPhase.WAITING
    old_epoch = first_broker.current_epoch()
    old_cursor = LiveCursor(
        stream_epoch=old_epoch,
        scope="session",
        scope_id="session-system",
        after_live_sequence=1,
    )
    completed_events = first_runner.list_events("run-sqlite-complete")

    await first_manager.close()
    first_broker.close()
    restarted_store = SQLiteStore(database)
    restarted_provider = _RecordingStreamingProvider()
    restarted_runner, restarted_manager, restarted_broker, restarted_gateway, _ = _build_system(
        tmp_path,
        restarted_provider,
        store=restarted_store,
        registry=registry,
        permission_policy=policy,
    )
    subscription = restarted_gateway.subscribe(
        SubscribeCommand(
            request_id="restart-subscribe",
            scope="session",
            scope_id="session-system",
            cursor=old_cursor,
            durable_cursors=(
                DurableRunCursor(run_id="run-sqlite-complete", after_sequence=1),
                DurableRunCursor(run_id="run-sqlite-waiting", after_sequence=0),
            ),
        )
    )
    initial = await asyncio.wait_for(anext(subscription), timeout=1)
    gap = await asyncio.wait_for(anext(subscription), timeout=1)

    assert isinstance(initial, DurableSyncItem)
    assert isinstance(gap, ReplayGap) and gap.reason == "epoch_changed"
    assert gap.current_epoch == restarted_broker.current_epoch()
    assert gap.current_epoch != old_epoch
    completed_page, waiting_page = initial.sync.runs
    assert [event.sequence for event in completed_page.events] == [2, 3, 4]
    assert completed_page.next_cursor == DurableRunCursor(
        run_id="run-sqlite-complete",
        after_sequence=4,
    )
    assert waiting_page.result is not None
    assert waiting_page.result.run.phase is RunPhase.WAITING
    assert waiting_page.result.pending_interaction is not None

    seen_sequences = [event.sequence for event in completed_page.events]
    cursor = completed_page.next_cursor
    while cursor is not None:
        sync_receipt = await restarted_gateway.handle(
            SyncCommand(request_id=f"sync-{cursor.after_sequence}", cursors=(cursor,))
        )
        assert isinstance(sync_receipt, SyncAccepted)
        page = sync_receipt.sync.runs[0]
        seen_sequences.extend(event.sequence for event in page.events)
        cursor = page.next_cursor
    assert seen_sequences == [event.sequence for event in completed_events if event.sequence > 1]
    assert len(seen_sequences) == len(set(seen_sequences))
    assert restarted_runner.get_run("run-sqlite-waiting").phase is RunPhase.WAITING
    assert restarted_provider.stream_requests == []

    with sqlite3.connect(database) as connection:
        identity = connection.execute("SELECT component, version FROM lifecycle_schema").fetchall()
        objects = {
            (row[0], row[1])
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        }
        dump = "\n".join(connection.iterdump())
    assert identity == [("agent_lifecycle", 2)]
    expected_tables = {
        "agent_runs",
        "lifecycle_schema",
        "run_activations",
        "run_checkpoints",
        "run_events",
        "run_interactions",
        "run_tool_calls",
        "session_messages",
        "session_run_lanes",
        "sessions",
    }
    assert objects == {("table", table) for table in expected_tables} | {
        ("index", "one_open_interaction_per_run")
    }
    assert "sqlite-failure-live-only" not in dump
    assert "model.block.delta" not in dump
    assert "provider_sequence" not in dump
    assert "stream_epoch" not in dump
    assert old_epoch not in dump
    assert restarted_broker.current_epoch() not in dump

    await subscription.aclose()
    await restarted_manager.close()
    restarted_broker.close()
