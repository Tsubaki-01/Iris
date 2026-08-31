"""StreamingGateway session binding、command 与 durable sync 测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.exceptions import IrisRunConflictError, IrisRunNotFoundError, IrisRunStateError
from iris.harness import AgentRunner, SessionManager, SubmitReceipt
from iris.lifecycle import AgentRunRequest, RunPhase, RunToolCallRecord, ToolCallPhase
from iris.message import (
    ModelBlockDelta,
    ModelBlockRef,
    ModelStreamScope,
    TextBlock,
    ToolUseBlock,
)
from iris.runtime import RuntimeStreamEvent
from iris.store import InMemoryLifecycleStore
from iris.streaming.broker import LiveStreamBroker
from iris.streaming.gateway import StreamingGateway
from iris.streaming.models import (
    CancelAccepted,
    CancelCommand,
    CommandRejected,
    DurableRunCursor,
    DurableSyncItem,
    LiveEnvelope,
    ResumeAccepted,
    ResumeCommand,
    SubmitAccepted,
    SubmitCommand,
    SubscribeCommand,
    SyncAccepted,
    SyncCommand,
)
from iris.tools import (
    ToolArtifact,
    ToolCapability,
    ToolErrorInfo,
    ToolRegistry,
    ToolResult,
)
from tests.harness.fakes import StaticProvider, build_runtime, text_response, tool_response


async def _completed_runner(tmp_path: Path) -> tuple[AgentRunner, SessionManager]:
    """构造含一个 terminal run 的 bound runner/manager。"""
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=StaticProvider(text_response("完成"))),
        store=InMemoryLifecycleStore(),
    )
    await runner.start(
        AgentRunRequest(input="开始", run_id="run-own", session_id="session-own")
    )
    return runner, SessionManager(runner, "session-own")


async def _runner_with_sync_facts(
    tmp_path: Path,
) -> tuple[AgentRunner, SessionManager]:
    """构造 terminal、waiting 与 cross-session durable facts。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                text_response("第一轮完成"),
                tool_response(
                    ToolUseBlock(
                        id="write-1",
                        name="write",
                        input={"value": "secret"},
                    )
                ),
                text_response("跨 session 完成"),
            ),
        ),
        store=InMemoryLifecycleStore(),
    )
    await runner.start(
        AgentRunRequest(input="第一轮", run_id="run-terminal", session_id="session-own")
    )
    waiting = await runner.start(
        AgentRunRequest(input="需要写入", run_id="run-waiting", session_id="session-own")
    )
    assert waiting.run.phase is RunPhase.WAITING
    await runner.start(
        AgentRunRequest(input="跨域", run_id="run-cross", session_id="session-other")
    )
    return runner, SessionManager(runner, "session-own")


def _broker() -> LiveStreamBroker:
    return LiveStreamBroker(replay_capacity_per_scope=32, subscription_capacity=16)


def _model_delta(*, channel: str, value: str, sequence: int) -> RuntimeStreamEvent:
    scope = ModelStreamScope(
        model_stream_id="stream-1",
        provider="fake",
        model="fake-model",
        attempt=1,
    )
    block_kind = (
        "thinking" if channel == "thinking" else "tool_call" if channel != "text" else "text"
    )
    return RuntimeStreamEvent(
        kind="model.event",
        run_id="run-live",
        session_id="session-own",
        activation_id="activation-live",
        step_index=0,
        model_event=ModelBlockDelta(
            scope=scope,
            sequence=sequence,
            occurred_at=datetime.now(UTC),
            block=ModelBlockRef(
                index=0,
                block_id="block-1",
                kind=block_kind,
                tool_call_id="tool-stream" if block_kind == "tool_call" else None,
            ),
            channel=channel,
            delta=value,
            snapshot=value,
        ),
    )


@pytest.mark.asyncio
async def test_constructor_validates_binding_and_capacity_without_reads(
    tmp_path: Path,
) -> None:
    runner, manager = await _completed_runner(tmp_path)
    broker = _broker()

    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=broker,
        session_id="  session-own  ",
        durable_page_size=2,
    )

    assert gateway.session_id == "session-own"
    with pytest.raises(IrisRunStateError):
        StreamingGateway(
            runner=runner,
            manager=manager,
            broker=broker,
            session_id=" ",
            durable_page_size=2,
        )
    with pytest.raises(IrisRunConflictError):
        StreamingGateway(
            runner=runner,
            manager=manager,
            broker=broker,
            session_id="session-other",
            durable_page_size=2,
        )
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(IrisRunStateError):
            StreamingGateway(
                runner=runner,
                manager=manager,
                broker=broker,
                session_id="session-own",
                durable_page_size=invalid,
            )
    await manager.close()


@pytest.mark.asyncio
async def test_subscribe_enforces_session_and_run_binding(tmp_path: Path) -> None:
    runner, manager = await _runner_with_sync_facts(tmp_path)
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
    )

    session_sub = gateway.subscribe(
        SubscribeCommand(request_id="sub-session", scope="session", scope_id="session-own")
    )
    run_sub = gateway.subscribe(
        SubscribeCommand(request_id="sub-run", scope="run", scope_id="run-terminal")
    )

    with pytest.raises(IrisRunConflictError):
        gateway.subscribe(
            SubscribeCommand(request_id="bad-session", scope="session", scope_id="session-other")
        )
    with pytest.raises(IrisRunConflictError):
        gateway.subscribe(
            SubscribeCommand(request_id="bad-run", scope="run", scope_id="run-cross")
        )
    with pytest.raises(IrisRunNotFoundError):
        gateway.subscribe(
            SubscribeCommand(request_id="missing", scope="run", scope_id="missing-run")
        )
    await session_sub.aclose()
    await run_sub.aclose()
    await manager.close()


@pytest.mark.asyncio
async def test_subscribe_yields_durable_sync_before_live(tmp_path: Path) -> None:
    runner, manager = await _runner_with_sync_facts(tmp_path)
    broker = _broker()
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=broker,
        session_id="session-own",
        durable_page_size=2,
    )
    subscription = gateway.subscribe(
        SubscribeCommand(
            request_id="subscribe-sync",
            scope="session",
            scope_id="session-own",
            durable_cursors=(DurableRunCursor(run_id="run-terminal", after_sequence=0),),
        )
    )
    broker.publish(
        RuntimeStreamEvent(
            kind="model.step.started",
            run_id="run-live",
            session_id="session-own",
            activation_id="activation-live",
            step_index=0,
        )
    )

    first = await anext(subscription)
    second = await anext(subscription)

    assert isinstance(first, DurableSyncItem)
    assert first.sync.runs[0].run.run_id == "run-terminal"
    assert isinstance(second, LiveEnvelope)
    await subscription.aclose()
    await manager.close()


@pytest.mark.asyncio
async def test_handle_routes_commands_and_does_not_deduplicate_request_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, manager = await _completed_runner(tmp_path)
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
    )
    result = runner.get_result("run-own")
    assert result is not None
    calls: list[tuple[str, object]] = []

    async def submit(input: str, *, mode: str | None = None, options: object = None):
        calls.append(("submit", (input, mode, options)))
        return SubmitReceipt(
            submission_id=f"submission-{len(calls)}",
            run_id="run-own",
            mode=None,
            state="delivered",
        )

    async def resume(*, interaction_id: str, response: object):
        calls.append(("resume", (interaction_id, response)))
        return result

    async def interrupt(*, reason: str | None = None):
        calls.append(("cancel", reason))
        return result.run

    def forbidden_events():
        raise AssertionError("gateway 不得消费 manager.events()")

    monkeypatch.setattr(manager, "submit", submit)
    monkeypatch.setattr(manager, "resume", resume)
    monkeypatch.setattr(manager, "interrupt", interrupt)
    monkeypatch.setattr(manager, "events", forbidden_events)
    submit_command = SubmitCommand(request_id="same-id", input="hello")

    first = await gateway.handle(submit_command)
    second = await gateway.handle(submit_command)
    resumed = await gateway.handle(
        ResumeCommand(
            request_id="resume-id",
            interaction_id="interaction-1",
            response={"kind": "question", "answer": "ok"},
        )
    )
    cancelled = await gateway.handle(CancelCommand(request_id="cancel-id", reason="stop"))
    synced = await gateway.handle(
        SyncCommand(
            request_id="sync-id",
            cursors=(DurableRunCursor(run_id="run-own", after_sequence=0),),
        )
    )

    assert isinstance(first, SubmitAccepted)
    assert isinstance(second, SubmitAccepted)
    assert [name for name, _ in calls].count("submit") == 2
    assert isinstance(resumed, ResumeAccepted)
    assert isinstance(cancelled, CancelAccepted)
    assert isinstance(synced, SyncAccepted)
    await manager.close()


@pytest.mark.asyncio
async def test_handle_maps_iris_error_without_context_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, manager = await _completed_runner(tmp_path)
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
    )

    async def fail(*args: object, **kwargs: object):
        del args, kwargs
        raise IrisRunStateError("当前状态不接受命令", secret="not-for-wire")

    monkeypatch.setattr(manager, "submit", fail)

    receipt = await gateway.handle(SubmitCommand(request_id="request-1", input="hello"))

    assert isinstance(receipt, CommandRejected)
    assert receipt.code == "RUN_STATE_ERROR"
    assert receipt.message == "当前状态不接受命令"
    assert "not-for-wire" not in receipt.model_dump_json()
    await manager.close()


@pytest.mark.asyncio
async def test_handle_maps_unexpected_error_without_detail_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner, manager = await _completed_runner(tmp_path)
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
    )

    async def fail(*args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError("secret payload detail")

    monkeypatch.setattr(manager, "submit", fail)

    receipt = await gateway.handle(SubmitCommand(request_id="request-1", input="hello"))

    assert isinstance(receipt, CommandRejected)
    assert receipt.code == "INTERNAL_ERROR"
    assert receipt.message == "命令处理失败"
    assert "secret payload detail" not in caplog.text
    await manager.close()


@pytest.mark.asyncio
async def test_durable_sync_pages_in_input_order_and_redacts_tool_arguments(
    tmp_path: Path,
) -> None:
    runner, manager = await _runner_with_sync_facts(tmp_path)
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
    )

    sync = gateway.durable_sync(
        (
            DurableRunCursor(run_id="run-waiting", after_sequence=0),
            DurableRunCursor(run_id="run-terminal", after_sequence=0),
        )
    )

    assert [page.run.run_id for page in sync.runs] == ["run-waiting", "run-terminal"]
    waiting = sync.runs[0]
    assert len(waiting.events) == 2
    assert waiting.next_cursor == DurableRunCursor(
        run_id="run-waiting",
        after_sequence=waiting.events[-1].sequence,
    )
    assert waiting.result is not None and waiting.result.pending_interaction is not None
    assert waiting.result.pending_interaction.request.tool_call.arguments == {}
    assert waiting.result.pending_interaction.request.tool_call.workspace_root == "<redacted>"
    assert waiting.result.assistant_message is not None
    assert waiting.result.assistant_message.tool_calls[0].input == {}
    assert waiting.tool_calls[0].arguments == {}

    allowed = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
        allow_tool_arguments=True,
    ).durable_sync((DurableRunCursor(run_id="run-waiting", after_sequence=0),))
    assert allowed.runs[0].tool_calls[0].arguments == {"value": "secret"}
    assert (
        allowed.runs[0].result.pending_interaction.request.tool_call.arguments
        == {"value": "secret"}
    )
    assert (
        allowed.runs[0].result.pending_interaction.request.tool_call.workspace_root
        == "<redacted>"
    )
    assert allowed.runs[0].result.assistant_message.tool_calls[0].input == {
        "value": "secret"
    }
    await manager.close()


@pytest.mark.asyncio
async def test_durable_sync_rejects_missing_or_cross_session_without_partial_page(
    tmp_path: Path,
) -> None:
    runner, manager = await _runner_with_sync_facts(tmp_path)
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
    )

    with pytest.raises(IrisRunConflictError):
        gateway.durable_sync(
            (
                DurableRunCursor(run_id="run-terminal", after_sequence=0),
                DurableRunCursor(run_id="run-cross", after_sequence=0),
            )
        )
    with pytest.raises(IrisRunNotFoundError):
        gateway.durable_sync((DurableRunCursor(run_id="missing", after_sequence=0),))
    await manager.close()


@pytest.mark.asyncio
async def test_durable_sync_removes_internal_tool_result_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, manager = await _completed_runner(tmp_path)
    now = datetime.now(UTC)
    record = RunToolCallRecord(
        run_id="run-own",
        step_index=0,
        ordinal=1,
        tool_call_id="tool-1",
        tool_name="probe",
        arguments={"path": "secret.txt"},
        fingerprint="a" * 64,
        phase=ToolCallPhase.COMMITTED,
        claim_activation_id="activation-1",
        result=ToolResult(
            tool_use_id="tool-1",
            tool_name="probe",
            content=[TextBlock(text="safe output")],
            is_error=True,
            error=ToolErrorInfo(
                code="FAILED",
                message="safe message",
                details={"traceback": "secret"},
            ),
            data={"secret": "data"},
            artifact=ToolArtifact(
                path=tmp_path / "secret.bin",
                mime_type="application/octet-stream",
                size_bytes=6,
                preview="preview",
            ),
            stats={"host": "secret"},
            metadata={"trace": "secret"},
        ),
        version=3,
        created_at=now,
        updated_at=now,
        claimed_at=now,
        committed_at=now,
    )
    monkeypatch.setattr(runner, "list_tool_calls", lambda run_id: [record])
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=_broker(),
        session_id="session-own",
        durable_page_size=2,
    )

    sync = gateway.durable_sync((DurableRunCursor(run_id="run-own", after_sequence=0),))
    filtered = sync.runs[0].tool_calls[0]

    assert filtered.arguments == {}
    assert filtered.result is not None
    assert filtered.result.data == {}
    assert filtered.result.artifact is None
    assert filtered.result.stats == {}
    assert filtered.result.metadata == {}
    assert filtered.result.error is not None
    assert filtered.result.error.details == {}
    await manager.close()


@pytest.mark.asyncio
async def test_subscription_filters_sensitive_partials_by_default_and_allows_opt_in(
    tmp_path: Path,
) -> None:
    runner, manager = await _completed_runner(tmp_path)
    broker = _broker()
    default_gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=broker,
        session_id="session-own",
        durable_page_size=2,
    )
    default_sub = default_gateway.subscribe(
        SubscribeCommand(request_id="default", scope="session", scope_id="session-own")
    )
    broker.publish(_model_delta(channel="thinking", value="hidden thought", sequence=1))
    broker.publish(_model_delta(channel="tool_name", value="secret_tool", sequence=2))
    broker.publish(_model_delta(channel="tool_arguments", value='{"secret":1}', sequence=3))
    broker.publish(
        RuntimeStreamEvent(
            kind="tool.started",
            run_id="run-live",
            session_id="session-own",
            activation_id="activation-live",
            step_index=0,
            tool_call_id="tool-1",
            tool_name="secret_tool",
            tool_ordinal=1,
        )
    )
    broker.publish(_model_delta(channel="text", value="visible", sequence=4))

    tool_item = await anext(default_sub)
    text_item = await anext(default_sub)

    assert isinstance(tool_item, LiveEnvelope)
    assert "tool_name" not in tool_item.payload
    assert isinstance(text_item, LiveEnvelope)
    assert text_item.payload["channel"] == "text"
    await default_sub.aclose()

    allowed_gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=broker,
        session_id="session-own",
        durable_page_size=2,
        allow_thinking=True,
        allow_tool_arguments=True,
    )
    allowed_sub = allowed_gateway.subscribe(
        SubscribeCommand(request_id="allowed", scope="session", scope_id="session-own")
    )
    broker.publish(_model_delta(channel="thinking", value="visible thought", sequence=5))
    broker.publish(_model_delta(channel="tool_name", value="visible_tool", sequence=6))
    broker.publish(_model_delta(channel="tool_arguments", value='{"ok":1}', sequence=7))
    broker.publish(
        RuntimeStreamEvent(
            kind="tool.started",
            run_id="run-live",
            session_id="session-own",
            activation_id="activation-live",
            step_index=0,
            tool_call_id="tool-2",
            tool_name="visible_tool",
            tool_ordinal=1,
        )
    )

    allowed_items = [await anext(allowed_sub) for _ in range(4)]

    assert [item.payload.get("channel") for item in allowed_items[:3]] == [
        "thinking",
        "tool_name",
        "tool_arguments",
    ]
    assert allowed_items[3].payload["tool_name"] == "visible_tool"
    await allowed_sub.aclose()
    await manager.close()
