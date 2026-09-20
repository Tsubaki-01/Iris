"""真实 child Runner 与 shared-store 委派的确定性集成测试。"""

import asyncio
import re
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from iris.agents import AgentConfig, ToolsConfig, build_tool_registry
from iris.exceptions import (
    IrisRunObservationTimeoutError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.harness import AgentRunner, ChildProviderFactory, SessionManager
from iris.hitl import (
    InteractionStatus,
    PermissionInteractionResponse,
    QuestionInteractionResponse,
    make_call_fingerprint,
)
from iris.lifecycle import (
    AdmitChildRun,
    AgentRunOptions,
    AgentRunRequest,
    CommitModelStep,
    CommitRunInput,
    CreateRun,
    FinishRun,
    ReserveModelStep,
    RunCheckpoint,
    RunErrorInfo,
    RunLimits,
    RunPhase,
    RunResult,
    RunStopReason,
    RuntimeExecutionOptions,
    RunToolCallRecord,
    RunUsage,
    ToolCallPhase,
    ToolErrorPolicy,
)
from iris.memory import MemoryConfig, MemoryService
from iris.message import (
    LLMRequest,
    ModelResponseCompleted,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    Msg,
    ToolUseBlock,
)
from iris.providers import CompletionProvider
from iris.runtime import RuntimeCursor, RuntimeStreamEvent
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import PreparedToolCall, ToolRegistry, ToolResult
from iris.tools._paths import safe_path_segment
from iris.tools.permissions import DefaultPermissionPolicy, PermissionDecision, PermissionEffect
from iris.tools.subagent import ChildWaiting, SubagentExecutionOutcome

from .fakes import (
    BlockingProvider,
    CountingAgentRuntime,
    FrozenClock,
    RecordingPublisher,
    StaticProvider,
    text_response,
    tool_response,
)


def _parent_provider() -> StaticProvider:
    return StaticProvider(
        tool_response(
            ToolUseBlock(
                id="delegate",
                name="subagent",
                input={"prompt": "Child task"},
            )
        ),
        text_response("Parent complete"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("readme", ["README.md", "README.en.md"])
async def test_documented_subagent_configs_run_through_public_runner(
    tmp_path: Path, readme: str
) -> None:
    source = Path(__file__).resolve().parents[2] / "src" / "iris" / "harness" / readme
    for filename, content in re.findall(
        r"```yaml\n# ([^\n]+)\n(.*?)```", source.read_text(encoding="utf-8"), re.DOTALL
    ):
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    runner = AgentRunner.from_config_path(
        tmp_path / "agent.yaml",
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                ),
                text_response("Child done"),
            )
        ),
    )
    runtime = CountingAgentRuntime(runner.runtime)
    runner.runtime = runtime
    waiting = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    result = await runner.resume(
        "parent",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="Continue"),
    )
    assert result.assistant_message.text == "Parent complete"
    assert runner.store.load_tool_call("parent", "delegate").result.model_content == "Child done"
    assert [activation.kind for activation in runtime.activations] == ["start", "resume"]
    assert all(activation.run_input == "Start" for activation in runtime.activations)
    assert all(activation.initial_session_message_count == 0 for activation in runtime.activations)


class RecordingInMemoryLifecycleStore(InMemoryLifecycleStore):
    """记录实际 finish 顺序以验证 child-first。"""

    def __init__(self) -> None:
        super().__init__()
        self.finished: list[str] = []

    def finish_run(self, command: FinishRun) -> object:
        result = super().finish_run(command)
        self.finished.append(command.run_id)
        return result


class StreamingStaticProvider(StaticProvider):
    """沿用静态响应队列，仅补充 parent live 测试所需的 typed stream。"""

    async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        response = await self.complete(request)
        scope = ModelStreamScope(
            model_stream_id=response.id, provider=response.provider, model=response.model, attempt=1
        )
        now = FrozenClock().now()
        yield ModelResponseStarted(
            scope=scope, sequence=1, occurred_at=now, response_id=response.id
        )
        yield ModelResponseCompleted(
            scope=scope,
            sequence=2,
            occurred_at=now,
            response=response,
            semantic_output_emitted=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", ["active", "proxy", "error", "linked"])
async def test_subagent_live_plane_contains_parent_facts_only(tmp_path: Path, flow: str) -> None:
    publisher = RecordingPublisher()
    parent = StreamingStaticProvider(
        *(
            []
            if flow == "linked"
            else [
                tool_response(
                    ToolUseBlock(id="delegate", name="subagent", input={"prompt": "Child task"})
                )
            ]
        ),
        text_response("Parent done"),
    )
    child = StaticProvider(
        *(
            [
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                )
            ]
            if flow == "proxy"
            else []
        ),
        text_response("Child done"),
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=parent,
        child_provider_factory=ChildProviders(child),
        live_publisher=publisher,
    )
    if flow == "linked":
        await _dispatch(runner, _prepare_parent(runner))
        result = await runner.recover(
            "parent", expected_activation_id=runner.store.load_run("parent").current_activation_id
        )
    else:
        if flow == "error":
            (tmp_path / "child.yaml").write_text("[broken", encoding="utf-8")
        result = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
        if flow == "proxy":
            assert result.pending_interaction is not None, result.error
            result = await runner.resume(
                "parent",
                interaction_id=result.pending_interaction.interaction_id,
                response=QuestionInteractionResponse(answer="Continue"),
            )
    assert result.run.stop_reason == RunStopReason.COMPLETED, result.error
    assert all(fact.run_id == "parent" for fact in publisher.facts)
    starts = [
        fact
        for fact in publisher.facts
        if isinstance(fact, RuntimeStreamEvent) and fact.kind == "tool.started"
    ]
    finals = [
        fact
        for fact in publisher.facts
        if isinstance(fact, RuntimeStreamEvent) and fact.kind == "tool.completed"
    ]
    assert len(starts) == int(flow != "linked")
    assert len(finals) == 1 and finals[0].tool_call_id == "delegate"
    assert finals[0].tool_result.is_error == (flow == "error")
    if flow == "proxy":
        assert any(fact.kind == "interaction.suspended" for fact in publisher.facts)
        assert any(fact.kind == "interaction.resolved" for fact in publisher.facts)


@pytest.mark.asyncio
async def test_active_parent_deadline_drains_child_before_parent(tmp_path: Path) -> None:
    clock = FrozenClock()
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        store=store,
        clock=clock,
        child_provider_factory=ChildProviders(BlockingProvider()),
    )
    result = await asyncio.wait_for(
        runner.start(
            AgentRunRequest(input="Start", run_id="parent"),
            options=AgentRunOptions(
                limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=0.02))
            ),
        ),
        timeout=1,
    )
    child_id = store.load_subagent_link("parent", "delegate").child_run_id
    assert result.run.stop_reason == RunStopReason.DEADLINE_EXCEEDED
    assert store.finished == [child_id, "parent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("child_waits", [False, True])
async def test_remote_parent_cancel_observed_before_child_result_commit(
    tmp_path: Path, child_waits: bool
) -> None:
    path = _write_configs(tmp_path)
    child = BlockingProvider(
        tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"}))
        if child_waits
        else text_response("Child done")
    )
    runner = AgentRunner.from_config_path(
        path, provider=_parent_provider(), child_provider_factory=ChildProviders(child)
    )
    task = asyncio.create_task(runner.start(AgentRunRequest(input="Start", run_id="parent")))
    await asyncio.wait_for(child.started.wait(), timeout=1)
    observer = AgentRunner.from_config_path(path, provider=StaticProvider(), store=runner.store)
    observer.request_cancel("parent")
    child.release.set()
    result = await task
    assert result.run.stop_reason == RunStopReason.CANCELLED
    child_id = runner.store.load_subagent_link("parent", "delegate").child_run_id
    assert runner.store.load_run(child_id).phase == RunPhase.TERMINAL


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_cancelling_parent_settles_linked_child_first(tmp_path: Path, waiting: bool) -> None:
    child_provider = (
        StaticProvider(
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
            )
        )
        if waiting
        else BlockingProvider()
    )
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        store=store,
        child_provider_factory=ChildProviders(child_provider),
    )
    running = asyncio.create_task(runner.start(AgentRunRequest(input="Start", run_id="parent")))
    if waiting:
        await running
    else:
        await asyncio.wait_for(child_provider.started.wait(), timeout=1)
    child_id = store.load_subagent_link("parent", "delegate").child_run_id
    snapshot = runner.request_cancel("parent")
    assert snapshot.phase == (RunPhase.WAITING if waiting else RunPhase.ACTIVE)
    result = await runner.cancel("parent", settlement_timeout=1)
    assert result.run.stop_reason == RunStopReason.CANCELLED
    assert store.load_run(child_id).phase == RunPhase.TERMINAL
    assert store.finished[-1] == "parent"
    if not waiting:
        assert store.finished == [child_id, "parent"]
        assert await running == result


@pytest.mark.asyncio
async def test_active_parent_cancel_drains_claimed_async_child_tool(
    tmp_path: Path, blocking_child_tool: "BlockingChildTool"
) -> None:
    """普通 child 工具先收到 Python 取消并完成清理，parent 才能结算。"""
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        store=store,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(ToolUseBlock(id="blocked", name="blocking_child", input={})),
                text_response("Child done"),
            )
        ),
    )
    running = asyncio.create_task(runner.start(AgentRunRequest(input="Start", run_id="parent")))
    tasks: list[asyncio.Task[object]] = [running]
    try:
        await asyncio.wait_for(blocking_child_tool.entered.wait(), timeout=1)
        child_id = store.load_subagent_link("parent", "delegate").child_run_id
        assert store.load_run("parent").phase == RunPhase.ACTIVE
        assert store.load_tool_call(child_id, "blocked").phase == ToolCallPhase.CLAIMED

        cancelling = asyncio.create_task(runner.cancel("parent"))
        tasks.append(cancelling)
        await asyncio.wait_for(blocking_child_tool.cancelled.wait(), timeout=1)
        assert not blocking_child_tool.finished.is_set()
        assert not running.done() and not cancelling.done()
        assert store.load_run("parent").phase == RunPhase.ACTIVE
        assert store.load_tool_call(child_id, "blocked").phase == ToolCallPhase.CLAIMED

        blocking_child_tool.cleanup_release.set()
        result = await asyncio.wait_for(asyncio.shield(cancelling), timeout=1)
        assert blocking_child_tool.finished.is_set()
        assert not blocking_child_tool.body_release.is_set()
        assert result == await asyncio.wait_for(asyncio.shield(running), timeout=1)
        assert store.load_tool_call(child_id, "blocked").phase == ToolCallPhase.OUTCOME_UNKNOWN
        assert store.load_run(child_id).stop_reason == RunStopReason.OUTCOME_UNKNOWN
        assert result.run.stop_reason == RunStopReason.CANCELLED
        assert store.finished == [child_id, "parent"]
    finally:
        blocking_child_tool.body_release.set()
        blocking_child_tool.cleanup_release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_active_outer_timeout_settles_child_and_parent_continues(tmp_path: Path) -> None:
    child = BlockingProvider()
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        store=store,
        child_provider_factory=ChildProviders(child),
    )
    result = await asyncio.wait_for(
        runner.start(
            AgentRunRequest(input="Start", run_id="parent"),
            options=AgentRunOptions(runtime=RuntimeExecutionOptions(tool_timeout_seconds=0.02)),
        ),
        timeout=1,
    )
    child_id = store.load_subagent_link("parent", "delegate").child_run_id
    assert store.finished == [child_id, "parent"]
    assert result.run.stop_reason == RunStopReason.COMPLETED
    assert store.load_tool_call("parent", "delegate").result.error.code == "SUBAGENT_TIMEOUT"


@pytest.mark.asyncio
async def test_repeated_active_cancel_does_not_interrupt_child_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = BlockingProvider()
    factory = ChildProviders(child)
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        store=store,
        child_provider_factory=factory,
    )
    task = asyncio.create_task(runner.start(AgentRunRequest(input="Start", run_id="parent")))
    await asyncio.wait_for(child.started.wait(), timeout=1)
    started, release = asyncio.Event(), asyncio.Event()
    original = runner._subagent_controller.cancel_linked

    async def delayed(**kwargs: object) -> None:
        started.set()
        await release.wait()
        await original(**kwargs)

    monkeypatch.setattr(runner._subagent_controller, "cancel_linked", delayed)
    runner.request_cancel("parent")
    await asyncio.wait_for(started.wait(), timeout=1)
    runner.request_cancel("parent")
    await asyncio.sleep(0)
    release.set()
    result = await asyncio.wait_for(task, timeout=1)
    child_id = store.load_subagent_link("parent", "delegate").child_run_id
    assert store.finished == [child_id, "parent"]
    assert result.run.stop_reason == RunStopReason.CANCELLED
    assert len(factory.configs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner", ["deadline", "parent_interaction", "outer", "child_interaction", "child_deadline"]
)
async def test_proxy_due_owner_settles_child_before_parent_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    clock = FrozenClock()
    build_start = AgentRunner._build_start_facts

    def child_limits(
        self: AgentRunner, request: AgentRunRequest, *, options: AgentRunOptions | None
    ) -> object:
        if self.runtime.environment.agent_config.name == "researcher":
            options = AgentRunOptions(
                limits=RunLimits(
                    deadline_at=clock.now() + timedelta(seconds=1)
                    if owner == "child_deadline"
                    else None,
                    interaction_timeout_seconds=1 if owner == "child_interaction" else None,
                )
            )
        return build_start(self, request, options=options)

    monkeypatch.setattr(AgentRunner, "_build_start_facts", child_limits)
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        store=store,
        clock=clock,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                )
            )
        ),
    )
    options = AgentRunOptions(
        limits=RunLimits(
            deadline_at=clock.now() + timedelta(seconds=1) if owner == "deadline" else None,
            interaction_timeout_seconds=1 if owner == "parent_interaction" else None,
        ),
        runtime=RuntimeExecutionOptions(tool_timeout_seconds=1 if owner == "outer" else None),
    )
    waiting = await runner.start(AgentRunRequest(input="Start", run_id="parent"), options=options)
    proxy = waiting.pending_interaction
    child_id = proxy.request.subagent_origin.child_run_id
    clock.advance(seconds=2)
    result = await runner.recover("parent")
    assert store.load_run(child_id).phase == RunPhase.TERMINAL
    assert (
        result.run.stop_reason
        == {
            "deadline": RunStopReason.DEADLINE_EXCEEDED,
            "parent_interaction": RunStopReason.INTERACTION_EXPIRED,
            "outer": RunStopReason.COMPLETED,
            "child_interaction": RunStopReason.COMPLETED,
            "child_deadline": RunStopReason.COMPLETED,
        }[owner]
    )
    assert store.load_interaction(proxy.interaction_id).response is None
    if owner in {"outer", "child_interaction", "child_deadline"}:
        assert store.load_tool_call("parent", "delegate").result.error.code == "SUBAGENT_TIMEOUT"
    if owner.startswith("child_"):
        assert store.load_run(child_id).stop_reason == (
            RunStopReason.INTERACTION_EXPIRED
            if owner == "child_interaction"
            else RunStopReason.DEADLINE_EXCEEDED
        )


@pytest.mark.asyncio
async def test_mixed_read_subagent_batch_keeps_serial_order(tmp_path: Path) -> None:
    path = _write_configs(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("tools:\n", "tools:\n  builtin: [file.read]\n"),
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Notes", encoding="utf-8")
    from .fakes import tool_batch_response

    child = BlockingProvider(text_response("Child answer"))
    runner = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(
            tool_batch_response(
                ToolUseBlock(id="before", name="read_file", input={"file_path": "notes.txt"}),
                ToolUseBlock(id="delegate", name="subagent", input={"prompt": "Read the task"}),
                ToolUseBlock(id="after", name="read_file", input={"file_path": "notes.txt"}),
            ),
            text_response("Parent done"),
        ),
        child_provider_factory=ChildProviders(child),
    )
    task = asyncio.create_task(runner.start(AgentRunRequest(input="Start", run_id="parent")))
    try:
        await asyncio.wait_for(child.started.wait(), timeout=1)
        calls = runner.store.list_tool_calls("parent")
        assert [call.phase.value for call in calls] == ["committed", "prepared", "prepared"]
        assert calls[1].claim_activation_id is None
    finally:
        child.release.set()
    completed = await task
    assert completed.run.usage.tool_calls_committed == 3
    assert [call.tool_call_id for call in runner.store.list_tool_calls("parent")] == [
        "before",
        "delegate",
        "after",
    ]


@pytest.mark.asyncio
async def test_fresh_process_cancel_recovers_linked_active_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(),
        store=store,
        child_provider_factory=ChildProviders(StaticProvider()),
    )
    prepared = _prepare_parent(runner)

    async def stop_after_admission(*args: object, **kwargs: object) -> RunResult:
        raise IrisRunPersistenceError("process stopped after admission")

    with monkeypatch.context() as patch:
        patch.setattr(AgentRunner, "_run_admitted_start", stop_after_admission)
        with pytest.raises(IrisRunPersistenceError):
            await _dispatch(runner, prepared)
    child_id = store.load_subagent_link("parent", "delegate").child_run_id
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(),
        store=store,
        child_provider_factory=ChildProviders(StaticProvider()),
    )
    result = await restarted.cancel("parent", settlement_timeout=1)
    assert result.run.stop_reason == RunStopReason.CANCELLED
    child = store.load_run(child_id)
    assert child.stop_reason == RunStopReason.CANCELLED
    assert child.cancellation_requested_at is not None
    assert store.finished == [child_id, "parent"]


@pytest.mark.asyncio
async def test_linked_cancel_budget_covers_child_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                )
            )
        ),
    )
    waiting = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    started, release = asyncio.Event(), asyncio.Event()
    controller = runner._subagent_controller
    original = controller.cancel_linked

    async def delayed(**kwargs: object) -> None:
        started.set()
        await release.wait()
        await original(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "cancel_linked", delayed)
        with pytest.raises(IrisRunObservationTimeoutError):
            await runner.cancel("parent", settlement_timeout=0.02)
    assert started.is_set()
    assert runner.store.load_run("parent").phase == RunPhase.WAITING
    assert runner.store.load_run("parent").cancellation_requested_at is not None
    assert (
        runner.store.load_run(
            waiting.pending_interaction.request.subagent_origin.child_run_id
        ).phase
        == RunPhase.WAITING
    )
    assert (await runner.recover("parent")).run.stop_reason == RunStopReason.CANCELLED


@pytest.mark.asyncio
async def test_active_parent_recovery_enforces_expired_waiting_child_budget(tmp_path: Path) -> None:
    path = _write_configs(tmp_path)
    clock = FrozenClock()
    factory = ChildProviders(
        StaticProvider(
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
            )
        )
    )
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), clock=clock, child_provider_factory=factory
    )
    prepared = _prepare_parent(
        runner, options=AgentRunOptions(runtime=RuntimeExecutionOptions(tool_timeout_seconds=1))
    )
    await _dispatch(runner, prepared)
    clock.advance(seconds=2)
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(text_response("Timeout handled")),
        clock=clock,
        store=runner.store,
        child_provider_factory=factory,
    )
    result = await restarted.recover(
        "parent", expected_activation_id=runner.store.load_run("parent").current_activation_id
    )
    assert result.run.stop_reason == RunStopReason.COMPLETED
    assert runner.store.load_tool_call("parent", "delegate").result.error.code == "SUBAGENT_TIMEOUT"


@pytest.mark.asyncio
@pytest.mark.parametrize("resuming", [False, True])
async def test_manager_interrupt_keeps_child_cleanup_owner_and_blocks_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resuming: bool
) -> None:
    from .test_session_manager import _wait_until

    factory = ChildProviders(
        StaticProvider(
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
            )
        )
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path), provider=_parent_provider(), child_provider_factory=factory
    )
    manager = SessionManager(runner, "managed-session")
    current = await manager.submit("Start")
    await asyncio.wait_for(asyncio.shield(manager._current_task), timeout=1)
    proxy = runner.store.load_result(current.run_id).pending_interaction
    if resuming:
        blocker = BlockingProvider()
        factory.provider = blocker
        await manager.admit_resume(
            interaction_id=proxy.interaction_id,
            response=QuestionInteractionResponse(answer="Continue"),
        )
        await asyncio.wait_for(blocker.started.wait(), timeout=1)
        continuation = manager._current_task
    started, release = asyncio.Event(), asyncio.Event()
    original = runner._subagent_controller.cancel_linked

    async def delayed(**kwargs: object) -> None:
        started.set()
        await release.wait()
        await original(**kwargs)

    monkeypatch.setattr(runner._subagent_controller, "cancel_linked", delayed)
    try:
        snapshot = await asyncio.wait_for(manager.interrupt(), timeout=1)
        assert snapshot.phase == RunPhase.WAITING
        await asyncio.wait_for(started.wait(), timeout=1)
        if resuming:
            assert manager._current_task is continuation
        await asyncio.wait_for(manager.interrupt(), timeout=1)
        follow = await manager.submit("Next run", mode="follow_up")
        assert runner.store.load_run(follow.run_id) is None
        assert runner.store.load_run(current.run_id).phase == RunPhase.WAITING
        release.set()
        await asyncio.wait_for(asyncio.shield(manager._current_task), timeout=1)
        await _wait_until(lambda: runner.store.load_result(follow.run_id) is not None)
        assert runner.store.load_run(current.run_id).stop_reason == RunStopReason.CANCELLED
        assert (
            runner.store.load_run(proxy.request.subagent_origin.child_run_id).phase
            == RunPhase.TERMINAL
        )
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_manager_close_drains_waiting_parent_resume_and_claimed_async_child_tool(
    tmp_path: Path, blocking_child_tool: "BlockingChildTool"
) -> None:
    """WAITING parent 的 close 等待原 child 调用和 managed resume 完整退出。"""
    store = RecordingInMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        store=store,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                ),
                tool_response(ToolUseBlock(id="blocked", name="blocking_child", input={})),
                text_response("Child done"),
            )
        ),
    )
    manager = SessionManager(runner, "managed-session")
    tasks: list[asyncio.Task[object]] = []
    try:
        receipt = await manager.submit("Start")
        initial = manager._current_task
        assert initial is not None
        tasks.append(initial)
        waiting = await asyncio.wait_for(asyncio.shield(initial), timeout=1)
        proxy = waiting.pending_interaction
        assert proxy is not None
        resuming = asyncio.create_task(
            manager.resume(
                interaction_id=proxy.interaction_id,
                response=QuestionInteractionResponse(answer="Continue"),
            )
        )
        tasks.append(resuming)
        await asyncio.wait_for(blocking_child_tool.entered.wait(), timeout=1)
        continuation = manager._current_task
        assert continuation is not None
        tasks.append(continuation)
        child_id = store.load_subagent_link(receipt.run_id, "delegate").child_run_id
        _, child_task = runner._subagent_controller._live_children[child_id]
        tasks.append(child_task)
        assert store.load_run(receipt.run_id).phase == RunPhase.WAITING
        assert store.load_tool_call(child_id, "blocked").phase == ToolCallPhase.CLAIMED

        closing = asyncio.create_task(manager.close(cancel_run=True))
        tasks.append(closing)
        await asyncio.wait_for(blocking_child_tool.cancelled.wait(), timeout=1)
        assert not blocking_child_tool.finished.is_set()
        assert not closing.done() and not continuation.done()
        assert not resuming.done() and not child_task.done()
        assert store.load_run(receipt.run_id).phase == RunPhase.WAITING

        blocking_child_tool.cleanup_release.set()
        await asyncio.wait_for(asyncio.shield(closing), timeout=1)
        assert blocking_child_tool.finished.is_set()
        assert not blocking_child_tool.body_release.is_set()
        assert child_task.done() and continuation.done() and resuming.done()
        assert child_task.result().run.stop_reason == RunStopReason.OUTCOME_UNKNOWN
        assert continuation.result() == resuming.result()
        assert resuming.result().run.stop_reason == RunStopReason.CANCELLED
        assert store.load_tool_call(child_id, "blocked").phase == ToolCallPhase.OUTCOME_UNKNOWN
        assert store.finished == [child_id, receipt.run_id]
    finally:
        blocking_child_tool.body_release.set()
        blocking_child_tool.cleanup_release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
async def test_outer_timeout_anchor_survives_permission_wait_proxy_rebind_and_restart(
    tmp_path: Path,
) -> None:
    path = _write_configs(tmp_path)
    clock = FrozenClock()
    policy = RecordingPermissionPolicy("subagent")
    runner = AgentRunner.from_config_path(
        path,
        provider=_parent_provider(),
        clock=clock,
        permission_policy=policy,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="ask1", name="ask_question", input={"question": "First?"})
                ),
                tool_response(
                    ToolUseBlock(id="ask2", name="ask_question", input={"question": "Second?"})
                ),
            )
        ),
    )
    waiting = await runner.start(
        AgentRunRequest(input="Start", run_id="parent"),
        options=AgentRunOptions(runtime=RuntimeExecutionOptions(tool_timeout_seconds=10)),
    )
    assert runner.store.load_subagent_link("parent", "delegate") is None
    clock.advance(seconds=30)
    first = await runner.resume(
        "parent",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    child_id = first.pending_interaction.request.subagent_origin.child_run_id
    anchor = runner.store.load_run(child_id).created_at
    assert anchor == clock.now()
    clock.advance(seconds=6)
    second = await runner.resume(
        "parent",
        interaction_id=first.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="One"),
    )
    assert second.pending_interaction.expires_at == anchor + timedelta(seconds=10)
    clock.advance(seconds=5)
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(text_response("Timed out child")),
        store=runner.store,
        clock=clock,
        permission_policy=policy,
        child_provider_factory=ChildProviders(StaticProvider()),
    )
    result = await restarted.recover("parent")
    assert result.run.stop_reason == RunStopReason.COMPLETED
    assert restarted.store.load_run(child_id).created_at == anchor
    assert (
        restarted.store.load_tool_call("parent", "delegate").result.error.code == "SUBAGENT_TIMEOUT"
    )


@pytest.mark.asyncio
async def test_subagent_parent_loop_commits_child_result_once(tmp_path: Path) -> None:
    parent_provider = _parent_provider()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=parent_provider,
        child_provider_factory=ChildProviders(StaticProvider(text_response("Child answer"))),
    )
    result = await runner.start(
        AgentRunRequest(input="Start", run_id="parent", session_id="parent-session")
    )
    assert result.run.stop_reason == RunStopReason.COMPLETED
    assert result.assistant_message.text == "Parent complete"
    call = runner.store.load_tool_call("parent", "delegate")
    assert call.phase.value == "committed" and call.claim_activation_id is None
    assert call.result.model_content == "Child answer"
    assert result.run.usage.tool_calls_committed == 1
    assert result.run.usage.total_tokens == 13
    assert runner.store.load_run(call.result.metadata["child_run_id"]).usage.total_tokens == 5


@pytest.mark.asyncio
async def test_subagent_unadmitted_config_error_is_committed_without_claim(tmp_path: Path) -> None:
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(
                    id="delegate", name="subagent", input={"prompt": "x", "agent": "broken"}
                )
            ),
            text_response("Config unavailable"),
        ),
    )
    result = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    assert result.run.stop_reason == RunStopReason.COMPLETED
    call = runner.store.load_tool_call("parent", "delegate")
    assert call.result.error.code == "SUBAGENT_CONFIG_ERROR"
    assert call.claim_activation_id is None
    assert runner.store.load_subagent_link("parent", "delegate") is None


@pytest.mark.asyncio
async def test_subagent_repeated_waiting_rebinds_proxy_without_advancing_parent_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_provider = StaticProvider(
        tool_response(ToolUseBlock(id="ask-1", name="ask_question", input={"question": "First?"})),
        tool_response(ToolUseBlock(id="ask-2", name="ask_question", input={"question": "Second?"})),
        text_response("Child done"),
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(child_provider),
    )
    first = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    resume_waiting = runner.store.resume_waiting_run

    def child_only_resume(command: object) -> object:
        assert command.run_id != "parent"
        return resume_waiting(command)

    monkeypatch.setattr(runner.store, "resume_waiting_run", child_only_resume)
    assert first.run.phase == RunPhase.WAITING
    first_proxy = first.pending_interaction
    assert first_proxy.run_id == "parent" and first_proxy.tool_call_id == "delegate"
    child_id = first_proxy.request.subagent_origin.child_run_id
    checkpoint = runner.store.load_checkpoint("parent")
    assert runner.store.load_tool_call("parent", "delegate").phase.value == "prepared"
    second = await runner.resume(
        "parent",
        interaction_id=first_proxy.interaction_id,
        response=QuestionInteractionResponse(answer="One"),
    )
    assert second.run.phase == RunPhase.WAITING
    assert second.pending_interaction.interaction_id != first_proxy.interaction_id
    assert second.pending_interaction.request.subagent_origin.child_run_id == child_id
    assert runner.store.load_checkpoint("parent").engine_cursor == checkpoint.engine_cursor
    assert second.run.usage == first.run.usage
    assert (
        runner.store.load_interaction(first_proxy.interaction_id).status == InteractionStatus.CLOSED
    )
    completed = await runner.resume(
        "parent",
        interaction_id=second.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="Two"),
    )
    assert completed.run.stop_reason == RunStopReason.COMPLETED
    assert completed.assistant_message.text == "Parent complete"
    assert completed.run.usage.tool_calls_committed == 1
    assert runner.store.load_tool_call("parent", "delegate").result.model_content == "Child done"
    assert runner.store.load_run(child_id).usage.tool_calls_committed == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("nondefault,drift", [(False, False), (True, False), (True, True)])
async def test_recover_resolved_proxy_uses_stored_response_and_same_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nondefault: bool,
    drift: bool,
) -> None:
    from iris.harness._subagent import HarnessSubagentController

    path = _write_configs(tmp_path)
    if nondefault:
        (tmp_path / "broken.yaml").write_text(
            (tmp_path / "child.yaml").read_text(encoding="utf-8"), encoding="utf-8"
        )
    parent_provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="delegate",
                name="subagent",
                input={"prompt": "Child task", **({"agent": "broken"} if nondefault else {})},
            )
        )
    )
    db = tmp_path / "state.db"
    runner = AgentRunner.from_config_path(
        path,
        provider=parent_provider,
        store=SQLiteStore(db),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(
                        id="ask",
                        name="ask_question",
                        input={"question": "Which?"},
                    )
                )
            )
        ),
    )
    first = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    proxy = first.pending_interaction
    child_id = proxy.request.subagent_origin.child_run_id

    async def stop_before_child(
        self: HarnessSubagentController, **kwargs: object
    ) -> SubagentExecutionOutcome:
        raise IrisRunPersistenceError("process stopped after response")

    with monkeypatch.context() as patch:
        patch.setattr(HarnessSubagentController, "resume_proxy", stop_before_child)
        with pytest.raises(IrisRunPersistenceError):
            await runner.resume(
                "parent",
                interaction_id=proxy.interaction_id,
                response=QuestionInteractionResponse(answer="Stored answer"),
            )
    saved = runner.store.load_interaction(proxy.interaction_id)
    assert saved.status == InteractionStatus.RESOLVED
    assert saved.response.answer == "Stored answer"
    child_interaction = runner.store.load_interaction(
        proxy.request.subagent_origin.child_interaction_id
    )
    assert child_interaction.status == InteractionStatus.PENDING
    if drift:
        catalog = tmp_path / "subagents.yaml"
        catalog.write_text(
            catalog.read_text(encoding="utf-8").replace("default: researcher", "default: broken"),
            encoding="utf-8",
        )
    factory = ChildProviders(StaticProvider(text_response("Child resumed")))
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(text_response("Parent recovered")),
        store=SQLiteStore(db),
        child_provider_factory=factory,
    )
    completed = await restarted.recover("parent")
    assert factory.configs[0][1] == tmp_path / ("broken.yaml" if nondefault else "child.yaml")
    assert completed.assistant_message.text == "Parent recovered"
    assert restarted.store.load_subagent_link("parent", "delegate").child_run_id == child_id
    assert (
        restarted.store.load_tool_call("parent", "delegate").result.model_content == "Child resumed"
    )
    assert (
        restarted.store.load_interaction(child_interaction.interaction_id).response.answer
        == "Stored answer"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("permission", [False, True])
async def test_recover_proxy_after_child_closes_saved_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, permission: bool
) -> None:
    path = _write_configs(tmp_path)
    db = tmp_path / "state.db"
    (tmp_path / "notes.txt").write_text("Saved notes", encoding="utf-8")
    policy = RecordingPermissionPolicy("read_file")
    factory = ChildProviders(
        StaticProvider(
            tool_response(
                ToolUseBlock(id="inner", name="read_file", input={"file_path": "notes.txt"})
                if permission
                else ToolUseBlock(id="inner", name="ask_question", input={"question": "Continue?"})
            )
        )
    )
    runner = AgentRunner.from_config_path(
        path,
        provider=_parent_provider(),
        store=SQLiteStore(db),
        permission_policy=policy,
        child_provider_factory=factory,
    )
    waiting = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    proxy = waiting.pending_interaction
    child_id = proxy.request.subagent_origin.child_run_id
    response = (
        PermissionInteractionResponse(decision="approve")
        if permission
        else QuestionInteractionResponse(answer="Saved answer")
    )
    original = AgentRunner._run_activation

    async def stop_child_resume(self: AgentRunner, active: object, **kwargs: object) -> RunResult:
        if active.run_id == child_id and kwargs["activation"].kind == "resume":
            raise IrisRunPersistenceError("Stopped after child ResumeWaitingRun")
        return await original(self, active, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(AgentRunner, "_run_activation", stop_child_resume)
        with pytest.raises(IrisRunPersistenceError):
            await runner.resume("parent", interaction_id=proxy.interaction_id, response=response)
    child_interaction = runner.store.load_interaction(
        proxy.request.subagent_origin.child_interaction_id
    )
    assert child_interaction.status == InteractionStatus.CLOSED
    assert child_interaction.response == response
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(text_response("Parent recovered")),
        store=SQLiteStore(db),
        permission_policy=policy,
        child_provider_factory=ChildProviders(StaticProvider(text_response("Child recovered"))),
    )
    result = await restarted.recover("parent")
    assert result.run.stop_reason == RunStopReason.COMPLETED
    assert restarted.store.load_subagent_link("parent", "delegate").child_run_id == child_id
    inner = restarted.store.load_tool_call(child_id, "inner")
    assert not inner.result.is_error
    assert inner.interaction_id == child_interaction.interaction_id
    if not permission:
        assert inner.result.data["answer"] == "Saved answer"


@pytest.mark.asyncio
async def test_parent_recovery_uses_current_child_system_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    db = tmp_path / "state.db"
    child_provider = StaticProvider(text_response("Child recovered"))
    factory = ChildProviders(child_provider)
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), store=SQLiteStore(db), child_provider_factory=factory
    )
    prepared = _prepare_parent(runner)

    async def stop_admitted(*args: object, **kwargs: object) -> RunResult:
        raise IrisRunPersistenceError("Stopped after admission")

    with monkeypatch.context() as patch:
        patch.setattr(AgentRunner, "_run_admitted_start", stop_admitted)
        with pytest.raises(IrisRunPersistenceError):
            await _dispatch(runner, prepared)
    child_id = runner.store.load_subagent_link("parent", "delegate").child_run_id
    child_path = tmp_path / "child.yaml"
    original_yaml = child_path.read_text(encoding="utf-8")
    child_path.write_text(
        original_yaml.replace("Child instructions", "Changed instructions"), encoding="utf-8"
    )
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(text_response("Parent recovered")),
        store=SQLiteStore(db),
        child_provider_factory=factory,
    )
    result = await restarted.recover(
        "parent", expected_activation_id=restarted.store.load_run("parent").current_activation_id
    )
    assert result.run.stop_reason == RunStopReason.COMPLETED
    assert restarted.store.load_run(child_id).phase == RunPhase.TERMINAL
    assert "Changed instructions" in child_provider.requests[0].messages[0].text
    assert "Child instructions" not in child_provider.requests[0].messages[0].text


class RecordingPermissionPolicy(DefaultPermissionPolicy):
    """记录 outer/actual tool，并支持在已有批准后改变执行裁决。"""

    def __init__(self, gated_tool: str) -> None:
        self.gated_tool = gated_tool
        self.effect = PermissionEffect.REQUIRE_HUMAN
        self.calls: list[str] = []

    def check(self, tool: object, params: object, context: object) -> PermissionDecision:
        self.calls.append(tool.name)
        return PermissionDecision(
            effect=self.effect if tool.name == self.gated_tool else PermissionEffect.ALLOW,
            reason="Confirm selected tool",
        )


@pytest.mark.asyncio
async def test_permission_proxy_keeps_actual_refresh_without_repeating_outer_gate(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("Selected notes", encoding="utf-8")
    policy = RecordingPermissionPolicy("read_file")
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        permission_policy=policy,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="read", name="read_file", input={"file_path": "notes.txt"})
                ),
                text_response("Child read notes"),
            )
        ),
    )
    waiting = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    proxy = waiting.pending_interaction
    assert proxy.request.prompt.kind.value == "permission"
    assert policy.calls == ["subagent", "subagent", "read_file"]
    completed = await runner.resume(
        "parent",
        interaction_id=proxy.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    assert completed.run.stop_reason == RunStopReason.COMPLETED
    assert policy.calls == ["subagent", "subagent", "read_file", "read_file", "read_file"]
    child_id = proxy.request.subagent_origin.child_run_id
    assert not runner.store.load_tool_call(child_id, "read").result.is_error


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize(
    "decision,deny", [("approve", False), ("approve", True), ("reject", False)]
)
async def test_recover_stored_outer_response_preserves_fresh_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, closed: bool, decision: str, deny: bool
) -> None:
    path = _write_configs(tmp_path)
    policy = RecordingPermissionPolicy("subagent")
    runner = AgentRunner.from_config_path(
        path, provider=_parent_provider(), permission_policy=policy
    )
    waiting = await runner.start(AgentRunRequest(input="Start", run_id="parent"))
    assert runner.store.load_subagent_link("parent", "delegate") is None

    async def stop_activation(*args: object, **kwargs: object) -> RunResult:
        raise IrisRunPersistenceError("process stopped after closing outer gate")

    def stop_begin(*args: object, **kwargs: object) -> object:
        raise IrisRunPersistenceError("process stopped after resolving outer gate")

    with monkeypatch.context() as patch:
        if closed:
            patch.setattr(runner, "_run_activation", stop_activation)
        else:
            patch.setattr(runner.store, "resume_waiting_run", stop_begin)
        with pytest.raises(IrisRunPersistenceError):
            await runner.resume(
                "parent",
                interaction_id=waiting.pending_interaction.interaction_id,
                response=PermissionInteractionResponse(decision=decision),
            )
    policy.calls.clear()
    if deny:
        policy.effect = PermissionEffect.DENY
    factory = ChildProviders(StaticProvider(text_response("Admitted child")))
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(text_response("Parent recovered")),
        store=runner.store,
        permission_policy=policy,
        child_provider_factory=factory,
    )
    run = restarted.store.load_run("parent")
    completed = await restarted.recover(
        "parent", expected_activation_id=run.current_activation_id if closed else None
    )
    assert completed.run.stop_reason == RunStopReason.COMPLETED
    admitted = decision == "approve" and not deny
    assert (restarted.store.load_subagent_link("parent", "delegate") is not None) == admitted
    assert policy.calls == ([] if decision == "reject" else ["subagent"])
    assert len(factory.configs) == int(admitted)
    if not admitted:
        assert restarted.store.load_tool_call("parent", "delegate").result.error.code == (
            "USER_REJECTED" if decision == "reject" else "PERMISSION_ERROR"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("child_phase", ["active", "waiting", "terminal"])
async def test_parent_active_recovery_reuses_linked_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_phase: str
) -> None:
    path = _write_configs(tmp_path)
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"}))
        if child_phase == "waiting"
        else text_response("Original child")
    )
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), child_provider_factory=ChildProviders(provider)
    )
    prepared = _prepare_parent(runner)

    async def stop_after_admission(*args: object, **kwargs: object) -> RunResult:
        raise IrisRunPersistenceError("process stopped after admission")

    if child_phase == "active":
        with monkeypatch.context() as patch:
            patch.setattr(AgentRunner, "_run_admitted_start", stop_after_admission)
            with pytest.raises(IrisRunPersistenceError):
                await _dispatch(runner, prepared)
    else:
        await _dispatch(runner, prepared)
    link = runner.store.load_subagent_link("parent", "delegate")
    assert runner.store.load_run(link.child_run_id).phase.value == child_phase
    policy = RecordingPermissionPolicy("subagent")
    policy.effect = PermissionEffect.DENY
    # 运行时裁决可收紧，但 linked call 不再次询问 outer。
    restarted = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(text_response("Parent recovered")),
        store=runner.store,
        child_provider_factory=ChildProviders(StaticProvider(text_response("Recovered child"))),
    )
    monkeypatch.setattr(
        restarted.runtime.environment.tool_bridge.tool_executor.permission_policy,
        "check",
        policy.check,
    )
    result = await restarted.recover(
        "parent", expected_activation_id=runner.store.load_run("parent").current_activation_id
    )
    assert restarted.store.load_subagent_link("parent", "delegate") == link
    assert policy.calls == []
    if child_phase == "waiting":
        assert result.run.phase == RunPhase.WAITING
        assert result.pending_interaction.request.subagent_origin.child_run_id == link.child_run_id
        with pytest.raises(IrisRunStateError, match="resume"):
            await restarted.recover("parent")
    else:
        assert result.run.stop_reason == RunStopReason.COMPLETED
        assert result.run.usage.tool_calls_committed == 1
        assert await restarted.recover("parent") == result


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "child", "artifact"])
async def test_waiting_final_uses_parent_artifact_and_publishes_before_continuing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    provider = _parent_provider()
    child_text = "Long child answer " * 3000
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=provider,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                ),
                text_response(child_text),
            )
        ),
    )
    waiting = await runner.start(
        AgentRunRequest(input="Start", run_id="parent", session_id="parent-session"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
        ),
    )
    old_activation = runner.store.load_checkpoint("parent").activation_id
    publisher = RecordingPublisher()
    # 仅接收 continuation 发布；无需让无 streaming 的测试 provider 改走 streaming。
    monkeypatch.setattr(runner, "_publish_live_fact", publisher.publish)
    original = provider.complete

    async def assert_final_before_model(request: object) -> object:
        assert any(
            isinstance(fact, RuntimeStreamEvent) and fact.kind == "tool.completed"
            for fact in publisher.facts
        )
        return await original(request)

    monkeypatch.setattr(provider, "complete", assert_final_before_model)
    if failure == "child":
        child_id = waiting.pending_interaction.request.subagent_origin.child_run_id
        child = runner._subagent_controller._assemble_child(
            runner._subagent_controller.routes.routes["researcher"]
        )
        child.request_cancel(child_id)
    elif failure == "artifact":
        (tmp_path / ".iris" / "tool-results").parent.mkdir(exist_ok=True)
        (tmp_path / ".iris" / "tool-results").write_text(
            "A file occupies the artifact directory", encoding="utf-8"
        )
    completed = await runner.resume(
        "parent",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="Continue"),
    )
    final = [
        fact
        for fact in publisher.facts
        if isinstance(fact, RuntimeStreamEvent) and fact.kind == "tool.completed"
    ]
    assert len(final) == 1 and final[0].activation_id == old_activation
    assert final[0].run_id == "parent" and final[0].tool_call_id == "delegate"
    call = runner.store.load_tool_call("parent", "delegate")
    if failure is not None:
        assert completed.run.stop_reason == RunStopReason.FAILED
        assert len(provider.requests) == 1
        assert call.result.error.code == (
            "SUBAGENT_CANCELLED" if failure == "child" else "ARTIFACT_ERROR"
        )
    else:
        assert completed.run.stop_reason == RunStopReason.COMPLETED
        assert call.result.artifact.path.read_text(encoding="utf-8") == child_text
        assert call.result.artifact.path.is_relative_to(
            tmp_path / ".iris" / "tool-results" / safe_path_segment("parent-session")
        )
    assert call.result.tool_use_id == "delegate" and call.claim_activation_id is None


@pytest.mark.asyncio
async def test_proxy_managed_admission_releases_lock_and_rejects_second_response(
    tmp_path: Path,
) -> None:
    child_factory = ChildProviders(
        StaticProvider(
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
            )
        )
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path), provider=_parent_provider(), child_provider_factory=child_factory
    )
    manager = SessionManager(runner, "managed-session")
    receipt = await manager.submit("Start")
    await asyncio.wait_for(asyncio.shield(manager._current_task), timeout=2)
    proxy = runner.store.load_result(receipt.run_id).pending_interaction
    blocker = BlockingProvider(text_response("Child complete"))
    child_factory.provider = blocker
    try:
        response = QuestionInteractionResponse(answer="Continue")
        admitted = await asyncio.wait_for(
            manager.admit_resume(interaction_id=proxy.interaction_id, response=response), timeout=1
        )
        assert admitted.run_id == receipt.run_id
        await asyncio.wait_for(blocker.started.wait(), timeout=1)
        assert not manager._lock.locked()
        with pytest.raises(IrisRunStateError, match="managed continuation"):
            await manager.admit_resume(interaction_id=proxy.interaction_id, response=response)
    finally:
        blocker.release.set()
        await manager.close()


class BlockingChildTool:
    """不读取 Iris 信号，以独立事件控制业务等待和取消后的清理。"""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.body_release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.finished = asyncio.Event()

    async def run(self) -> str:
        """在 Python cancellation 到达后继续等待测试放行 finally。"""
        self.entered.set()
        try:
            await self.body_release.wait()
            return "Child body done"
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            await self.cleanup_release.wait()
            self.finished.set()


@pytest.fixture
def blocking_child_tool(monkeypatch: pytest.MonkeyPatch) -> BlockingChildTool:
    """保留真实 runtime 装配，仅为 child registry 增加受控普通 callable。"""
    tool = BlockingChildTool()

    def with_child_tool(
        config: ToolsConfig,
        *,
        memory_service: MemoryService | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> ToolRegistry:
        registry = build_tool_registry(
            config, memory_service=memory_service, memory_config=memory_config
        )
        if "human.ask" in config.builtin:
            registry.register_function(tool.run, name="blocking_child")
        return registry

    monkeypatch.setattr("iris.runtime._assembly.build_tool_registry", with_child_tool)
    return tool


def _write_configs(tmp_path: Path) -> Path:
    (tmp_path / "agent.yaml").write_text(
        "name: parent\nmodel: openai/parent-model\nsystem: Parent instructions\n"
        "tools:\n  subagent: subagents.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "subagents.yaml").write_text(
        "default: researcher\nagents:\n  researcher:\n    path: child.yaml\n"
        "    description: Research selected task\n  broken:\n    path: broken.yaml\n"
        "    description: Invalid unselected child\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "name: researcher\nmodel: openai/child-model\nsystem: Child instructions\n"
        "tools:\n  builtin: [file.read, human.ask]\n  subagent: missing-catalog.yaml\n"
        "session:\n  backend: sqlite\n  path: never-created.db\n",
        encoding="utf-8",
    )
    (tmp_path / "broken.yaml").write_text("[broken", encoding="utf-8")
    return tmp_path / "agent.yaml"


class ChildProviders(ChildProviderFactory):
    """记录按 selected child config 构造 provider 的实际输入。"""

    def __init__(self, provider: StaticProvider) -> None:
        self.provider = provider
        self.configs: list[tuple[AgentConfig, Path]] = []

    def __call__(self, config: AgentConfig, *, config_path: Path) -> CompletionProvider:
        self.configs.append((config, config_path))
        return self.provider


def _prepare_parent(
    runner: AgentRunner,
    *,
    selector: str | None = None,
    options: AgentRunOptions | None = None,
) -> PreparedToolCall:
    """只提交真实 parent model/tool facts，phase04 不运行 parent tool loop。"""
    command, cursor = runner._build_start_facts(
        AgentRunRequest(
            input="Parent private request",
            session_id="parent-session",
            run_id="parent",
            metadata={"parent-only": True},
        ),
        options=options,
    )
    created = runner.store.create_run(command)
    input_committed = runner.store.commit_run_input(
        CommitRunInput(
            run_id="parent",
            expected_run_revision=created.run.revision,
            activation_id=command.start_activation_id,
            expected_session_revision=0,
            message_delta=[Msg.user("Parent private history")],
            checkpoint=command.initial_checkpoint.model_copy(
                update={
                    "sequence": 2,
                    "session_revision": 1,
                    "engine_cursor": cursor.model_copy(
                        update={"position": "before_model"}
                    ).model_dump(mode="json"),
                }
            ),
            now=runner.clock.now(),
        )
    )
    reserved = runner.store.reserve_model_step(
        ReserveModelStep(
            run_id="parent",
            expected_run_revision=input_committed.run.revision,
            activation_id=command.start_activation_id,
            now=runner.clock.now(),
        )
    )
    tool_use = ToolUseBlock(id="delegate", name="subagent", input={"prompt": "Child task"})
    if selector is not None:
        tool_use.input["agent"] = selector
    assistant = Msg.assistant([tool_use])
    prepared = runner.runtime.environment.tool_bridge.preflight_once(
        assistant_message=assistant,
        session_id="parent-session",
        run_id="parent",
        agent_id="parent",
        workspace_root=runner.runtime.environment.workspace_root,
        permission_mode="default",
        metadata=None,
    ).calls[0]
    after = RuntimeCursor(
        position="tool_batch", step_index=0, tool_calls=(tool_use,), assistant_message=assistant
    )
    fingerprint = make_call_fingerprint(
        session_id="parent-session",
        run_id="parent",
        tool_call_id="delegate",
        tool_name="subagent",
        arguments=prepared.arguments,
        workspace_root=str(runner.runtime.environment.workspace_root),
    )
    runner.store.commit_model_step(
        CommitModelStep(
            run_id="parent",
            expected_run_revision=reserved.run.revision,
            activation_id=command.start_activation_id,
            expected_session_revision=1,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1, total_tokens=11),
            prepared_tool_calls=[
                RunToolCallRecord(
                    run_id="parent",
                    step_index=0,
                    ordinal=1,
                    tool_call_id="delegate",
                    tool_name="subagent",
                    arguments=prepared.arguments,
                    fingerprint=fingerprint,
                    phase="prepared",
                    version=1,
                    created_at=runner.clock.now(),
                    updated_at=runner.clock.now(),
                )
            ],
            checkpoint=command.initial_checkpoint.model_copy(
                update={
                    "sequence": 3,
                    "session_revision": 2,
                    "model_steps_reserved": 1,
                    "model_steps_committed": 1,
                    "engine_cursor": after.model_dump(mode="json"),
                }
            ),
            assistant_message=assistant,
            now=runner.clock.now(),
        )
    )
    return prepared


async def _dispatch(
    runner: AgentRunner, prepared: PreparedToolCall, *, linked: bool = False
) -> SubagentExecutionOutcome:
    return await runner.runtime.environment.tool_bridge.execute_subagent_prepared(
        prepared,
        session_id="parent-session",
        run_id="parent",
        agent_id="parent",
        workspace_root=runner.runtime.environment.workspace_root,
        permission_mode="default",
        metadata=None,
        linked_continuation=linked,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", [None, "researcher"])
async def test_child_uses_fresh_context_options_and_shared_store(
    tmp_path: Path, selector: str | None
) -> None:
    path = _write_configs(tmp_path)
    (tmp_path / "notes.txt").write_text("local notes", encoding="utf-8")
    child_provider = StaticProvider(
        tool_response(
            ToolUseBlock(id="child-read", name="read_file", input={"file_path": "notes.txt"})
        ),
        text_response("Child answer"),
    )
    factory = ChildProviders(child_provider)
    parent_provider = StaticProvider()
    runner = AgentRunner.from_config_path(
        path, provider=parent_provider, child_provider_factory=factory
    )
    # 装配后的 catalog 不再读取；未选 child YAML 从未需要有效。
    (tmp_path / "subagents.yaml").write_text("[invalid now", encoding="utf-8")
    assert factory.configs == []
    options = AgentRunOptions(
        limits=RunLimits(max_model_steps=1),
        runtime=RuntimeExecutionOptions(
            request_options={"temperature": 0.1},
            memory_results=[{"parent-only": True}],
        ),
    )
    prepared = _prepare_parent(runner, selector=selector, options=options)
    result = await _dispatch(runner, prepared)
    assert isinstance(result, ToolResult)
    assert result.model_content == "Child answer"
    child_id = result.metadata["child_run_id"]
    child = runner.store.load_run(child_id)
    assert child.request.input == "Child task"
    assert child.request.metadata == {}
    assert child.options == AgentRunOptions()
    assert child.session_id != "parent-session"
    assert child.run_id != "parent"
    assert child.phase == RunPhase.TERMINAL
    assert child.usage.tool_calls_committed == 1
    assert result.metadata == {"agent_selector": "researcher", "child_run_id": child_id}
    assert runner.store.load_run("parent").usage.total_tokens == 11
    assert runner.store.load_run("parent").usage.tool_calls_committed == 0
    assert runner.store.load_tool_call("parent", "delegate").phase.value == "prepared"
    assert parent_provider.requests == []
    assert len(factory.configs) == 1
    assert factory.configs[0][0].model.name == "child-model"
    assert factory.configs[0][1] == tmp_path / "child.yaml"
    request = child_provider.requests[0]
    texts = "\n".join(message.text for message in request.messages)
    assert "Child instructions" in texts and "Child task" in texts
    assert "Parent private" not in texts and "Parent instructions" not in texts
    assert not (tmp_path / "never-created.db").exists()
    assert await _dispatch(runner, prepared, linked=True) == result
    assert len(factory.configs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,code", [("broken", "SUBAGENT_CONFIG_ERROR"), ("disjoint", "SUBAGENT_WORKSPACE_DISJOINT")]
)
async def test_fresh_selected_config_failures_do_not_admit_child(
    tmp_path: Path, kind: str, code: str
) -> None:
    path = _write_configs(tmp_path)
    if kind == "disjoint":
        with (tmp_path / "child.yaml").open("a", encoding="utf-8") as config:
            config.write("permissions:\n  workspace: ../outside\n")
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), child_provider_factory=ChildProviders(StaticProvider())
    )
    prepared = _prepare_parent(runner, selector="broken" if kind == "broken" else None)
    result = await _dispatch(runner, prepared)
    assert result.error.code == code
    assert not result.error.retryable
    assert result.metadata == {"agent_selector": "broken" if kind == "broken" else "researcher"}
    assert runner.store.load_subagent_link("parent", "delegate") is None


@pytest.mark.asyncio
async def test_waiting_child_reentry_reads_original_result_without_yaml_or_provider(
    tmp_path: Path,
) -> None:
    path = _write_configs(tmp_path)
    factory = ChildProviders(
        StaticProvider(
            tool_response(
                ToolUseBlock(
                    id="ask",
                    name="ask_question",
                    input={"question": "Which notes?"},
                )
            )
        )
    )
    clock = FrozenClock()
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), child_provider_factory=factory, clock=clock
    )
    prepared = _prepare_parent(
        runner,
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(tool_timeout_seconds=10),
            limits=RunLimits(interaction_timeout_seconds=30),
        ),
    )
    waiting = await _dispatch(runner, prepared)
    assert isinstance(waiting, ChildWaiting)
    assert waiting.expiry_owner.value == "outer_tool_timeout"
    assert waiting.proxy_expires_at == clock.now() + timedelta(seconds=10)
    (tmp_path / "child.yaml").write_text("[broken after admission", encoding="utf-8")
    clock.advance(seconds=2)
    again = await _dispatch(runner, prepared, linked=True)
    assert again == waiting
    assert len(factory.configs) == 1
    assert runner.store.load_run("parent").phase == RunPhase.ACTIVE


def _admit_durable_child(runner: AgentRunner) -> str:
    parent = runner.store.load_run("parent")
    child_create = CreateRun(
        request=AgentRunRequest(input="Child task", session_id="child-session", run_id="child"),
        options=AgentRunOptions(),
        agent_id="researcher",
        start_activation_id="child-a",
        initial_checkpoint=RunCheckpoint(
            run_id="child",
            sequence=1,
            activation_id="child-a",
            engine_cursor={},
            session_revision=0,
            model_steps_reserved=0,
            model_steps_committed=0,
        ),
        now=runner.clock.now(),
    )
    return runner.store.admit_child_run(
        AdmitChildRun(
            parent_run_id="parent",
            expected_parent_run_revision=parent.revision,
            parent_activation_id=parent.current_activation_id,
            parent_tool_call_id="delegate",
            expected_parent_tool_version=1,
            child_create=child_create,
        )
    ).child_run_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,code",
    [
        (RunStopReason.FAILED, "SUBAGENT_FAILED"),
        (RunStopReason.BUDGET_EXHAUSTED, "SUBAGENT_BUDGET_EXHAUSTED"),
        (RunStopReason.DEADLINE_EXCEEDED, "SUBAGENT_TIMEOUT"),
        (RunStopReason.INTERACTION_EXPIRED, "SUBAGENT_TIMEOUT"),
        (RunStopReason.CANCELLED, "SUBAGENT_CANCELLED"),
        (RunStopReason.OUTCOME_UNKNOWN, "SUBAGENT_OUTCOME_UNKNOWN"),
        (RunStopReason.COMPLETED, "SUBAGENT_FAILED"),
    ],
)
async def test_linked_terminal_projection_is_nonretryable_and_run_local(
    tmp_path: Path,
    reason: RunStopReason,
    code: str,
) -> None:
    runner = AgentRunner.from_config_path(_write_configs(tmp_path), provider=StaticProvider())
    prepared = _prepare_parent(runner)
    child_id = _admit_durable_child(runner)
    runner.store.finish_run(
        FinishRun(
            run_id=child_id,
            expected_run_revision=1,
            activation_id="child-a",
            stop_reason=reason,
            error=RunErrorInfo(code="CHILD_FAILURE", message="details", source="runtime")
            if reason in {RunStopReason.FAILED, RunStopReason.OUTCOME_UNKNOWN}
            else None,
            now=runner.clock.now(),
        )
    )
    result = await _dispatch(runner, prepared, linked=True)
    assert result.error.code == code
    assert not result.error.retryable
    assert result.content == [] and result.data == {} and result.stats == {}
    assert result.metadata == {"agent_selector": "researcher", "child_run_id": child_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", [False, True])
async def test_restart_recovers_original_active_child_with_current_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: bool,
) -> None:
    path = _write_configs(tmp_path)
    db = tmp_path / "state.db"
    child_provider = StaticProvider(text_response("Recovered child"))
    factory = ChildProviders(child_provider)
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), store=SQLiteStore(db), child_provider_factory=factory
    )
    prepared = _prepare_parent(runner)

    async def stop_after_admission(
        self: AgentRunner, *, run_id: str, activation_id: str
    ) -> RunResult:
        raise IrisRunPersistenceError("process stopped after admission")

    with monkeypatch.context() as patch:
        patch.setattr(AgentRunner, "_run_admitted_start", stop_after_admission)
        with pytest.raises(IrisRunPersistenceError):
            await _dispatch(runner, prepared)
    child_id = runner.store.load_subagent_link("parent", "delegate").child_run_id
    if drift:
        child_path = tmp_path / "child.yaml"
        child_path.write_text(
            child_path.read_text(encoding="utf-8").replace(
                "Child instructions", "Changed instructions"
            ),
            encoding="utf-8",
        )
    restarted = AgentRunner.from_config_path(
        path, provider=StaticProvider(), store=SQLiteStore(db), child_provider_factory=factory
    )
    durable = restarted.store.load_tool_call("parent", "delegate")
    rebuilt = restarted.runtime.environment.tool_bridge.prepare_subagent_continuation(
        ToolUseBlock(id=durable.tool_call_id, name=durable.tool_name, input=durable.arguments),
        session_id="parent-session",
        run_id="parent",
        agent_id="parent",
        workspace_root=tmp_path,
        permission_mode="default",
        metadata=None,
    )
    result = await _dispatch(restarted, rebuilt, linked=True)
    assert result.model_content == "Recovered child"
    assert result.metadata["child_run_id"] == child_id
    assert restarted.store.load_run(child_id).phase == RunPhase.TERMINAL
    expected_system = "Changed instructions" if drift else "Child instructions"
    assert expected_system in child_provider.requests[0].messages[0].text
    assert restarted.store.load_subagent_link("parent", "delegate").child_run_id == child_id
    assert (
        len(
            [
                event
                for event in restarted.store.list_events(child_id)
                if event.kind.value == "run.started"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_success_keeps_empty_assistant_text(tmp_path: Path) -> None:
    factory = ChildProviders(StaticProvider(text_response("")))
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path), provider=StaticProvider(), child_provider_factory=factory
    )
    result = await _dispatch(runner, _prepare_parent(runner))
    assert not result.is_error
    assert result.model_content == ""


@pytest.mark.asyncio
async def test_runner_injected_parent_policy_applies_to_child_only_tool(tmp_path: Path) -> None:
    class ParentPolicy(DefaultPermissionPolicy):
        def check(self, tool: object, params: object, context: object) -> PermissionDecision:
            if tool.name == "read_file":
                return PermissionDecision(
                    effect=PermissionEffect.DENY, reason="parent denies reading"
                )
            return PermissionDecision(effect=PermissionEffect.ALLOW)

    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="read",
                name="read_file",
                input={"file_path": "notes.txt"},
            )
        ),
        text_response("Cannot read"),
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=StaticProvider(),
        permission_policy=ParentPolicy(),
        child_provider_factory=ChildProviders(provider),
    )
    result = await _dispatch(runner, _prepare_parent(runner))
    assert result.model_content == "Cannot read"
    child_result = runner.store.load_tool_call(result.metadata["child_run_id"], "read").result
    assert child_result.error.code == "PERMISSION_ERROR"
    assert child_result.error.message == "parent denies reading"


def test_runner_reads_catalog_once_and_defers_child_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    read_text = Path.read_text
    reads: list[Path] = []

    def record_read(self: Path, *args: object, **kwargs: object) -> str:
        reads.append(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)
    AgentRunner.from_config_path(path, provider=StaticProvider())
    assert reads.count(tmp_path / "subagents.yaml") == 1
    assert tmp_path / "child.yaml" not in reads
    assert tmp_path / "broken.yaml" not in reads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate,owner",
    [
        ("none", None),
        ("parent_deadline", "parent_run_deadline"),
        ("parent_timeout", "parent_interaction_timeout"),
        ("child_interaction", "child_interaction_expiry"),
        ("child_deadline", "child_effective_deadline"),
        ("outer", "outer_tool_timeout"),
        ("tie", "parent_run_deadline"),
    ],
)
async def test_waiting_expiry_projection_uses_earliest_absolute_candidate(
    tmp_path: Path,
    candidate: str,
    owner: str | None,
) -> None:
    clock = FrozenClock()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=StaticProvider(),
        clock=clock,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(
                        id="ask",
                        name="ask_question",
                        input={"question": "Continue?"},
                    )
                )
            )
        ),
    )
    waiting = await _dispatch(runner, _prepare_parent(runner))
    parent = runner.store.load_run("parent")
    child = runner.store.load_run(waiting.child_run_id)
    result = runner.store.load_result(child.run_id)
    soon = clock.now() + timedelta(seconds=5)
    limits = RunLimits(
        deadline_at=soon if candidate in {"parent_deadline", "tie"} else None,
        interaction_timeout_seconds=5 if candidate in {"parent_timeout", "tie"} else None,
    )
    parent = parent.model_copy(
        update={
            "options": AgentRunOptions(
                limits=limits,
                runtime=RuntimeExecutionOptions(
                    tool_timeout_seconds=5 if candidate in {"outer", "tie"} else None
                ),
            )
        }
    )
    if candidate == "child_deadline":
        child = child.model_copy(
            update={"options": AgentRunOptions(limits=RunLimits(deadline_at=soon))}
        )
    if candidate == "child_interaction":
        result = result.model_copy(
            update={
                "pending_interaction": result.pending_interaction.model_copy(
                    update={"expires_at": soon}
                )
            }
        )
    projected = runner._subagent_controller._project_outcome("researcher", parent, child, result)
    assert projected.expiry_owner == owner
    assert projected.proxy_expires_at == (None if owner is None else soon)
