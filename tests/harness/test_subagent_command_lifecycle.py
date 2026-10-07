"""Child 的 live 资源归属、收据交接与父调用因果范围。"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.command import (
    CommandMode,
    CommandOutcome,
    CommandOutputStats,
    CommandRequest,
    CommandScope,
    CommandStatus,
    ShellCommand,
    StopOperation,
)
from iris.exceptions import IrisCommandCleanupError
from iris.harness import AgentRunner, CommandCleanupFailed
from iris.harness.streaming import LineagedLiveFact
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import (
    AdmitChildRun,
    AgentRunOptions,
    AgentRunRequest,
    CreateRun,
    RunLimits,
    RunPhase,
    RunResult,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolErrorPolicy,
)
from iris.message import ToolUseBlock
from iris.runtime import RuntimeCursor

from .fakes import RecordingPublisher, StaticProvider, text_response, tool_response
from .test_command_settlement import ControlledService, FailedProvider, bind_service
from .test_runner_subagent import (
    ChildProviders,
    StreamingStaticProvider,
    _parent_provider,
    _prepare_parent,
    _write_configs,
)


class StreamingFailedProvider(FailedProvider, StreamingStaticProvider):
    """通过 typed stream 复用原始 provider 失败。"""


@pytest.mark.asyncio
async def test_child_command_target_routes_exact_live_run_and_borrows_owner(
    tmp_path: Path,
) -> None:
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(StaticProvider(text_response("child"))),
    )
    controller = runner._subagent_controller
    assert controller is not None
    route = controller.routes.routes["researcher"]
    first = controller._assemble_child(route)
    second = controller._assemble_child(route)

    async def pending() -> RunResult:
        await asyncio.Event().wait()
        raise AssertionError("test task never finishes normally")

    tasks = [asyncio.create_task(pending()), asyncio.create_task(pending())]
    controller._live_children["first"] = (first, tasks[0])
    controller._live_children["second"] = (second, tasks[1])
    try:
        async with controller.open_command_target(route, "second") as borrowed:
            assert borrowed is second
            assert borrowed._command_lifecycle is runner._command_lifecycle
        assert not second._resources_closed
        async with controller.open_command_target(route, "not-live") as reopened:
            assert reopened not in (first, second)
            assert reopened._command_lifecycle is runner._command_lifecycle
        assert reopened._resources_closed
        assert not runner._resources_closed
    finally:
        controller._live_children.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await first.aclose()
        await second.aclose()
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", ["active", "proxy", "continue"])
async def test_child_stop_receipt_only_applies_to_current_parent_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flow: str
) -> None:
    child_provider = StaticProvider(
        *(
            [
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                )
            ]
            if flow == "proxy"
            else []
        )
    )
    # 没有后续响应会产生真实 provider failure；continue 模式检查父调用已接受后不能复用旧收据。
    parent_provider = _parent_provider()
    if flow == "continue":
        parent_provider.responses.pop()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=parent_provider,
        child_provider_factory=ChildProviders(child_provider),
    )
    service = runner.runtime.environment.command_binding.service
    original = service.stop
    stopped: list[CommandScope] = []

    def stop(scope: CommandScope) -> StopOperation:
        stopped.append(scope)
        return original(scope)

    monkeypatch.setattr(service, "stop", stop)
    try:
        result = await runner.start(
            AgentRunRequest(input="parent", run_id="parent", session_id="parent-session"),
            options=AgentRunOptions(
                runtime=RuntimeExecutionOptions(
                    tool_error_policy=ToolErrorPolicy.RETURN_TO_MODEL
                    if flow == "continue"
                    else ToolErrorPolicy.STOP
                )
            ),
        )
        if flow == "proxy":
            assert result.run.phase is RunPhase.WAITING
            result = await runner.resume(
                "parent",
                interaction_id=result.pending_interaction.interaction_id,
                response=QuestionInteractionResponse(answer="continue"),
            )
        assert result.run.stop_reason is RunStopReason.FAILED
        assert len(stopped) == (2 if flow == "continue" else 1)
        assert stopped[0].run_id != "parent"
        assert not runner._command_lifecycle.settled_receipts
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_child_waiting_deadline_reopens_runner_after_temporary_close(tmp_path: Path) -> None:
    """child 的绝对期限属于 root，即使临时 runner 已关闭也无需新输入。"""
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
    service = ControlledService()
    service.release()
    bind_service(runner, service)
    controller = runner._subagent_controller
    assert controller is not None
    controller.parent_boundary = replace(
        controller.parent_boundary, command_binding=runner.runtime.environment.command_binding
    )
    route = controller.routes.routes["researcher"]
    child = controller._assemble_child(route)
    try:
        waiting = await child.start(
            AgentRunRequest(input="child", run_id="child", session_id="child-session"),
            options=AgentRunOptions(
                limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(seconds=0.15))
            ),
        )
        assert waiting.run.phase is RunPhase.WAITING
        timer = runner._command_lifecycle.deadlines["child"].task
        await child.aclose()
        assert child._resources_closed
        assert not timer.done()
        await asyncio.wait_for(asyncio.shield(timer), 2)
        assert runner.store.load_run("child").stop_reason is RunStopReason.DEADLINE_EXCEEDED
        assert [scope.run_id for scope in service.calls] == ["child"]
        assert "child" not in runner._command_lifecycle.deadlines
    finally:
        await child.aclose()
        await runner.aclose()


@pytest.mark.asyncio
async def test_child_pending_survives_close_and_retry_preserves_original_failure(
    tmp_path: Path,
) -> None:
    """关闭临时 child 不丢 pending；重建后的 cancel 只补清理并保留 provider 原因。"""
    publisher = RecordingPublisher()
    provider = StreamingFailedProvider()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(provider),
        live_publisher=publisher,
    )
    service = ControlledService()
    service.fail = True
    bind_service(runner, service)
    controller = runner._subagent_controller
    assert controller is not None
    controller.parent_boundary = replace(
        controller.parent_boundary, command_binding=runner.runtime.environment.command_binding
    )
    route = controller.routes.routes["researcher"]
    child = controller._assemble_child(route)
    _prepare_parent(runner)
    parent = runner.store.load_run("parent")
    child_create, _ = child._build_start_facts(
        AgentRunRequest(input="child", run_id="child", session_id="child-session"), options=None
    )
    runner.store.admit_child_run(
        AdmitChildRun(
            parent_run_id="parent",
            expected_parent_run_revision=parent.revision,
            parent_activation_id=parent.current_activation_id,
            parent_tool_call_id="delegate",
            expected_parent_tool_version=1,
            child_create=child_create,
            agent_selector=route.selector,
        )
    )
    controller._bind_child_live(child, "child")
    try:
        with pytest.raises(IrisCommandCleanupError):
            await child._run_admitted_start(
                run_id="child", activation_id=child_create.start_activation_id
            )
        await child.aclose()
        assert child._resources_closed
        assert "child" in runner._command_lifecycle.pending
        assert runner.store.load_run("child").phase is RunPhase.ACTIVE
        assert (
            len(
                [
                    fact
                    for fact in publisher.facts
                    if isinstance(fact, LineagedLiveFact)
                    and isinstance(fact.fact, CommandCleanupFailed)
                ]
            )
            == 1
        )
        service.fail = False
        service.release()
        async with controller.open_command_target(route, "child") as reopened:
            result = await reopened.cancel("child")
        assert result.run.stop_reason is RunStopReason.FAILED
        assert result.error is not None and result.error.code == "PROVIDER_ERROR"
        assert provider.calls == 1
        assert "child" not in runner._command_lifecycle.pending
        failures = [
            fact.fact
            for fact in publisher.facts
            if isinstance(fact, LineagedLiveFact) and isinstance(fact.fact, CommandCleanupFailed)
        ]
        assert len(failures) == 1 and failures[0].run_id == "child"
    finally:
        service.fail = False
        service.release()
        await child.aclose()
        await runner.aclose()


@pytest.mark.asyncio
async def test_timer_terminal_child_hands_old_receipt_to_delayed_parent_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """child timer 已停止旧环境后，父 proxy 到期不能再停止独立 B 启动的新环境。"""
    original = AgentRunner._build_start_facts

    def child_deadline(
        self: AgentRunner,
        request: AgentRunRequest,
        *,
        options: AgentRunOptions | None = None,
    ) -> tuple[CreateRun, RuntimeCursor]:
        if self.runtime.environment.agent_config.name == "researcher":
            options = AgentRunOptions(
                limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(seconds=0.2))
            )
        return original(self, request, options=options)

    monkeypatch.setattr(AgentRunner, "_build_start_facts", child_deadline)
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
    service = ControlledService()
    service.release()
    bind_service(runner, service)
    controller = runner._subagent_controller
    assert controller is not None
    controller.parent_boundary = replace(
        controller.parent_boundary, command_binding=runner.runtime.environment.command_binding
    )

    async def independent_command(scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        assert scope.run_id == "independent"
        return CommandOutcome(
            CommandMode.DOCKER,
            CommandStatus.EXITED,
            0,
            "new environment",
            "",
            CommandOutputStats(15, 0, 15, 0, frozenset()),
            0,
            "/workspace",
        )

    monkeypatch.setattr(service, "execute", independent_command)
    try:
        waiting = await runner.start(
            AgentRunRequest(input="parent", run_id="parent", session_id="parent-session"),
            options=AgentRunOptions(
                runtime=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
            ),
        )
        proxy = waiting.pending_interaction
        child_id = proxy.request.subagent_origin.child_run_id
        timer = runner._command_lifecycle.deadlines[child_id].task
        await asyncio.wait_for(asyncio.shield(timer), 2)
        assert runner.get_run(child_id).stop_reason is RunStopReason.DEADLINE_EXCEEDED
        assert child_id in runner._command_lifecycle.settled_receipts
        await service.execute(
            CommandScope("independent", "other-session"),
            CommandRequest("b", ShellCommand("echo next"), tmp_path, 10),
        )
        result = await runner.recover("parent")
        assert result.run.stop_reason is RunStopReason.FAILED
        assert [scope.run_id for scope in service.calls] == [child_id]
        assert not runner._command_lifecycle.settled_receipts
    finally:
        await runner.aclose()
