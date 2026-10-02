"""真实 child Run 的独立事件与 root 持有的结束处理生命周期。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.command import CommandOutcome, CommandRequest, CommandScope, CommandStopReceipt
from iris.exceptions import IrisCommandCleanupError, IrisToolOutcomeUnknownError
from iris.harness import AgentRunner
from iris.hooks import HookEvent, HookRegistration, RunFinishedEvent
from iris.hooks._dispatch_types import CommandHookRegistration
from iris.hooks.command import CommandHookAdapter
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import ToolUseBlock
from iris.tools.subagent import SubagentRoute

from .fakes import StaticProvider, text_response, tool_response
from .test_command_settlement import ControlledService, bind_service
from .test_runner_subagent import ChildProviders, _parent_provider, _write_configs


def _child_hooks(
    runner: AgentRunner,
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], HookDispatcher],
) -> list[AgentRunner]:
    """在真实 child assembly 后注入自身配置；重建同样获得新处理器实例。"""
    controller = runner._subagent_controller
    assert controller is not None
    original = controller._assemble_child
    children: list[AgentRunner] = []

    def assemble(route: SubagentRoute) -> AgentRunner:
        child = original(route)
        child.runtime.environment = replace(child.runtime.environment, hook_dispatcher=factory())
        children.append(child)
        return child

    monkeypatch.setattr(controller, "_assemble_child", assemble)
    return children


@pytest.mark.asyncio
async def test_parent_and_actual_admitted_child_have_independent_run_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实际 parent委派触发各自START/FINISHED，不继承父处理器或重复开始。"""
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(StaticProvider(text_response("child done"))),
    )
    seen: list[tuple[str, str, str | None]] = []

    def handlers(owner: str) -> HookDispatcher:
        async def record(event: HookEvent) -> None:
            seen.append((owner, event.event, event.run_id))

        return HookDispatcher(
            [
                HookRegistration(event="run.started", name=owner + "-start", handler=record),
                HookRegistration(event="run.finished", name=owner + "-finish", handler=record),
            ]
        )

    runner.runtime.environment = replace(
        runner.runtime.environment, hook_dispatcher=handlers("parent")
    )
    children = _child_hooks(runner, monkeypatch, lambda: handlers("child"))
    try:
        result = await runner.start(AgentRunRequest(input="go", run_id="parent"))
        assert result.run.stop_reason is RunStopReason.COMPLETED
        link = runner.store.load_subagent_link("parent", "delegate")
        assert link is not None
        assert seen == [
            ("parent", "run.started", "parent"),
            ("child", "run.started", link.child_run_id),
            ("child", "run.finished", link.child_run_id),
            ("parent", "run.finished", "parent"),
        ]
        assert len(children) == 1 and children[0]._resources_closed
        assert children[0]._hook_lifecycle is runner._hook_lifecycle
        assert (await runner.recover("parent")) == result
        assert len(seen) == 4
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_rebuilt_waiting_child_finished_is_joined_by_root_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """临时child已关闭，后台期限删除timer后，root仍持有并等待实际finished任务。"""
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "continue?"}))
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(provider),
    )
    service = ControlledService()
    service.release()
    bind_service(runner, service)
    controller = runner._subagent_controller
    assert controller is not None
    controller.parent_boundary = replace(
        controller.parent_boundary,
        command_binding=runner.runtime.environment.command_binding,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[HookEvent] = []

    def handlers() -> HookDispatcher:
        async def record_start(event: HookEvent) -> None:
            seen.append(event)

        async def wait_finished(event: HookEvent) -> None:
            seen.append(event)
            entered.set()
            await release.wait()

        return HookDispatcher(
            [
                HookRegistration(event="run.started", name="start", handler=record_start),
                HookRegistration(event="run.finished", name="finish", handler=wait_finished),
            ]
        )

    children = _child_hooks(runner, monkeypatch, handlers)
    child = controller._assemble_child(controller.routes.routes["researcher"])
    closing: asyncio.Task[None] | None = None
    try:
        waiting = await child.start(
            AgentRunRequest(input="child", run_id="child", session_id="child-session"),
            options=AgentRunOptions(
                limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(seconds=0.2))
            ),
        )
        assert waiting.run.phase is RunPhase.WAITING
        assert [event.event for event in seen] == ["run.started"]
        await child.aclose()
        assert child._resources_closed
        await asyncio.wait_for(entered.wait(), 3)
        assert len(children) > 1 and children[-1] is not child
        assert all(rebuilt._hook_lifecycle is runner._hook_lifecycle for rebuilt in children)
        assert "child" not in runner._command_lifecycle.deadlines
        assert "child" not in runner._command_lifecycle.pending
        assert runner.get_run("child").stop_reason is RunStopReason.DEADLINE_EXCEEDED
        assert isinstance(seen[-1], RunFinishedEvent)
        closing = asyncio.create_task(runner.aclose())
        await asyncio.sleep(0)
        assert not closing.done() and service.close_calls == 0
        release.set()
        await asyncio.wait_for(closing, 3)
        assert service.close_calls == 1
        assert [event.event for event in seen] == ["run.started", "run.finished"]
        assert len(provider.requests) == 1
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await child.aclose()
        await runner.aclose()


class FinishedFailureService(ControlledService):
    """命令 Hook 前台状态未知，停止收据的排空暂时失败。"""

    async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        """通过真实 adapter 路径交出 unknown 和 receipt。"""
        raise IrisToolOutcomeUnknownError("finished command unknown", stop_receipt=self.receipt)

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        """只由资源 owner 判断收口是否完成。"""
        if self.fail:
            raise IrisCommandCleanupError("finished command cleanup pending")
        await super().wait_drained(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["root", "child"])
async def test_child_finished_cleanup_blocks_new_admission_after_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admission: str
) -> None:
    """child 的终态保持COMPLETED；prepare中的root/child均在durable准入前拒绝。"""
    root_provider = (
        StaticProvider(text_response("must not start"))
        if admission == "root"
        else _parent_provider()
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=root_provider,
        child_provider_factory=ChildProviders(StaticProvider(text_response("child done"))),
    )
    service = FinishedFailureService()
    service.release()
    service.fail = True
    bind_service(runner, service)
    controller = runner._subagent_controller
    assert controller is not None
    binding = runner.runtime.environment.command_binding
    assert binding is not None
    controller.parent_boundary = replace(controller.parent_boundary, command_binding=binding)

    _child_hooks(
        runner,
        monkeypatch,
        lambda: HookDispatcher(
            [
                CommandHookRegistration(
                    "run.finished",
                    "failure",
                    CommandHookAdapter(
                        binding=binding,
                        workspace=tmp_path,
                        command="configured-finished-command",
                    ),
                ),
            ]
        ),
    )
    child = controller._assemble_child(controller.routes.routes["researcher"])
    preparing = asyncio.Event()
    release = asyncio.Event()

    def delay_prepare(target: AgentRunner) -> None:
        prepare = target.aprepare

        async def delayed_prepare() -> None:
            preparing.set()
            await release.wait()
            await prepare()

        monkeypatch.setattr(target, "aprepare", delayed_prepare)

    if admission == "root":
        delay_prepare(runner)
    else:
        assemble = controller._assemble_child

        def delayed_assemble(route: SubagentRoute) -> AgentRunner:
            fresh = assemble(route)
            delay_prepare(fresh)
            return fresh

        monkeypatch.setattr(controller, "_assemble_child", delayed_assemble)
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="new", run_id="new-root", session_id="new-session"),
        )
    )
    try:
        await asyncio.wait_for(preparing.wait(), 2)
        with pytest.raises(IrisCommandCleanupError) as child_error:
            await child.start(
                AgentRunRequest(input="child", run_id="child", session_id="child-session")
            )
        assert runner.get_run("child").stop_reason is RunStopReason.COMPLETED
        assert "child" not in runner._command_lifecycle.pending
        release.set()
        with pytest.raises(IrisCommandCleanupError) as admission_error:
            await asyncio.wait_for(task, 2)
        assert admission_error.value is child_error.value
        if admission == "root":
            assert runner.store.load_run("new-root") is None
            assert root_provider.requests == []
        else:
            assert runner.store.load_subagent_link("new-root", "delegate") is None
            assert len(root_provider.requests) == 1
        # 资源可恢复不自动解除准入错误；既有结果仍可读取，close仍可用。
        service.fail = False
        result = await child.recover("child")
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert (await child.cancel("child")) == result
        if admission == "child":
            assert (await runner.recover("new-root")).run.stop_reason is RunStopReason.FAILED
        with pytest.raises(IrisCommandCleanupError) as still_blocked:
            await runner.start(AgentRunRequest(input="again", run_id="again"))
        assert still_blocked.value is child_error.value
    finally:
        release.set()
        service.fail = False
        await asyncio.gather(task, return_exceptions=True)
        await child.aclose()
        await runner.aclose()
