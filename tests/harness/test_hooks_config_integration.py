"""真实 YAML/Python 工厂、宿主脚本与完整 Runner 的组合验收。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from iris.exceptions import IrisCommandCleanupError
from iris.harness import AgentRunner, SessionManager
from iris.hitl import QuestionInteractionResponse
from iris.hooks import HookRegistration
from iris.lifecycle import AgentRunRequest, RunPhase, RunStopReason
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import SQLiteStore

from ..hooks.config_extensions import (
    RecordingMiddleware,
    command_hook,
    create_hook,
    python_hook,
    records,
    script_command,
    write_config,
    write_hook_script,
)
from .fakes import StaticProvider, text_response, tool_response
from .test_child_run_hooks import FinishedFailureService
from .test_command_settlement import bind_service
from .test_runner_subagent import ChildProviders, _parent_provider, _write_configs
from .test_session_manager import _wait_until


@pytest.mark.asyncio
async def test_yaml_mixed_hooks_sdk_append_filters_and_middleware_order(tmp_path: Path) -> None:
    """四事件及两工具通过真实配置；别名过滤不命中，工厂跨 session 仅构造一次。"""
    order = tmp_path / "order.jsonl"
    (tmp_path / "source.txt").write_text("file body", encoding="utf-8")
    (tmp_path / "body.py").write_text("print('command body')\n", encoding="utf-8")
    script = script_command(write_hook_script(tmp_path))
    path = write_config(
        tmp_path,
        [
            python_hook(order, "run.started", "started"),
            python_hook(order, "tool.before", "python-before", tools=["exec_command"]),
            command_hook("tool.before", script, tools=["exec_command"]),
            python_hook(order, "tool.before", "python-later", tools=["exec_command"]),
            python_hook(order, "tool.before", "wrong-alias", tools=["exec.command"]),
            python_hook(order, "tool.after", "python-after", feedback="Python反馈"),
            command_hook("tool.after", script, tools=["exec_command"]),
            python_hook(order, "run.finished", "finished"),
        ],
        tools={"builtin": ["exec.command", "file.read"]},
        middleware={
            "tools": [
                {
                    "factory": "tests.hooks.config_extensions:create_middleware",
                    "options": {"path": str(order), "label": "yaml-middleware"},
                }
            ]
        },
    )
    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="command",
                name="exec_command",
                input={"command": script_command(tmp_path / "body.py")},
            )
        ),
        tool_response(ToolUseBlock(id="read", name="read_file", input={"file_path": "source.txt"})),
        text_response("completed"),
        text_response("second session"),
    )
    runner = AgentRunner.from_config_path(
        path,
        provider=provider,
        hooks=[
            HookRegistration(
                event="tool.before",
                name="sdk",
                handler=create_hook(path=str(order), label="sdk"),
                tool_names=("exec_command",),
            )
        ],
        tool_middlewares=[RecordingMiddleware(str(order), "sdk-middleware")],
    )
    try:
        result = await runner.start(
            AgentRunRequest(input="two tools", run_id="first", session_id="s")
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
        assert records(order) == [
            "started:run.started:first",
            "python-before:tool.before:first",
            "script:tool.before:first",
            "python-later:tool.before:first",
            "sdk:tool.before:first",
            "yaml-middleware:before:exec_command",
            "sdk-middleware:before:exec_command",
            "sdk-middleware:after:exec_command",
            "yaml-middleware:after:exec_command",
            "python-after:tool.after:first",
            "script:tool.after:first",
            "yaml-middleware:before:read_file",
            "sdk-middleware:before:read_file",
            "sdk-middleware:after:read_file",
            "yaml-middleware:after:read_file",
            "python-after:tool.after:first",
            "finished:run.finished:first",
        ]
        record = runner.store.load_tool_call("first", "command")
        assert record is not None and record.result is not None
        assert record.result.hook_feedback == ("Python反馈", "脚本反馈")
        delivered = [
            item for message in provider.requests[1].messages for item in message.tool_results
        ]
        assert len(delivered) == 1
        assert delivered[0].text == record.result.model_content
        assert "diagnostic-only" not in delivered[0].text
        assert "command body" in delivered[0].text
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["tool.before", "tool.after"]
        assert all(event["call_id"] == "command" for event in events)
        assert events[1]["body_status"] == "success"
        await runner.start(AgentRunRequest(input="reuse", run_id="second", session_id="s2"))
        created = records(Path(str(order) + ".created"))
        assert len(created) == len(set(created)) == 8
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_yaml_hitl_resume_does_not_repeat_run_started_or_finish_waiting(
    tmp_path: Path,
) -> None:
    """真实 human.ask 暂停不发 finished，恢复沿原注册实例继续。"""
    order = tmp_path / "order.jsonl"
    path = write_config(
        tmp_path,
        [python_hook(order, event, event) for event in ("run.started", "run.finished")],
        tools={"builtin": ["human.ask"]},
    )
    runner = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="q", name="ask_question", input={"question": "继续？"})),
            text_response("done"),
        ),
    )
    try:
        waiting = await runner.start(AgentRunRequest(input="ask", run_id="run"))
        assert waiting.run.phase is RunPhase.WAITING and waiting.pending_interaction is not None
        assert records(order) == ["run.started:run.started:run"]
        result = await runner.resume(
            "run",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert records(order) == ["run.started:run.started:run", "run.finished:run.finished:run"]
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_yaml_sqlite_recovery_reuses_committed_feedback_without_replay(
    tmp_path: Path,
) -> None:
    """新 Runner 重建真实工厂但不重放 START 或已提交工具处理器。"""
    order = tmp_path / "order.jsonl"
    (tmp_path / "source.txt").write_text("durable body", encoding="utf-8")
    path = write_config(
        tmp_path,
        [
            python_hook(order, "run.started", "start"),
            python_hook(order, "tool.before", "before"),
            python_hook(order, "tool.after", "after", feedback="恢复反馈"),
            python_hook(order, "run.finished", "finish"),
        ],
        tools={"builtin": ["file.read"]},
        session={"backend": "sqlite", "path": str(tmp_path / "session.db")},
    )
    entered = asyncio.Event()

    class PauseAfterCommit(StaticProvider):
        """模型第二步已能看到落盘结果时暂停。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            if not self.requests:
                return await super().complete(request)
            self.requests.append(request)
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("测试 provider 应由 SDK 取消")

    runner = AgentRunner.from_config_path(
        path,
        provider=PauseAfterCommit(
            tool_response(
                ToolUseBlock(id="read", name="read_file", input={"file_path": "source.txt"})
            )
        ),
    )
    task = asyncio.create_task(
        runner.start(AgentRunRequest(input="read", run_id="run", session_id="s"))
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await runner.aclose()
    assert records(order) == [
        "start:run.started:run",
        "before:tool.before:run",
        "after:tool.after:run",
    ]
    provider = StaticProvider(text_response("recovered"))
    resumed = AgentRunner.from_config_path(path, provider=provider)
    try:
        assert isinstance(resumed.store, SQLiteStore)
        active = resumed.get_run("run")
        saved = resumed.store.load_tool_call("run", "read")
        result = await resumed.recover("run", expected_activation_id=active.current_activation_id)
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert saved is not None and saved.result is not None
        assert saved.result.hook_feedback == ("恢复反馈",)
        delivered = [
            item for message in provider.requests[0].messages for item in message.tool_results
        ]
        assert len(delivered) == 1 and delivered[0].text == saved.result.model_content
        assert records(order) == [
            "start:run.started:run",
            "before:tool.before:run",
            "after:tool.after:run",
            "finish:run.finished:run",
        ]
        assert await resumed.recover("run") == result
        assert len(records(order)) == 4
    finally:
        await resumed.aclose()


@pytest.mark.asyncio
async def test_parent_sdk_hooks_do_not_inherit_into_child_yaml(tmp_path: Path) -> None:
    """实际委派绕过父工具链，child YAML 独立构造并借用同一个命令服务。"""
    path = _write_configs(tmp_path)
    order = tmp_path / "order.jsonl"
    (tmp_path / "source.txt").write_text("child body", encoding="utf-8")
    child_path = tmp_path / "child.yaml"
    child = yaml.safe_load(child_path.read_text(encoding="utf-8"))
    child["hooks"] = [
        python_hook(order, event, "child")
        for event in ("run.started", "tool.before", "tool.after", "run.finished")
    ]
    child["middleware"] = {
        "tools": [
            {
                "factory": "tests.hooks.config_extensions:create_middleware",
                "options": {"path": str(order), "label": "child-middleware"},
            }
        ]
    }
    child_path.write_text(yaml.safe_dump(child), encoding="utf-8")
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    root["permissions"] = {"workspace": str(tmp_path)}
    path.write_text(yaml.safe_dump(root), encoding="utf-8")
    runner = AgentRunner.from_config_path(
        path,
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="read", name="read_file", input={"file_path": "source.txt"})
                ),
                text_response("child done"),
            )
        ),
        hooks=[
            HookRegistration(
                event=event, name="parent", handler=create_hook(path=str(order), label="parent")
            )
            for event in ("run.started", "tool.before", "tool.after", "run.finished")
        ],
        tool_middlewares=[RecordingMiddleware(str(order), "parent-middleware")],
    )
    try:
        result = await runner.start(AgentRunRequest(input="delegate", run_id="parent"))
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
        link = runner.store.load_subagent_link("parent", "delegate")
        assert link is not None
        assert records(order) == [
            "parent:run.started:parent",
            f"child:run.started:{link.child_run_id}",
            f"child:tool.before:{link.child_run_id}",
            "child-middleware:before:read_file",
            "child-middleware:after:read_file",
            f"child:tool.after:{link.child_run_id}",
            f"child:run.finished:{link.child_run_id}",
            "parent:run.finished:parent",
        ]
        controller = runner._subagent_controller
        assert controller is not None
        assert (
            controller.parent_boundary.command_binding is runner.runtime.environment.command_binding
        )
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_child_yaml_finished_cleanup_blocks_root_but_preserves_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实 child 配置脚本的资源失败锁 root 准入，不改终态或重放脚本。"""
    path = _write_configs(tmp_path)
    child_path = tmp_path / "child.yaml"
    config = yaml.safe_load(child_path.read_text(encoding="utf-8"))
    config["hooks"] = [command_hook("run.finished", "configured-child-finished-command")]
    child_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    provider = StaticProvider(text_response("must not start"))
    runner = AgentRunner.from_config_path(
        path,
        provider=provider,
        child_provider_factory=ChildProviders(StaticProvider(text_response("child done"))),
    )
    service = FinishedFailureService()
    service.release()
    service.fail = True
    execute = AsyncMock(wraps=service.execute)
    monkeypatch.setattr(service, "execute", execute)
    bind_service(runner, service)
    controller = runner._subagent_controller
    assert controller is not None
    controller.parent_boundary = replace(
        controller.parent_boundary, command_binding=runner.runtime.environment.command_binding
    )
    child = controller._assemble_child(controller.routes.routes["researcher"])
    try:
        with pytest.raises(IrisCommandCleanupError) as failure:
            await child.start(AgentRunRequest(input="child", run_id="child", session_id="child-s"))
        saved = runner.store.load_result("child")
        assert saved is not None and saved.run.stop_reason is RunStopReason.COMPLETED
        assert saved.assistant_message is not None and saved.assistant_message.text == "child done"
        assert (
            child.runtime.environment.command_binding is runner.runtime.environment.command_binding
        )
        assert execute.await_count == 1
        assert execute.await_args.args[1].payload.command == "configured-child-finished-command"
        with pytest.raises(IrisCommandCleanupError) as rejected:
            await runner.start(AgentRunRequest(input="new", run_id="new", session_id="root-s"))
        assert rejected.value is failure.value
        assert runner.store.load_run("new") is None and provider.requests == []
        service.fail = False
        assert await child.recover("child") == saved
        assert await child.cancel("child") == saved
        assert execute.await_count == 1
        await child.aclose()
        assert service.close_calls == 0
        await runner.aclose()
        assert service.close_calls == 1
        assert runner.store.load_result("child") == saved
    finally:
        service.fail = False
        await child.aclose()
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("next_action", ["input", "goal"])
async def test_yaml_finished_script_holds_manager_and_goal_admission(
    tmp_path: Path, next_action: str
) -> None:
    """真实 finished 进程持有完成任务，文件放行后 Manager/Goal 自动继续。"""
    path = write_config(
        tmp_path,
        [command_hook("run.finished", script_command(write_hook_script(tmp_path, gate=True)))],
        goal={"enabled": True},
        context_policy={"enabled": True},
        command={"timeout_seconds": 0.001},
    )
    provider = StaticProvider(text_response("ordinary"), text_response("next"))
    runner = AgentRunner.from_config_path(path, provider=provider)
    manager = SessionManager(runner, "s")
    direct = asyncio.create_task(
        runner.start(AgentRunRequest(input="ordinary", run_id="first", session_id="s"))
    )
    submitted: asyncio.Task | None = None
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "entered").exists():
                await asyncio.sleep(0.01)
        assert runner.get_run("first").stop_reason is RunStopReason.COMPLETED
        if next_action == "goal":
            await manager.goal.create("one round", max_rounds=1)
            assert runner._goal_service.get_current("s").rounds_started == 0
        else:
            submitted = asyncio.create_task(manager.submit("next"))
        await asyncio.sleep(0)
        assert len(provider.requests) == 1
        assert submitted is None or not submitted.done()
        (tmp_path / "release").touch()
        await asyncio.wait_for(direct, 5)
        if submitted is not None:
            await asyncio.wait_for(submitted, 5)
        await _wait_until(lambda: len(provider.requests) == 2)
        if next_action == "goal":
            assert runner._goal_service.get_current("s").rounds_started == 1
    finally:
        (tmp_path / "release").touch()
        if submitted is not None:
            await asyncio.gather(submitted, return_exceptions=True)
        await asyncio.gather(direct, return_exceptions=True)
        await manager.close(cancel_run=True)
        await runner.aclose()
