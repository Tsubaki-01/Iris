"""Goal CLI 的最小语法、后台派发与真实两轮执行。"""

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from iris.agents import ToolsConfig
from iris.cli.chat import (
    ChatOptions,
    _ChatLiveOutput,
    _ChatSessionHost,
    _parse_goal_command,
    run_chat_loop,
)
from iris.goal.models import GoalChanged
from iris.harness import AgentRunner
from iris.lifecycle import RunErrorInfo, RunPhase
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from tests.harness.fakes import StaticProvider, build_runtime, text_response, tool_response
from tests.harness.test_goal_execution import goal_config


@pytest.mark.parametrize(
    ("text", "operation", "objective", "rounds"),
    [
        ('修复  "空输入" 并测试', "create", '修复  "空输入" 并测试', None),
        ('-- status  "说明"', "create", 'status  "说明"', None),
        ('edit 修改  "正文"', "edit", '修改  "正文"', None),
        ("edit -- --max-rounds  这个是正文", "edit", "--max-rounds  这个是正文", None),
        ("edit --max-rounds 3", "edit", None, 3),
        ("status", "status", None, None),
        ("pause", "pause", None, None),
        ("resume", "resume", None, None),
        ("complete", "complete", None, None),
        ("clear", "clear", None, None),
    ],
)
def test_goal_parser_preserves_text_and_has_explicit_controls(
    text: str, operation: str, objective: str | None, rounds: int | None
) -> None:
    """只拆解命令前缀，正文的引号和内部空格不参与 token 解析。"""
    command = _parse_goal_command(text)
    assert command is not None
    assert (command.operation, command.objective, command.max_rounds) == (
        operation,
        objective,
        rounds,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "--",
        "status extra",
        "pause now",
        "resume id",
        "complete x",
        "clear x",
        "edit",
        "edit --",
        "edit --other x",
        "edit --max-rounds",
        "edit --max-rounds 0",
        "edit --max-rounds -2",
        "edit --max-rounds 2 extra",
        "--max-rounds 3",
    ],
)
def test_invalid_goal_arguments_are_not_objectives(text: str) -> None:
    """保留词误用和未知选项只给用法，不退化成目标正文或 steer。"""
    assert _parse_goal_command(text) is None


@pytest.mark.parametrize("use_live_output", [False, True])
def test_chat_goal_reaches_completed_after_two_bound_runs(
    tmp_path: Path, use_live_output: bool
) -> None:
    """真实 CLI host/manager/runtime/store 完成两轮并收到最终 GoalChanged。"""
    store = InMemoryLifecycleStore()
    provider_threads: set[int] = set()

    class Provider(StaticProvider):
        """在第二轮用真实 Goal 身份申报完成。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """第一轮无报告，第二轮报告后输出最终正文。"""
            provider_threads.add(threading.get_ident())
            self.requests.append(request)
            if len(self.requests) == 2:
                goal = store.get_current_goal("cli")
                assert goal is not None
                return tool_response(
                    ToolUseBlock(
                        id="report",
                        name="report_goal",
                        input={
                            "goal_id": goal.goal_id,
                            "revision": goal.revision,
                            "decision": "complete",
                            "reason": "结果已检查",
                        },
                    )
                )
            return text_response("本轮结果")

    provider = Provider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    finished = threading.Event()
    output: list[str] = []
    errors: list[str] = []
    initial = True

    def read_input(prompt: str) -> str:
        """输入目标后等待后台最终通知，主输入线程不承担执行。"""
        nonlocal initial
        if initial:
            initial = False
            return '/goal 修复  "空输入" 并测试'
        assert finished.wait(5)
        return "/exit"

    def write(message: str) -> None:
        """收到目标完成而非单轮文本时允许 CLI 退出。"""
        output.append(message)
        if "Goal: completed" in message:
            finished.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml", max_steps=4),
        live_output=_ChatLiveOutput(write) if use_live_output else None,
        input_func=read_input,
        output_func=write,
        error_func=errors.append,
    )
    goal = store.get_current_goal("cli")
    assert code == 0 and errors == []
    assert goal.status == "completed" and goal.rounds_started == 2
    assert goal.objective == '修复  "空输入" 并测试'
    assert goal.run_options.limits.max_model_steps == 4
    assert goal.run_options.runtime.include_tools
    assert len(store._runs) == 2
    assert all(run.phase is RunPhase.TERMINAL for run in store._runs.values())
    assert all(store.get_goal_run(run_id).goal_id == goal.goal_id for run_id in store._runs)
    assert len(store.load_session("cli").messages) >= 6
    assert provider_threads and threading.get_ident() not in provider_threads
    assert any("轮数: 2/20" in text for text in output)


def test_goal_controls_use_host_loop_and_preserve_active_run(tmp_path: Path) -> None:
    """所有命令走真实 GoalSession；编辑不清次数、不取消当前执行。"""
    started = threading.Event()

    class Provider(StaticProvider):
        """保留一个在途模型调用供同步 CLI 操作。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """等待 host 关闭时由 run cancellation 收口。"""
            self.requests.append(request)
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("模型应被取消")

    provider = Provider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    output: list[str] = []
    errors: list[str] = []
    host = _ChatSessionHost(
        runner, ChatOptions(config_path=tmp_path / "agent.yaml"), None, output.append, errors.append
    )
    host.start()
    try:
        host.goal_command('-- status  "原正文"')
        assert started.wait(5)
        goal = runner._goal_service.get_current("cli")
        run_id = runner.store.load_session_lane("cli")
        host.goal_command("status")
        host.goal_command("pause")
        host.goal_command('edit 新正文  "引号"')
        host.goal_command("edit --max-rounds 5")
        host.goal_command("edit -- --max-rounds 是正文")
        edited = runner._goal_service.get_current("cli")
        assert edited.goal_id == goal.goal_id
        assert edited.objective == "--max-rounds 是正文"
        assert edited.rounds_started == 1 and edited.max_rounds == 5 and edited.status == "paused"
        assert runner.get_run(run_id).cancellation_requested_at is None
        host.goal_command("resume")
        assert runner._goal_service.get_current("cli").status == "active"
        host.goal_command("complete")
        assert runner._goal_service.get_current("cli").status == "completed"
        host.goal_command("clear")
        assert runner._goal_service.get_current("cli") is None
        assert runner.get_run(run_id).cancellation_requested_at is None
        host.goal_command("status")
        assert any("Goal: absent" in message for message in output)
        assert any("当前执行可以继续收尾" in message for message in output)
        assert any(run_id in message for message in output)
        assert errors == []
    finally:
        host.close()


def test_invalid_and_disabled_goal_commands_never_become_chat_input(tmp_path: Path) -> None:
    """真实主输入循环消费 /goal 指令，关闭能力只提示配置且无 Run。"""
    provider = StaticProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    commands = iter(["/help", "/goal", "/goal status extra", "/goal 创建目标", "/exit"])
    output: list[str] = []
    errors: list[str] = []
    assert (
        run_chat_loop(
            runner=runner,
            options=ChatOptions(config_path=tmp_path / "agent.yaml"),
            input_func=lambda prompt: next(commands),
            output_func=output.append,
            error_func=errors.append,
        )
        == 0
    )
    assert not provider.requests and not store._runs
    assert any("/goal" in message for message in output)
    assert any("goal.enabled" in message for message in output)
    assert any("用法" in message for message in output)
    assert errors == []


def test_live_goal_output_keeps_pending_and_error_information(tmp_path: Path) -> None:
    """Live 输出与 status 使用同一领域视图，可显示无 Run 的状态。"""
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=StaticProvider())
    service = runner._goal_service
    service.create("cli", "目标正文")
    view = replace(
        service.get_view("cli"),
        settlement_pending=True,
        driver_error=RunErrorInfo(
            code="GOAL_PERSISTENCE_ERROR", source="persistence", message="写失败"
        ),
    )
    output: list[str] = []
    live = _ChatLiveOutput(output.append)
    live.publish(GoalChanged(session_id="cli", view=view))
    text = "".join(output)
    assert "目标正文" in text and "待结算" in text and "写失败" in text
    assert "自动推进" in text
    live.close()
    asyncio.run(runner.aclose())


def test_goal_resume_displays_exact_existing_activation_for_sdk_recovery(tmp_path: Path) -> None:
    """新 CLI 不自动接管遗留 ACTIVE，恢复提示保留精确身份并说明 SDK 入口。"""
    provider = StaticProvider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    goal = runner._goal_service.create("cli", "需要显式恢复")
    runner._admit_goal_start(goal.ref, run_id="existing-run")
    activation_id = runner.get_run("existing-run").current_activation_id
    output: list[str] = []
    errors: list[str] = []
    host = _ChatSessionHost(
        runner, ChatOptions(config_path=tmp_path / "agent.yaml"), None, output.append, errors.append
    )
    host.start()
    try:
        host.goal_command("status")
        host.goal_command("resume")
        text = "\n".join(output)
        assert "显式 SDK 恢复" in text
        assert "existing-run" in text and activation_id in text
        assert "expected_activation_id=" in text
        assert "自动推进: 关闭" in text
        assert not provider.requests and errors == []
    finally:
        host.close()


def test_goal_create_respects_cli_no_tools_without_saving_invalid_goal(tmp_path: Path) -> None:
    """CLI 将当前 include_tools 传给 SDK，不绕过 Goal 的模型工具契约。"""
    provider = StaticProvider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    commands = iter(["/goal 使用当前CLI选项", "/exit"])
    errors: list[str] = []
    assert (
        run_chat_loop(
            runner=runner,
            options=ChatOptions(config_path=tmp_path / "agent.yaml", include_tools=False),
            input_func=lambda prompt: next(commands),
            output_func=lambda message: None,
            error_func=errors.append,
        )
        == 0
    )
    assert len(errors) == 1 and "include_tools" in errors[0]
    assert runner._goal_service.get_current("cli") is None
    assert not provider.requests


def test_goal_resume_waiting_restores_original_prompt_and_answer_route(tmp_path: Path) -> None:
    """重新附着 WAITING 后，下一行回答原 interaction，不误当普通聊天或新一轮。"""
    store = InMemoryLifecycleStore()
    config = goal_config(tmp_path).model_copy(update={"tools": ToolsConfig(builtin=["human.ask"])})

    async def leave_waiting() -> str:
        """前一进程保存有原问题的 Goal Run。"""
        first = AgentRunner.from_config(
            config,
            store=store,
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(
                        id="ask", name="ask_question", input={"question": "原问题：是否继续？"}
                    )
                )
            ),
        )
        goal = first._goal_service.create("cli", "保留原人工交互", max_rounds=1)
        result = await first._start_goal_managed(goal.ref, run_id="waiting")
        assert result.run.phase is RunPhase.WAITING
        interaction_id = result.pending_interaction.interaction_id
        await first.aclose()
        return interaction_id

    interaction_id = asyncio.run(leave_waiting())
    provider = StaticProvider(text_response("回答收到"))
    runner = AgentRunner.from_config(config, store=store, provider=provider)
    output: list[str] = []
    errors: list[str] = []
    finished = threading.Event()

    def write(text: str) -> None:
        """等原 Run 输出结果后才检查持久事实。"""
        output.append(text)
        if text == "回答收到":
            finished.set()

    host = _ChatSessionHost(
        runner, ChatOptions(config_path=tmp_path / "agent.yaml"), None, write, errors.append
    )
    host.start()
    try:
        host.goal_command("resume")
        assert any("原问题：是否继续？" in text for text in output)
        assert not provider.requests
        host.goal_command("pause")
        host.submit("确认继续")
        assert finished.wait(5)
        assert store.load_interaction(interaction_id).response.answer == "确认继续"
        assert runner.get_run("waiting").phase is RunPhase.TERMINAL
        assert len(store._runs) == 1
        goal = store.get_current_goal("cli")
        assert goal.rounds_started == 1 and goal.status == "paused"
        assert errors == []
    finally:
        host.close()
