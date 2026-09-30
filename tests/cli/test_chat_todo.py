"""真实 chat 输入循环中的 Todo 查询、并发执行与待回答交互。"""

import asyncio
import threading
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.cli.chat import ChatOptions, run_chat_loop
from iris.harness import AgentRunner
from iris.hitl import InteractionStatus, PermissionInteractionResponse, QuestionInteractionResponse
from iris.lifecycle import RunPhase, RunStopReason
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from tests.harness.fakes import StaticProvider, text_response, tool_response


def _config(workspace: Path, *, enabled: bool = True, goal: bool = False) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "name": "todo-cli",
            "model": "openai/test",
            "system": "读取当前工作清单",
            "todo": {"enabled": enabled},
            "goal": {"enabled": goal, "max_rounds": 1},
            "permissions": {"workspace": str(workspace)},
            "tools": {"builtin": ["human.ask", "file.write"]},
        }
    )


def _todo_file(workspace: Path, content: str) -> Path:
    path = workspace / ".iris/todos/636c69.md"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_idle_todo_rereads_three_states_and_manual_edits(tmp_path: Path) -> None:
    """外层 slash 路由仅查询当前文件；每次人工修改都立即显示。"""
    path = _todo_file(tmp_path, "- [x] 完成事项\n- [-] 进行事项\n- [ ] 等待事项\n")
    provider = StaticProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=store)
    output: list[str] = []
    errors: list[str] = []
    index = 0

    def read_input(prompt: str) -> str:
        nonlocal index
        index += 1
        if index == 1:
            return "/todo"
        if index == 2:
            assert "Todo: 1/3 已完成" in output[-1]
            assert "[x] 完成事项\n[-] 进行事项\n[ ] 等待事项" in output[-1]
            path.write_text("- [x] 人工完成\n", encoding="utf-8")
            return "/todo"
        if index == 3:
            assert "Todo: 1/1 已完成" in output[-1] and "人工完成" in output[-1]
            path.write_text("", encoding="utf-8")
            return "/todo"
        if index == 4:
            assert "暂无待办" in output[-1]
            path.write_text("# Todo\n错误正文", encoding="utf-8")
            return "/todo"
        if index == 5:
            assert "第 2 行" in output[-1] and "0/0" not in output[-1]
            path.unlink()
            return "/todo"
        if index == 6:
            assert "暂无待办" in output[-1]
            return "/help"
        return "/exit"

    assert (
        run_chat_loop(
            runner=runner,
            options=ChatOptions(config_path=tmp_path / "agent.yaml", session_id=" cli "),
            input_func=read_input,
            output_func=output.append,
            error_func=errors.append,
        )
        == 0
    )
    assert all(str(path) in text for text in output[:5])
    assert any("/todo" in text and "待办" in text for text in output)
    assert provider.requests == [] and store._runs == {}
    assert store._sessions == {}
    assert not path.exists() and errors == []


@pytest.mark.parametrize("failure", ["disabled", "io"])
def test_todo_errors_and_argument_usage_leave_chat_available(tmp_path: Path, failure: str) -> None:
    """领域错误局部显示，用法与未知命令均不成为聊天输入。"""
    path = _todo_file(tmp_path, "- [x] 完成")
    if failure == "io":
        path.unlink()
        path.mkdir()
    provider = StaticProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(
        _config(tmp_path, enabled=failure != "disabled"), provider=provider, store=store
    )
    commands = iter(["/todo", "/todo extra", "/todo\tother", "/todoman", "/help", "/exit"])
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
    assert len(errors) == 1
    assert "TODO_ERROR" in errors[0]
    assert ("未启用" if failure == "disabled" else "读取失败") in errors[0]
    assert output.count("用法：/todo") == 2
    assert any("未知命令" in text for text in output)
    assert any("可用命令" in text for text in output)
    assert provider.requests == [] and store._runs == {}


@pytest.mark.parametrize("goal", [False, True])
def test_active_todo_does_not_steer_or_take_over_lane(tmp_path: Path, goal: bool) -> None:
    """模型在途时查看不提交输入，普通和 Goal Run 均保持同一 lane。"""
    started = threading.Event()
    finished = threading.Event()

    class Provider(StaticProvider):
        """使用事件保持一个确定的在途请求。"""

        def __init__(self) -> None:
            super().__init__()
            self.loop: asyncio.AbstractEventLoop | None = None
            self.release = asyncio.Event()

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """输入线程核对查询只读后才允许当前模型步骤结束。"""
            self.requests.append(request)
            self.loop = asyncio.get_running_loop()
            started.set()
            await self.release.wait()
            return text_response("在途执行完成")

    _todo_file(tmp_path, "- [x] 已核对")
    provider = Provider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(_config(tmp_path, goal=goal), provider=provider, store=store)
    output: list[str] = []
    errors: list[str] = []
    index = 0
    lane: str | None = None
    revision = 0

    def read_input(prompt: str) -> str:
        nonlocal index, lane, revision
        index += 1
        if index == 1:
            return "/goal 执行目标" if goal else "执行任务"
        if index == 2:
            assert started.wait(5)
            lane = store.load_session_lane("cli")
            revision = store.load_session_revision("cli")
            return "/todo"
        if index == 3:
            assert "Todo: 1/1 已完成" in output[-1]
            assert store.load_session_lane("cli") == lane
            assert store.load_session_revision("cli") == revision
            assert len(provider.requests) == 1
            assert provider.loop is not None
            provider.loop.call_soon_threadsafe(provider.release.set)
            assert finished.wait(5)
            return "/exit"
        raise AssertionError("不应产生额外输入")

    def write_output(message: str) -> None:
        output.append(message)
        if message == "在途执行完成":
            finished.set()

    assert (
        run_chat_loop(
            runner=runner,
            options=ChatOptions(config_path=tmp_path / "agent.yaml"),
            input_func=read_input,
            output_func=write_output,
            error_func=errors.append,
        )
        == 0
    )
    assert len(provider.requests) == 1 and len(store._runs) == 1
    assert all("/todo" not in message.text for message in store.load_session("cli").messages)
    assert errors == []


@pytest.mark.parametrize(
    ("interaction", "goal", "mode"),
    [
        ("question", False, "enabled"),
        ("permission", False, "enabled"),
        ("question", True, "enabled"),
        ("permission", True, "enabled"),
        ("question", False, "disabled"),
        ("question", False, "io"),
    ],
)
def test_todo_preserves_waiting_interaction_until_actual_answer(
    tmp_path: Path, interaction: str, goal: bool, mode: str
) -> None:
    """成功、失败和带参数的查看都不消费 pending，实际下一答案才能恢复。"""
    path = _todo_file(tmp_path, "- [x] 已核对")
    call = (
        ToolUseBlock(id="ask", name="ask_question", input={"question": "继续吗？"})
        if interaction == "question"
        else ToolUseBlock(
            id="write", name="write_file", input={"file_path": "result.txt", "content": "已批准"}
        )
    )
    provider = StaticProvider(tool_response(call), text_response("回答后完成"))
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(
        _config(tmp_path, enabled=mode != "disabled", goal=goal), provider=provider, store=store
    )
    prompted = threading.Event()
    finished = threading.Event()
    output: list[str] = []
    errors: list[str] = []
    index = 0
    run_id: str | None = None
    interaction_id: str | None = None
    revision = 0

    def read_input(prompt: str) -> str:
        nonlocal index, run_id, interaction_id, revision
        index += 1
        if index == 1:
            return "/goal 等待确认后执行" if goal else "请执行"
        if index == 2:
            assert prompted.wait(5)
            run_id = store.load_session_lane("cli")
            pending = runner.get_result(run_id).pending_interaction
            interaction_id = pending.interaction_id
            revision = store.load_session_revision("cli")
            if mode == "io":
                path.unlink()
                path.mkdir()
            return "/todo"
        if index in {3, 4}:
            assert runner.get_run(run_id).phase is RunPhase.WAITING
            assert store.load_session_lane("cli") == run_id
            assert store.load_session_revision("cli") == revision
            assert store.load_interaction(interaction_id).status is InteractionStatus.PENDING
            assert runner.get_result(run_id).pending_interaction.interaction_id == interaction_id
            assert len(provider.requests) == 1
            if index == 3:
                return "/todo\tignored"
            assert "用法：/todo" in output
            if mode == "enabled":
                assert any("Todo: 1/1 已完成" in text for text in output)
            else:
                assert len(errors) == 1 and "TODO_ERROR" in errors[0]
            if mode == "io":
                path.rmdir()
                path.write_text("- [x] 已核对", encoding="utf-8")
            return "实际答案" if interaction == "question" else "y"
        assert finished.wait(5)
        return "/exit"

    def write_output(message: str) -> None:
        output.append(message)
        if message in {"请输入回答。", "批准该调用？ [y/N]"}:
            prompted.set()
        if message == "回答后完成":
            finished.set()

    assert (
        run_chat_loop(
            runner=runner,
            options=ChatOptions(config_path=tmp_path / "agent.yaml"),
            input_func=read_input,
            output_func=write_output,
            error_func=errors.append,
        )
        == 0
    )
    assert len(provider.requests) == 2 and len(store._runs) == 1
    assert runner.get_run(run_id).stop_reason is RunStopReason.COMPLETED
    resolved = store.load_interaction(interaction_id)
    assert resolved.response == (
        QuestionInteractionResponse(answer="实际答案")
        if interaction == "question"
        else PermissionInteractionResponse(decision="approve")
    )
    assert all("/todo" not in message.text for message in store.load_session("cli").messages)
    if interaction == "permission":
        assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "已批准"
