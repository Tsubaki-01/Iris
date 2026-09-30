"""Todo 自查与真实 Goal 申报、文件工具结算的交互。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.goal import GoalStatus
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunOptions, RunLimits, RunStopReason
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.todo import TodoStatus
from tests.harness.fakes import StaticProvider, text_response, tool_response


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> InMemoryLifecycleStore | SQLiteStore:
    """让申报结算经由两个真实 store。"""
    return (
        SQLiteStore(tmp_path / "goal.db") if request.param == "sqlite" else InMemoryLifecycleStore()
    )


class _GoalProvider(StaticProvider):
    """按给定顺序返回真实 Goal 与文件工具调用。"""

    def __init__(
        self, store: InMemoryLifecycleStore | SQLiteStore, path: Path, actions: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.store = store
        self.path = path
        self.actions = actions

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """报告使用准入后当前版本，其余调用走真实工具执行链。"""
        index = len(self.requests)
        self.requests.append(request)
        action = self.actions[index]
        if action == "report":
            goal = self.store.get_current_goal("s")
            assert goal is not None
            return tool_response(
                ToolUseBlock(
                    id=f"report-{index}",
                    name="report_goal",
                    input={
                        "goal_id": goal.goal_id,
                        "revision": goal.revision,
                        "decision": "complete",
                        "reason": "已依据实际结果确认本轮目标",
                    },
                )
            )
        if action == "read":
            return tool_response(
                ToolUseBlock(
                    id=f"read-{index}", name="read_file", input={"file_path": str(self.path)}
                )
            )
        if action == "write":
            return tool_response(
                ToolUseBlock(
                    id=f"write-{index}",
                    name="write_file",
                    input={"file_path": str(self.path), "content": "- [x] 整理清单\n"},
                )
            )
        return text_response(action)


def _runner(workspace: Path, provider: _GoalProvider) -> AgentRunner:
    config = AgentConfig.model_validate(
        {
            "name": "goal-todo",
            "model": "openai/test",
            "system": "按实际结果执行目标与工作清单",
            "goal": {"enabled": True},
            "todo": {"enabled": True},
            "permissions": {"workspace": str(workspace), "writes": "allow"},
            "tools": {"builtin": ["file.read", "file.write"]},
        }
    )
    return AgentRunner.from_config(config, provider=provider, store=provider.store)


def _todo_file(workspace: Path, *, completed: bool = False) -> Path:
    path = workspace / ".iris/todos/73.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"- [{'x' if completed else ' '}] 整理清单\n", encoding="utf-8")
    return path


def _snapshot_text(request: LLMRequest) -> str:
    return "\n".join(
        message.text
        for message in request.messages
        if message.metadata.get("context_kind") == "runtime_snapshot"
    )


@pytest.mark.asyncio
async def test_plain_todo_check_does_not_supersede_report(
    tmp_path: Path, store: InMemoryLifecycleStore | SQLiteStore
) -> None:
    """仅自查是同一 Run 的动态指令，不使真实 Goal 完成报告过时。"""
    provider = _GoalProvider(store, _todo_file(tmp_path), ("report", "candidate", "final"))
    runner = _runner(tmp_path, provider)
    try:
        service = runner._goal_service
        goal = service.create("s", "完成目标")
        result = await runner._start_goal_managed(goal.ref, run_id="goal-run")
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert len(provider.requests) == 3
        assert ["Todo 结束自查" in _snapshot_text(request) for request in provider.requests] == [
            False,
            False,
            True,
        ]
        [report] = runner.list_tool_calls("goal-run")
        assert report.tool_name == "report_goal" and not report.result.is_error
        settlement = service.settle_run("goal-run", now=runner._now())
        assert settlement.goal.status is GoalStatus.COMPLETED
        assert settlement.binding.applied_report_call_id == "report-0"
        assert (await runner.get_todo("s")).items[0].status is TodoStatus.PENDING
        assert all(
            "Todo 结束自查" not in message.text for message in store.load_session("s").messages
        )
        assert settlement.goal.rounds_started == 1
    finally:
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("report_again", [False, True])
async def test_file_work_after_reminder_requires_separate_new_report(
    tmp_path: Path, store: InMemoryLifecycleStore | SQLiteStore, report_again: bool
) -> None:
    """提醒后的真实读写使旧报告失效；文件工作后另一步报告才可完成。"""
    actions = ("report", "candidate", "read", "write")
    actions += ("report", "final") if report_again else ("final",)
    path = _todo_file(tmp_path)
    provider = _GoalProvider(store, path, actions)
    runner = _runner(tmp_path, provider)
    try:
        service = runner._goal_service
        goal = service.create(
            "s", "完成目标", run_options=AgentRunOptions(limits=RunLimits(max_model_steps=8))
        )
        result = await runner._start_goal_managed(goal.ref, run_id="goal-run")
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert len(provider.requests) == len(actions)
        assert ["Todo 结束自查" in _snapshot_text(request) for request in provider.requests] == [
            index == 2 for index in range(len(actions))
        ]
        records = runner.list_tool_calls("goal-run")
        assert [record.tool_name for record in records] == [
            "report_goal",
            "read_file",
            "write_file",
            *(["report_goal"] if report_again else []),
        ]
        assert all(not record.result.is_error for record in records)
        assert path.read_text(encoding="utf-8") == "- [x] 整理清单\n"
        assert (await runner.get_todo("s")).items[0].status is TodoStatus.COMPLETED
        settlement = service.settle_run("goal-run", now=runner._now())
        if report_again:
            assert records[-1].step_index > records[-2].step_index
            assert settlement.goal.status is GoalStatus.COMPLETED
            assert settlement.binding.applied_report_call_id == "report-4"
        else:
            assert settlement.goal.status is GoalStatus.ACTIVE
            assert settlement.goal.reason.code == "report_superseded"
            assert settlement.binding.applied_report_call_id is None
        assert result.run.usage.model_steps_committed == len(actions)
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_completed_todo_does_not_report_or_complete_goal(
    tmp_path: Path, store: InMemoryLifecycleStore | SQLiteStore
) -> None:
    """Todo 状态与 Goal 结算独立，完成清单不会伪造工具报告或再开 Run。"""
    provider = _GoalProvider(store, _todo_file(tmp_path, completed=True), ("final",))
    runner = _runner(tmp_path, provider)
    try:
        service = runner._goal_service
        goal = service.create("s", "目标需要独立申报")
        result = await runner._start_goal_managed(goal.ref, run_id="goal-run")
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert len(provider.requests) == 1
        assert runner.list_tool_calls("goal-run") == []
        settlement = service.settle_run("goal-run", now=runner._now())
        assert settlement.goal.status is GoalStatus.ACTIVE
        assert settlement.binding.applied_report_call_id is None
        assert settlement.goal.rounds_started == 1
    finally:
        await runner.aclose()
