"""真实文件工具、SQLite 和 Manager 的两轮 Goal 核心集成验收。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.goal import GoalChanged, GoalSnapshot
from iris.harness import AgentRunner, SessionManager
from iris.harness.streaming import LiveFact
from iris.lifecycle import RunEvent, RunEventKind, RunPhase, RunStopReason, ToolCallPhase
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import SQLiteStore

from .fakes import StaticProvider, text_response, tool_response

SESSION = "file-goal"
OBJECTIVE = "把 [7, 11, 13] 保存到 numbers.json，下一轮读取求和并保存 result.json，逐次读取验证。"


def file_goal_config(workspace: Path) -> AgentConfig:
    """文件读写直接使用内置工具和真实 workspace，写入无需人工确认。"""
    return AgentConfig.model_validate(
        {
            "name": "file-goal",
            "model": "deepseek/deepseek-chat",
            "system": "按目标与验收要求操作文件。看到实际工具结果后单独报告 Goal 结果。",
            "tools": {"builtin": ["file.read", "file.write"]},
            "permissions": {"workspace": str(workspace), "writes": "allow"},
            "goal": {"enabled": True, "max_rounds": 2},
        }
    )


class GoalFacts:
    """记录实际发布事实，并为测试提供已结算和完成的同步点。"""

    def __init__(self) -> None:
        self.facts: list[LiveFact] = []
        self.first_settled = asyncio.Event()
        self.completed = asyncio.Event()

    def publish(self, fact: LiveFact) -> None:
        """只观察，不补结算或触发下一轮。"""
        self.facts.append(fact)
        if isinstance(fact, GoalChanged) and fact.view.goal is not None:
            view = fact.view
            if view.goal.rounds_started == 1 and view.run is None and not view.settlement_pending:
                self.first_settled.set()
            if view.goal.status == "completed":
                self.completed.set()


async def collect_completed_goal(manager: SessionManager, *, timeout: float = 10) -> list[object]:
    """只消费宿主事件，等待真实 GoalChanged 完成并及时暴露停止事实。"""
    observed: list[object] = []
    async with asyncio.timeout(timeout):
        async for event in manager.events():
            observed.append(event)
            if isinstance(event, GoalChanged):
                view = event.view
                if view.driver_error is not None:
                    pytest.fail(f"Goal driver failed: {view.driver_error.code}")
                if view.goal is not None and view.goal.status in {"paused", "blocked"}:
                    reason = None if view.goal.reason is None else view.goal.reason.code
                    pytest.fail(f"Goal stopped: status={view.goal.status}, reason={reason}")
                if view.goal is not None and view.goal.status == "completed":
                    return observed
    raise AssertionError("未收到目标完成通知")


def assert_file_goal_result(
    workspace: Path,
    store: SQLiteStore,
    goal_id: str,
    *,
    require_tokens: bool = False,
) -> tuple[GoalSnapshot, tuple[str, ...]]:
    """联合检验实际文件、已提交工具、两轮绑定、持久结果和重开事实。"""
    assert json.loads((workspace / "numbers.json").read_text(encoding="utf-8")) == [7, 11, 13]
    assert json.loads((workspace / "result.json").read_text(encoding="utf-8")) == {"sum": 31}
    goal = store.get_goal(goal_id)
    assert goal.status == "completed" and goal.rounds_started == 2
    assert store.list_unsettled_goal_runs(SESSION) == ()
    points = store.list_fork_points(SESSION).items
    assert len(points) == 2
    bindings = sorted(
        (store.get_goal_run(point.run_id) for point in points), key=lambda item: item.round_no
    )
    assert [binding.round_no for binding in bindings] == [1, 2]
    for binding in bindings:
        assert binding.goal_id == goal_id and binding.settled_at is not None
        result = store.load_result(binding.run_id)
        assert result.run.phase is RunPhase.TERMINAL
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert result.run.usage.model_steps_committed > 0
        if require_tokens:
            assert result.run.usage.input_tokens > 0
            assert result.run.usage.output_tokens > 0
            assert result.run.usage.total_tokens > 0
        calls = store.list_tool_calls(binding.run_id)
        assert all(call.phase is ToolCallPhase.COMMITTED for call in calls)
        file_calls = [call for call in calls if call.tool_name in {"read_file", "write_file"}]
        assert {call.tool_name for call in file_calls} == {"read_file", "write_file"}
        assert all(not call.result.is_error for call in file_calls)
        reports = [
            call for call in calls if call.tool_name == "report_goal" and not call.result.is_error
        ]
        assert reports
        assert reports[-1].result.data["goal_report"]["decision"] == (
            "continue" if binding.round_no == 1 else "complete"
        )
        required_file = "numbers.json" if binding.round_no == 1 else "result.json"
        assert any(
            call.tool_name == "write_file" and call.arguments["file_path"] == required_file
            for call in file_calls
        )
        assert any(
            call.tool_name == "read_file" and call.arguments["file_path"] == "numbers.json"
            for call in file_calls
        )
        if binding.round_no == 2:
            assert any(
                call.tool_name == "read_file" and call.arguments["file_path"] == "result.json"
                for call in file_calls
            )
            assert binding.applied_report_call_id == reports[-1].tool_call_id
    reopened = SQLiteStore(store.path)
    assert reopened.get_current_goal(SESSION) == goal
    assert reopened.load_session(SESSION) == store.load_session(SESSION)
    for binding in bindings:
        assert reopened.get_goal_run(binding.run_id) == binding
        assert reopened.load_result(binding.run_id) == store.load_result(binding.run_id)
    return goal, tuple(binding.run_id for binding in bindings)


class _FileProvider(StaticProvider):
    """只编排模型响应，实际文件操作和报告提交全部经过现有内核。"""

    def __init__(self, store: SQLiteStore) -> None:
        super().__init__()
        self.store = store
        self.steps: dict[int, int] = {}

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        goal = self.store.get_current_goal(SESSION)
        round_no = goal.rounds_started
        step = self.steps.get(round_no, 0)
        self.steps[round_no] = step + 1
        work = (
            [
                ("write_file", {"file_path": "numbers.json", "content": "[7, 11, 13]"}),
                ("read_file", {"file_path": "numbers.json"}),
            ]
            if round_no == 1
            else [
                ("read_file", {"file_path": "numbers.json"}),
                ("write_file", {"file_path": "result.json", "content": '{"sum": 31}'}),
                ("read_file", {"file_path": "result.json"}),
            ]
        )
        if step < len(work):
            name, arguments = work[step]
            return tool_response(
                ToolUseBlock(id=f"round-{round_no}-{step}", name=name, input=arguments)
            )
        if step == len(work):
            return tool_response(
                ToolUseBlock(
                    id=f"report-{round_no}",
                    name="report_goal",
                    input={
                        "goal_id": goal.goal_id,
                        "revision": goal.revision,
                        "decision": "continue" if round_no == 1 else "complete",
                        "reason": "已读取并核对本轮文件",
                    },
                )
            )
        return text_response("本轮结束")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["mixed_slow", "broker_only"])
async def test_two_round_file_goal_with_bounded_observation_and_durable_reopen(
    tmp_path: Path,
    mode: str,
) -> None:
    """慢消费者释放唯一 tracker 后续跑；broker-only 不消费本地 events 也完成。"""
    store = SQLiteStore(tmp_path / "goal.db")
    provider, publisher = _FileProvider(store), GoalFacts()
    runner = AgentRunner.from_config(file_goal_config(tmp_path), provider=provider, store=store)
    manager = SessionManager(
        runner,
        SESSION,
        observation_mode="mixed" if mode == "mixed_slow" else "broker_only",
        submission_publisher=publisher,
        max_tracked_durable_runs=1,
    )
    try:
        created = await manager.goal.create(OBJECTIVE)
        if mode == "mixed_slow":
            await asyncio.wait_for(publisher.first_settled.wait(), 5)
            assert store.get_goal(created.view.goal.goal_id).rounds_started == 1
            assert manager._event_buffer.tracked_run_count == 1
            observed = await collect_completed_goal(manager)
            completion = next(
                index
                for index, event in enumerate(observed)
                if isinstance(event, GoalChanged) and event.view.goal.status == "completed"
            )
            terminal_positions = [
                index
                for index, event in enumerate(observed)
                if isinstance(event, RunEvent) and event.kind is RunEventKind.RUN_TERMINAL
            ]
            assert len(terminal_positions) == 2 and max(terminal_positions) < completion
            assert manager._event_buffer.tracked_run_count <= 1
        else:
            await asyncio.wait_for(publisher.completed.wait(), 5)
            assert manager._event_buffer is None
        goal, run_ids = assert_file_goal_result(tmp_path, store, created.view.goal.goal_id)
        assert not (await manager.goal.get()).armed
        assert len(provider.requests) == 9
        assert [call.tool_name for call in store.list_tool_calls(run_ids[0])] == [
            "write_file",
            "read_file",
            "report_goal",
        ]
        assert [call.tool_name for call in store.list_tool_calls(run_ids[1])] == [
            "read_file",
            "write_file",
            "read_file",
            "report_goal",
        ]
        assert any(
            isinstance(fact, GoalChanged) and fact.view.goal == goal for fact in publisher.facts
        )
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()
