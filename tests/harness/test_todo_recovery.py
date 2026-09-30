"""Todo 自查在真实 Run 恢复和人工交互中的持久化语义。"""

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisRunPersistenceError, IrisRunRecoveryError
from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse, QuestionInteractionResponse
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CheckpointResumability,
    CommitModelStep,
    FinishRun,
    RunCommit,
    RunEventKind,
    RunLimits,
    RunPhase,
    RunStopReason,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import BlockingProvider, StaticProvider, text_response, tool_response


def _config(workspace: Path) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "name": "todo-recovery",
            "model": "openai/test",
            "system": "按当前清单工作并诚实报告状态。",
            "permissions": {"workspace": str(workspace)},
            "todo": {"enabled": True},
            "tools": {"builtin": ["file.write", "human.ask"]},
        }
    )


async def _write_todo(runner: AgentRunner) -> Path:
    path = (await runner.get_todo("work")).path
    path.parent.mkdir(parents=True)
    path.write_text("- [ ] 仍等待外部反馈的任务\n", encoding="utf-8")
    return path


def _snapshot(request: LLMRequest) -> str:
    return "\n".join(
        message.text
        for message in request.messages
        if message.metadata.get("context_kind") == "runtime_snapshot"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("clear_file", [False, True])
async def test_cancelled_reminder_recovers_same_target_and_pending_reservation(
    tmp_path: Path, persistent: bool, clear_file: bool
) -> None:
    """目标 provider 尚未返回时取消 task，再从原 Run fence 恢复同一步。"""

    class BlockReminderProvider(StaticProvider):
        """初次返回候选回复，在自查请求真正到达时阻塞。"""

        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """为测试暴露精确的已预留、自查尚未提交边界。"""
            self.requests.append(request)
            if len(self.requests) == 1:
                return text_response("初次候选回复")
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("目标步骤应由测试取消")

    database = tmp_path / "recovery.db"
    store = SQLiteStore(database) if persistent else InMemoryLifecycleStore()
    provider = BlockReminderProvider()
    first = AgentRunner.from_config(_config(tmp_path), store=store, provider=provider)
    path = await _write_todo(first)
    running = asyncio.create_task(
        first.start(
            AgentRunRequest(input="完成工作", session_id="work", run_id="recover"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=2)),
        )
    )
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        crashed = store.load_run("recover")
        checkpoint = store.load_checkpoint("recover")
        assert crashed.phase is RunPhase.ACTIVE
        assert crashed.current_activation_id is not None
        assert checkpoint.engine_cursor["position"] == "before_model"
        assert checkpoint.engine_cursor["step_index"] == 1
        assert checkpoint.engine_cursor["todo_reminder_step"] == 1
        assert checkpoint.model_steps_reserved == 2 and checkpoint.model_steps_committed == 1
        assert "Todo 结束自查" not in _snapshot(provider.requests[0])
        assert _snapshot(provider.requests[1]).count("Todo 结束自查") == 1
        assert "仍等待外部反馈的任务" in _snapshot(provider.requests[1])
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await first.aclose()

    path.write_text("" if clear_file else "- [x] 人工已经完成的新工作\n", encoding="utf-8")
    reopened = SQLiteStore(database) if persistent else store
    recovery_provider = StaticProvider(text_response("根据当前状态结束"))
    second = AgentRunner.from_config(_config(tmp_path), store=reopened, provider=recovery_provider)
    try:
        result = await second.recover(
            "recover", expected_activation_id=crashed.current_activation_id
        )
        assert result.run.run_id == "recover"
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert result.run.current_activation_id != crashed.current_activation_id
        assert result.run.usage.model_steps_reserved == 2
        assert result.run.usage.model_steps_committed == 2
        assert len(recovery_provider.requests) == 1
        current = _snapshot(recovery_provider.requests[0])
        assert current.count("Todo 结束自查") == 1
        assert "仍等待外部反馈的任务" not in current
        if not clear_file:
            assert "- [x] 人工已经完成的新工作" in current
        assert reopened.load_checkpoint("recover").engine_cursor["todo_reminder_step"] == 1
        events = reopened.list_events("recover")
        assert sum(event.kind is RunEventKind.MODEL_STEP_RESERVED for event in events) == 2
        assert sum(event.kind is RunEventKind.ACTIVATION_ABANDONED for event in events) == 1
        assert all(
            "Todo 结束自查" not in message.model_dump_json()
            for message in reopened.load_session("work").messages
        )
    finally:
        await second.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_recover_scheduled_reminder_before_its_model_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    """候选回复和目标编号已提交、下一步尚未预留时，恢复正常预留一次。"""
    database = tmp_path / "before-reserve.db"
    store = SQLiteStore(database) if persistent else InMemoryLifecycleStore()
    first_provider = StaticProvider(text_response("候选回复"))
    first = AgentRunner.from_config(_config(tmp_path), store=store, provider=first_provider)
    path = await _write_todo(first)
    commit_model_step = store.commit_model_step

    def fail_after_scheduling(command: CommitModelStep) -> RunCommit:
        committed = commit_model_step(command)
        if command.checkpoint.engine_cursor["todo_reminder_step"] == 1:
            raise IrisRunPersistenceError("crash after scheduling reminder")
        return committed

    try:
        with monkeypatch.context() as patch:
            patch.setattr(store, "commit_model_step", fail_after_scheduling)
            with pytest.raises(IrisRunPersistenceError, match="crash after scheduling reminder"):
                await first.start(
                    AgentRunRequest(input="完成工作", session_id="work", run_id="scheduled"),
                    options=AgentRunOptions(limits=RunLimits(max_model_steps=2)),
                )
        crashed = store.load_run("scheduled")
        checkpoint = store.load_checkpoint("scheduled")
        assert crashed.phase is RunPhase.ACTIVE
        assert checkpoint.engine_cursor["position"] == "before_model"
        assert checkpoint.engine_cursor["todo_reminder_step"] == 1
        assert checkpoint.model_steps_reserved == checkpoint.model_steps_committed == 1
        assert len(first_provider.requests) == 1
    finally:
        await first.aclose()

    path.write_text("- [-] 恢复前人工更新的清单\n", encoding="utf-8")
    reopened = SQLiteStore(database) if persistent else store
    provider = StaticProvider(text_response("如实说明等待"))
    second = AgentRunner.from_config(_config(tmp_path), store=reopened, provider=provider)
    try:
        result = await second.recover(
            "scheduled", expected_activation_id=crashed.current_activation_id
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert result.run.usage.model_steps_reserved == 2
        assert result.run.usage.model_steps_committed == 2
        assert len(provider.requests) == 1
        assert "Todo 结束自查" in _snapshot(provider.requests[0])
        assert "恢复前人工更新的清单" in _snapshot(provider.requests[0])
        assert "仍等待外部反馈的任务" not in _snapshot(provider.requests[0])
    finally:
        await second.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("interaction", ["question", "permission"])
async def test_reminder_target_waits_then_typed_resume_does_not_repeat_hint(
    tmp_path: Path, interaction: str
) -> None:
    """目标步工具进入真实 WAITING，重建 Runner typed resume 后不再提示。"""
    call = (
        ToolUseBlock(id="ask", name="ask_question", input={"question": "继续等待吗？"})
        if interaction == "question"
        else ToolUseBlock(
            id="write", name="write_file", input={"file_path": "result.txt", "content": "已确认"}
        )
    )
    database = tmp_path / "waiting.db"
    first_provider = StaticProvider(text_response("候选回复"), tool_response(call))
    first = AgentRunner.from_config(
        _config(tmp_path), store=SQLiteStore(database), provider=first_provider
    )
    try:
        await _write_todo(first)
        waiting = await first.start(
            AgentRunRequest(input="处理清单", session_id="work", run_id="waiting"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=5)),
        )
        assert waiting.run.phase is RunPhase.WAITING
        assert waiting.pending_interaction is not None
        checkpoint = first.store.load_checkpoint("waiting")
        assert checkpoint.engine_cursor["todo_reminder_step"] == 1
        assert checkpoint.engine_cursor["step_index"] == 1
        assert "Todo 结束自查" in _snapshot(first_provider.requests[1])
    finally:
        await first.aclose()

    provider = StaticProvider(text_response("清单仍有等待事项，结束本轮"))
    second = AgentRunner.from_config(
        _config(tmp_path), store=SQLiteStore(database), provider=provider
    )
    try:
        response = (
            QuestionInteractionResponse(answer="继续等待")
            if interaction == "question"
            else PermissionInteractionResponse(decision="approve")
        )
        result = await second.resume(
            "waiting", interaction_id=waiting.pending_interaction.interaction_id, response=response
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert result.run.usage.model_steps_reserved == 3
        assert result.run.usage.model_steps_committed == 3
        assert len(provider.requests) == 1
        assert "Todo 结束自查" not in _snapshot(provider.requests[0])
        assert "仍等待外部反馈的任务" in _snapshot(provider.requests[0])
        assert second.store.load_checkpoint("waiting").engine_cursor["todo_reminder_step"] == 1
        [record] = second.list_tool_calls("waiting")
        assert record.result is not None and not record.result.is_error
        if interaction == "permission":
            assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "已确认"
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_v4_cursor_missing_reminder_field_is_rejected_only_by_runtime_recovery(
    tmp_path: Path,
) -> None:
    """Store 保留不透明 v4 payload，runtime 恢复入口拒绝缺失的必需字段。"""
    database = tmp_path / "missing-field.db"
    store = SQLiteStore(database)
    provider = BlockingProvider()
    first = AgentRunner.from_config(_config(tmp_path), store=store, provider=provider)
    running = asyncio.create_task(
        first.start(AgentRunRequest(input="工作", session_id="work", run_id="missing-field"))
    )
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        crashed = store.load_run("missing-field")
        checkpoint = store.load_checkpoint("missing-field")
        assert checkpoint.checkpoint_version == 4
        payload = dict(checkpoint.engine_cursor)
        payload.pop("todo_reminder_step")
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await first.aclose()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE run_checkpoints SET cursor_json = ? WHERE run_id = ?",
            (json.dumps(payload), "missing-field"),
        )
    reopened = SQLiteStore(database)
    assert reopened.load_checkpoint("missing-field").engine_cursor == payload
    recovery_provider = StaticProvider()
    second = AgentRunner.from_config(_config(tmp_path), store=reopened, provider=recovery_provider)
    try:
        with pytest.raises(IrisRunRecoveryError, match="cursor 无法恢复"):
            await second.recover(
                "missing-field", expected_activation_id=crashed.current_activation_id
            )
        assert recovery_provider.requests == []
        assert reopened.load_run("missing-field") == crashed
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_reminder_outcome_ready_recovery_only_finalizes(tmp_path: Path) -> None:
    """自查后的最终回复已提交时，recover 只终态化而不重新提醒。"""

    class FailFinishOnceStore(InMemoryLifecycleStore):
        """在最终结果已准备好后中断一次 finish。"""

        failed = False

        def finish_run(self, command: FinishRun) -> RunCommit:
            """保留 outcome_ready checkpoint，供另一个 runner 恢复。"""
            if not self.failed:
                self.failed = True
                raise IrisRunPersistenceError("finish after reminder failed")
            return super().finish_run(command)

    store = FailFinishOnceStore()
    provider = StaticProvider(text_response("候选回复"), text_response("仍在等待外部反馈"))
    first = AgentRunner.from_config(_config(tmp_path), store=store, provider=provider)
    try:
        await _write_todo(first)
        with pytest.raises(IrisRunPersistenceError, match="finish after reminder failed"):
            await first.start(
                AgentRunRequest(input="完成当前工作", session_id="work", run_id="finish"),
                options=AgentRunOptions(limits=RunLimits(max_model_steps=3)),
            )
        checkpoint = store.load_checkpoint("finish")
        assert checkpoint.resumability is CheckpointResumability.OUTCOME_READY
        assert checkpoint.engine_cursor["todo_reminder_step"] == 1
        crashed = store.load_run("finish")
        assert crashed is not None and crashed.current_activation_id is not None
        assert len(provider.requests) == 2
    finally:
        await first.aclose()

    recovery_provider = StaticProvider()
    second = AgentRunner.from_config(_config(tmp_path), store=store, provider=recovery_provider)
    try:
        result = await second.recover(
            "finish", expected_activation_id=crashed.current_activation_id
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert result.assistant_message.text == "仍在等待外部反馈"
        assert recovery_provider.requests == []
        assert result.run.usage.model_steps_reserved == 2
        assert result.run.usage.model_steps_committed == 2
    finally:
        await second.aclose()
