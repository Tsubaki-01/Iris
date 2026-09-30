"""目标申报窗口和跨 backend 原子结算的行为契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.exceptions import IrisGoalStateError
from iris.goal.models import GoalReason, GoalReport, GoalSettlement, GoalStatus
from iris.goal.store import (
    AdmitGoalRun,
    ClearGoal,
    CompleteGoal,
    CreateGoal,
    GoalStore,
    PauseGoal,
    ResumeGoal,
)
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import (
    AgentRunOptions,
    ClaimToolCall,
    CommitModelStep,
    CommitToolResult,
    FinishRun,
    ReserveModelStep,
    ResolveInteraction,
    ResumeWaitingRun,
    RunCommit,
    RunErrorInfo,
    RunStopReason,
    RunToolCallRecord,
    RunUsage,
    SuspendRun,
)
from iris.message import Msg, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolResult
from tests.store.test_lifecycle_store_contract import (
    _NOW,
    _checkpoint,
    _complete_history_turn,
    _create_command,
    _interaction,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> GoalStore:
    """让相同结算场景经过两个真实存储实现。"""
    return (
        SQLiteStore(tmp_path / "goal.db") if request.param == "sqlite" else InMemoryLifecycleStore()
    )


def _admit(store: GoalStore, *, max_rounds: int = 3) -> RunCommit:
    goal = store.create_goal(
        CreateGoal(
            goal_id="goal",
            session_id="session-1",
            objective="修复并验证",
            max_rounds=max_rounds,
            run_options=AgentRunOptions(),
            now=_NOW,
        )
    )
    return store.admit_goal_run(
        AdmitGoalRun(expected=goal.ref, create_run=_create_command())
    ).commit


def _report(
    store: GoalStore, decision: str = "complete", *, revision: int | None = None
) -> dict[str, object]:
    goal = store.get_goal("goal")
    return GoalReport(
        goal_id=goal.goal_id,
        revision=goal.revision if revision is None else revision,
        decision=decision,
        reason="已核对实际结果",
    ).model_dump(mode="json")


def _step(
    store: GoalStore,
    current: RunCommit,
    calls: list[tuple[str, dict[str, object], bool]],
    *,
    steer: bool = False,
    commit_results: bool = True,
) -> RunCommit:
    """通过真实模型/工具提交路径保存一个 batch，可在结果前交付 steer。"""
    run = current.run
    step = run.usage.model_steps_committed
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id=run.run_id,
            expected_run_revision=run.revision,
            activation_id=run.current_activation_id,
            now=_NOW,
        )
    ).commit
    checkpoint = store.load_checkpoint(run.run_id)
    session_revision = store.load_session_revision(run.session_id)
    uses = [
        ToolUseBlock(id=f"call-{step}-{ordinal}", name=name, input={})
        for ordinal, (name, _, _) in enumerate(calls, 1)
    ]
    prepared = [
        RunToolCallRecord(
            run_id=run.run_id,
            step_index=step,
            ordinal=ordinal,
            tool_call_id=use.id,
            tool_name=use.name,
            arguments={},
            fingerprint="a" * 64,
            phase="prepared",
            version=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
        for ordinal, use in enumerate(uses, 1)
    ]
    assistant = Msg.assistant(uses)
    current = store.commit_model_step(
        CommitModelStep(
            run_id=run.run_id,
            expected_run_revision=reserved.run.revision,
            activation_id=run.current_activation_id,
            expected_session_revision=session_revision,
            message_delta=[assistant],
            usage=RunUsage(
                model_steps_reserved=step + 1,
                model_steps_committed=step + 1,
                tool_calls_committed=run.usage.tool_calls_committed,
            ),
            prepared_tool_calls=prepared,
            assistant_message=assistant,
            checkpoint=_checkpoint(
                run_id=run.run_id,
                sequence=checkpoint.sequence + 1,
                activation_id=run.current_activation_id,
                session_revision=session_revision + 1,
                reserved=step + 1,
                committed=step + 1,
            ),
            now=_NOW,
        )
    )
    if not commit_results:
        return current
    for call, (_, data, is_error) in zip(prepared, calls, strict=True):
        current = _commit_call(store, current, call, data, is_error=is_error, steer=steer)
        steer = False
    return current


def _commit_call(
    store: GoalStore,
    current: RunCommit,
    call: RunToolCallRecord,
    data: dict[str, object],
    *,
    is_error: bool = False,
    steer: bool = False,
) -> RunCommit:
    run = current.run
    current = store.claim_tool_call(
        ClaimToolCall(
            run_id=run.run_id,
            expected_run_revision=run.revision,
            activation_id=run.current_activation_id,
            tool_call_id=call.tool_call_id,
            fingerprint=call.fingerprint,
            expected_tool_version=call.version,
            now=_NOW,
        )
    )
    result = ToolResult(
        tool_use_id=call.tool_call_id, tool_name=call.tool_name, data=data, is_error=is_error
    )
    checkpoint = store.load_checkpoint(run.run_id)
    session_revision = store.load_session_revision(run.session_id)
    messages = ([Msg.user("补充一个要求")] if steer else []) + [result.to_msg()]
    return store.commit_tool_result(
        CommitToolResult(
            run_id=run.run_id,
            expected_run_revision=current.run.revision,
            activation_id=run.current_activation_id,
            expected_session_revision=session_revision,
            tool_call_id=call.tool_call_id,
            expected_tool_version=call.version + 1,
            result=result,
            message_delta=messages,
            checkpoint=_checkpoint(
                run_id=run.run_id,
                sequence=checkpoint.sequence + 1,
                activation_id=run.current_activation_id,
                session_revision=session_revision + 1,
                reserved=run.usage.model_steps_reserved,
                committed=run.usage.model_steps_committed,
            ),
            now=_NOW,
        )
    )


def _finish(
    store: GoalStore, current: RunCommit, reason: RunStopReason = RunStopReason.COMPLETED
) -> None:
    store.finish_run(
        FinishRun(
            run_id=current.run.run_id,
            expected_run_revision=current.run.revision,
            activation_id=current.run.current_activation_id,
            stop_reason=reason,
            now=_NOW,
            error=RunErrorInfo(code="TEST_FAILURE", message="执行未完成", source="runtime")
            if reason in {RunStopReason.FAILED, RunStopReason.OUTCOME_UNKNOWN}
            else None,
        )
    )


@pytest.mark.parametrize("decision", ["complete", "blocked"])
def test_valid_report_wins_over_final_round_limit(store: GoalStore, decision: str) -> None:
    current = _admit(store, max_rounds=1)
    current = _step(
        store, current, [("report_goal", {"goal_report": _report(store, decision)}, False)]
    )
    _finish(store, current)
    result = store.settle_goal_run("run-1", now=_NOW)
    assert result.goal.status.value == {"complete": "completed", "blocked": "blocked"}[decision]
    assert result.goal_changed
    assert result.binding.applied_report_call_id == "call-0-1"
    assert result.goal.rounds_started == 1
    assert store.list_unsettled_goal_runs("session-1") == ()
    again = store.settle_goal_run("run-1", now=_NOW)
    assert again.goal == result.goal
    assert again.binding == result.binding
    assert not again.goal_changed


@pytest.mark.parametrize("last", ["continue", "old_revision"])
def test_last_report_replaces_earlier_complete(store: GoalStore, last: str) -> None:
    current = _admit(store)
    report = _report(store)
    current = _step(store, current, [("report_goal", {"goal_report": report}, False)])
    last_report = _report(store, "continue") if last == "continue" else _report(store, revision=0)
    current = _step(store, current, [("report_goal", {"goal_report": last_report}, False)])
    _finish(store, current)
    result = store.settle_goal_run("run-1", now=_NOW)
    assert result.goal.status is GoalStatus.ACTIVE
    assert result.binding.applied_report_call_id is None
    if last == "old_revision":
        assert result.goal.reason.code == "report_superseded"


@pytest.mark.parametrize(
    "scenario", ["same_batch", "later_error", "later_prepared", "steer_before_result"]
)
def test_new_work_or_delivered_steer_invalidates_report(store: GoalStore, scenario: str) -> None:
    current = _admit(store)
    calls = [("report_goal", {"goal_report": _report(store)}, False)]
    if scenario == "same_batch":
        calls.append(("work", {}, False))
    current = _step(store, current, calls, steer=scenario == "steer_before_result")
    if scenario == "later_error":
        current = _step(store, current, [("work", {}, True)])
    elif scenario == "later_prepared":
        current = _step(store, current, [("work", {}, False)], commit_results=False)
    _finish(store, current)
    result = store.settle_goal_run("run-1", now=_NOW)
    assert result.goal.status is GoalStatus.ACTIVE
    assert result.goal.reason.code == "report_superseded"


def test_goal_only_batch_uses_last_report_and_allows_later_read(store: GoalStore) -> None:
    """同批多个 Goal 调用按 ordinal 选最后申报，后续只读不使报告过时。"""
    current = _admit(store)
    current = _step(
        store,
        current,
        [
            ("report_goal", {"goal_report": _report(store, "blocked")}, False),
            ("get_goal", {}, False),
            ("report_goal", {"goal_report": _report(store)}, False),
        ],
    )
    current = _step(store, current, [("get_goal", {}, False)])
    _finish(store, current)
    result = store.settle_goal_run("run-1", now=_NOW)
    assert result.goal.status is GoalStatus.COMPLETED
    assert result.binding.applied_report_call_id == "call-0-3"


@pytest.mark.parametrize(
    "reason", [reason for reason in RunStopReason if reason is not RunStopReason.COMPLETED]
)
def test_abnormal_run_pauses_even_with_complete_report(
    store: GoalStore, reason: RunStopReason
) -> None:
    current = _admit(store)
    current = _step(store, current, [("report_goal", {"goal_report": _report(store)}, False)])
    _finish(store, current, reason)
    result = store.settle_goal_run("run-1", now=_NOW)
    assert result.goal.status is GoalStatus.PAUSED
    assert result.goal.reason.code == reason.value
    assert result.binding.applied_report_call_id is None


@pytest.mark.parametrize("kind", ["prepared", "error", "invalid_payload"])
def test_non_successful_reports_are_not_completion(store: GoalStore, kind: str) -> None:
    current = _admit(store, max_rounds=1)
    data = {"goal_report": _report(store)} if kind != "invalid_payload" else {"goal_report": {}}
    current = _step(
        store, current, [("report_goal", data, kind == "error")], commit_results=kind != "prepared"
    )
    _finish(store, current)
    result = store.settle_goal_run("run-1", now=_NOW)
    assert result.goal.status is GoalStatus.PAUSED
    assert result.goal.reason.code == "round_limit"


@pytest.mark.parametrize("change", ["pause", "complete", "clear", "replace"])
def test_user_control_is_not_overwritten(store: GoalStore, change: str) -> None:
    current = _admit(store)
    current = _step(store, current, [("report_goal", {"goal_report": _report(store)}, False)])
    goal = store.get_goal("goal")
    if change == "pause":
        store.update_goal(
            PauseGoal(expected=goal.ref, reason=GoalReason(code="user", text="暂停"), now=_NOW)
        )
    elif change == "complete":
        store.update_goal(
            CompleteGoal(expected=goal.ref, reason=GoalReason(code="user", text="已完成"), now=_NOW)
        )
    else:
        store.update_goal(ClearGoal(session_id=goal.session_id, expected=goal.ref, now=_NOW))
        if change == "replace":
            store.create_goal(
                CreateGoal(
                    goal_id="replacement",
                    session_id=goal.session_id,
                    objective="新目标",
                    max_rounds=3,
                    run_options=AgentRunOptions(),
                    now=_NOW,
                )
            )
    before = store.get_current_goal(goal.session_id)
    _finish(store, current)
    result = store.settle_goal_run("run-1", now=_NOW)
    assert result.goal == before
    assert not result.goal_changed
    assert result.binding.settled_at is not None


def test_later_run_messages_do_not_pollute_old_report(store: GoalStore) -> None:
    current = _admit(store)
    current = _step(store, current, [("report_goal", {"goal_report": _report(store)}, False)])
    _finish(store, current)
    _complete_history_turn(store, run_id="later", session_id="session-1")
    if isinstance(store, SQLiteStore):
        store = SQLiteStore(store.path)
    assert store.settle_goal_run("run-1", now=_NOW).goal.status is GoalStatus.COMPLETED


def test_active_run_cannot_be_settled(store: GoalStore) -> None:
    _admit(store)
    with pytest.raises(IrisGoalStateError):
        store.settle_goal_run("run-1", now=_NOW)


def test_memory_settlement_build_failure_publishes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """回执候选构造失败时，进程内目标和绑定都不提前发布。"""
    from iris.store import in_memory

    store = InMemoryLifecycleStore()
    current = _admit(store)
    current = _step(store, current, [("report_goal", {"goal_report": _report(store)}, False)])
    _finish(store, current)
    goal, binding = store.get_goal("goal"), store.get_goal_run("run-1")
    copy = in_memory.deepcopy

    def reject_settlement(value: object) -> object:
        if isinstance(value, GoalSettlement):
            raise RuntimeError("injected return projection failure")
        return copy(value)

    monkeypatch.setattr(in_memory, "deepcopy", reject_settlement)
    with pytest.raises(RuntimeError, match="projection failure"):
        store.settle_goal_run("run-1", now=_NOW)
    assert store.get_goal("goal") == goal
    assert store.get_goal_run("run-1") == binding


def test_new_version_report_after_pause_resume_can_complete(store: GoalStore) -> None:
    current = _admit(store)
    current = _step(store, current, [("report_goal", {"goal_report": _report(store)}, False)])
    paused = store.update_goal(
        PauseGoal(
            expected=store.get_goal("goal").ref,
            reason=GoalReason(code="user", text="暂停"),
            now=_NOW,
        )
    )
    store.update_goal(ResumeGoal(expected=paused.ref, now=_NOW))
    current = _step(store, current, [("report_goal", {"goal_report": _report(store)}, False)])
    _finish(store, current)
    assert store.settle_goal_run("run-1", now=_NOW).goal.status is GoalStatus.COMPLETED


@pytest.mark.parametrize("report_again", [False, True])
def test_resolved_interaction_requires_new_report(store: GoalStore, report_again: bool) -> None:
    """真实 HITL 回答没有 user 消息，旧报告仍失效，回答后重新申报才生效。"""
    current = _admit(store)
    payload = {"goal_report": _report(store)}
    current = _step(store, current, [("report_goal", payload, False)], commit_results=False)
    prepared = store.load_tool_call("run-1", "call-0-1")
    interaction = _interaction()
    interaction = interaction.model_copy(
        update={
            "tool_call_id": prepared.tool_call_id,
            "request": interaction.request.model_copy(
                update={
                    "tool_call": interaction.request.tool_call.model_copy(
                        update={
                            "tool_call_id": prepared.tool_call_id,
                            "tool_name": "report_goal",
                            "arguments": {},
                        }
                    ),
                }
            ),
        }
    )
    current = store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=current.run.revision,
            activation_id=current.run.current_activation_id,
            expected_session_revision=current.checkpoint.session_revision,
            checkpoint=current.checkpoint.model_copy(
                update={"sequence": current.checkpoint.sequence + 1}
            ),
            pending_interaction=interaction,
            usage=current.run.usage,
            now=_NOW,
        )
    )
    with pytest.raises(IrisGoalStateError):
        store.settle_goal_run("run-1", now=_NOW)
    current = store.resolve_interaction(
        ResolveInteraction(
            run_id="run-1",
            expected_run_revision=current.run.revision,
            interaction_id=interaction.interaction_id,
            expected_interaction_version=interaction.version,
            response=QuestionInteractionResponse(answer="继续"),
            now=_NOW,
        )
    )
    current = store.resume_waiting_run(
        ResumeWaitingRun(
            run_id="run-1",
            expected_run_revision=current.run.revision,
            new_activation_id="activation-resume",
            kind="resume",
            expected_checkpoint_sequence=current.checkpoint.sequence,
            now=_NOW,
        )
    )
    current = _commit_call(store, current, prepared, payload)
    if report_again:
        current = _step(store, current, [("report_goal", payload, False)])
    _finish(store, current)
    result = store.settle_goal_run("run-1", now=_NOW)
    if report_again:
        assert result.goal.status is GoalStatus.COMPLETED
    else:
        assert result.goal.status is GoalStatus.ACTIVE
        assert result.goal.reason.code == "report_superseded"
