"""Goal 与 logical run 共用 backend 的状态及原子准入契约。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.exceptions import IrisGoalConflictError, IrisGoalStateError, IrisRunConflictError
from iris.goal.models import GoalReason, GoalRef
from iris.goal.store import (
    AdmitGoalRun,
    ClearGoal,
    CompleteGoal,
    CreateGoal,
    EditGoal,
    GoalStore,
    PauseGoal,
    ResumeGoal,
)
from iris.lifecycle import AgentRunOptions, RunStopReason
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .test_lifecycle_store_contract import _create_command, _finish

NOW = datetime(2026, 9, 30, tzinfo=UTC)


@pytest.fixture(params=["memory", "sqlite"])
def goal_store(request: pytest.FixtureRequest, tmp_path: Path) -> GoalStore:
    """两种存储执行相同的领域与事务契约。"""
    return (
        SQLiteStore(tmp_path / "goal.db") if request.param == "sqlite" else InMemoryLifecycleStore()
    )


def create_command(goal_id: str = "goal-1", *, max_rounds: int = 3) -> CreateGoal:
    """构造完整且可重复的 Goal 创建命令。"""
    return CreateGoal(
        goal_id=goal_id,
        session_id="session-1",
        objective="修复计算并通过测试",
        max_rounds=max_rounds,
        run_options=AgentRunOptions(),
        now=NOW,
    )


def test_create_before_chat_keeps_empty_session_and_unique_current(goal_store: GoalStore) -> None:
    goal = goal_store.create_goal(create_command())
    session = goal_store.load_session(goal.session_id)
    assert session.revision == 0
    assert session.messages == []
    assert goal_store.load_session_lane(goal.session_id) is None
    assert goal_store.get_current_goal(goal.session_id) == goal
    with pytest.raises(IrisGoalStateError):
        goal_store.create_goal(create_command("goal-2"))
    assert goal_store.get_goal(goal.goal_id) == goal


def test_mutations_only_change_goal_revision_and_preserve_spent_rounds(
    goal_store: GoalStore,
) -> None:
    goal = goal_store.create_goal(create_command())
    admitted = goal_store.admit_goal_run(
        AdmitGoalRun(
            expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
            create_run=_create_command(),
        )
    )
    goal = admitted.goal
    assert goal.rounds_started == 1
    edited = goal_store.update_goal(
        EditGoal(
            expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
            objective="修复并增加用例",
            now=NOW,
        )
    )
    assert edited.status == "paused"
    assert edited.rounds_started == 1
    assert edited.revision == goal.revision + 1
    assert goal_store.load_session_revision(goal.session_id) == 0
    resumed = goal_store.update_goal(
        ResumeGoal(expected=GoalRef(goal_id=goal.goal_id, revision=edited.revision), now=NOW)
    )
    assert resumed.status == "active"
    assert resumed.rounds_started == 1
    assert (
        goal_store.update_goal(
            ResumeGoal(expected=GoalRef(goal_id=goal.goal_id, revision=resumed.revision), now=NOW)
        )
        == resumed
    )
    with pytest.raises(IrisGoalConflictError):
        goal_store.update_goal(
            PauseGoal(
                expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
                reason=GoalReason(code="user", text="暂停"),
                now=NOW,
            )
        )
    with pytest.raises(IrisGoalStateError):
        goal_store.update_goal(
            EditGoal(
                expected=GoalRef(goal_id=goal.goal_id, revision=resumed.revision),
                max_rounds=0,
                now=NOW,
            )
        )


def test_complete_replace_and_clear_keep_old_facts(goal_store: GoalStore) -> None:
    goal = goal_store.create_goal(create_command())
    completed = goal_store.update_goal(
        CompleteGoal(
            expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
            reason=GoalReason(code="user", text="完成"),
            now=NOW,
        )
    )
    with pytest.raises(IrisGoalStateError):
        goal_store.update_goal(
            ResumeGoal(expected=GoalRef(goal_id=goal.goal_id, revision=completed.revision), now=NOW)
        )
    second = goal_store.create_goal(create_command("goal-2"))
    assert goal_store.get_goal(goal.goal_id) == completed
    goal_store.update_goal(
        ClearGoal(
            session_id=second.session_id,
            expected=GoalRef(goal_id=second.goal_id, revision=second.revision),
            now=NOW,
        )
    )
    assert goal_store.get_current_goal(second.session_id) is None
    assert goal_store.get_goal(second.goal_id).revision == second.revision + 1
    assert (
        goal_store.update_goal(ClearGoal(session_id=second.session_id, expected=None, now=NOW))
        is None
    )


def test_admission_records_one_binding_and_rejects_stale_or_occupied(goal_store: GoalStore) -> None:
    goal = goal_store.create_goal(create_command())
    expected = GoalRef(goal_id=goal.goal_id, revision=goal.revision)
    first = goal_store.admit_goal_run(AdmitGoalRun(expected=expected, create_run=_create_command()))
    assert first.goal.revision == goal.revision + 1
    assert first.binding.admission_revision == goal.revision
    assert first.binding.round_no == first.goal.rounds_started == 1
    assert goal_store.get_goal_run("run-1") == first.binding
    with pytest.raises(IrisGoalConflictError):
        goal_store.admit_goal_run(
            AdmitGoalRun(
                expected=expected,
                create_run=_create_command(run_id="run-2", activation_id="activation-2"),
            )
        )
    assert goal_store.load_run("run-2") is None
    assert goal_store.get_goal(goal.goal_id) == first.goal
    _finish(goal_store, first.commit, RunStopReason.COMPLETED)
    with pytest.raises(IrisGoalStateError):
        goal_store.admit_goal_run(
            AdmitGoalRun(
                expected=GoalRef(goal_id=goal.goal_id, revision=first.goal.revision),
                create_run=_create_command(run_id="run-2", activation_id="activation-2"),
            )
        )
    assert goal_store.get_goal_run("run-2") is None


def test_failed_run_admission_does_not_change_goal(goal_store: GoalStore) -> None:
    goal = goal_store.create_goal(create_command())
    goal_store.create_run(_create_command())
    with pytest.raises(IrisRunConflictError):
        goal_store.admit_goal_run(
            AdmitGoalRun(
                expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
                create_run=_create_command(run_id="goal-run", activation_id="goal-activation"),
            )
        )
    assert goal_store.get_goal(goal.goal_id) == goal
    assert goal_store.get_goal_run("goal-run") is None
    assert goal_store.load_run("goal-run") is None


def test_concurrent_admission_has_one_winner(goal_store: GoalStore) -> None:
    goal = goal_store.create_goal(create_command())
    expected = GoalRef(goal_id=goal.goal_id, revision=goal.revision)

    def admit(index: int) -> bool:
        try:
            goal_store.admit_goal_run(
                AdmitGoalRun(
                    expected=expected,
                    create_run=_create_command(
                        run_id=f"run-{index}", activation_id=f"activation-{index}"
                    ),
                )
            )
            return True
        except IrisGoalConflictError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(admit, [1, 2])) == [False, True]
    assert goal_store.get_goal(goal.goal_id).rounds_started == 1


def test_last_inflight_run_can_resume_without_more_budget(goal_store: GoalStore) -> None:
    goal = goal_store.create_goal(create_command(max_rounds=1))
    admission = goal_store.admit_goal_run(
        AdmitGoalRun(
            expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
            create_run=_create_command(),
        )
    )
    paused = goal_store.update_goal(
        PauseGoal(
            expected=GoalRef(goal_id=goal.goal_id, revision=admission.goal.revision),
            reason=GoalReason(code="user", text="暂停"),
            now=NOW,
        )
    )
    assert (
        goal_store.update_goal(
            ResumeGoal(expected=GoalRef(goal_id=goal.goal_id, revision=paused.revision), now=NOW)
        ).status
        == "active"
    )
    goal_store.update_goal(
        ClearGoal(
            session_id=goal.session_id,
            expected=GoalRef(goal_id=goal.goal_id, revision=paused.revision + 1),
            now=NOW,
        )
    )
    assert goal_store.list_unsettled_goal_runs(goal.session_id) == (admission.binding,)


def test_memory_admission_build_failure_publishes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryLifecycleStore()
    goal = store.create_goal(create_command())

    def fail_create(command: object) -> None:
        raise RuntimeError("候选构造失败")

    monkeypatch.setattr(store, "_prepare_create_run", fail_create)
    with pytest.raises(RuntimeError, match="候选构造失败"):
        store.admit_goal_run(
            AdmitGoalRun(
                expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
                create_run=_create_command(),
            )
        )
    assert store.get_goal(goal.goal_id) == goal
    assert store.load_run("run-1") is None
    assert store.get_goal_run("run-1") is None
    assert store.load_session_lane(goal.session_id) is None


def test_admission_session_must_match_goal(goal_store: GoalStore) -> None:
    goal = goal_store.create_goal(create_command())
    with pytest.raises(IrisGoalStateError):
        goal_store.admit_goal_run(
            AdmitGoalRun(
                expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
                create_run=_create_command(session_id="elsewhere"),
            )
        )
    assert goal_store.get_goal(goal.goal_id).rounds_started == 0


def test_goal_creation_preserves_existing_history(goal_store: GoalStore) -> None:
    first = goal_store.create_run(_create_command())
    _finish(goal_store, first, RunStopReason.COMPLETED)
    before = goal_store.load_session("session-1")
    goal_store.create_goal(create_command())
    assert goal_store.load_session("session-1") == before


def test_goal_reopens_with_binding_and_count(tmp_path: Path) -> None:
    path = tmp_path / "durable.db"
    store = SQLiteStore(path)
    goal = store.create_goal(create_command())
    admitted = store.admit_goal_run(
        AdmitGoalRun(
            expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
            create_run=_create_command(),
        )
    )
    reopened = SQLiteStore(path)
    assert reopened.get_current_goal(goal.session_id) == admitted.goal
    assert reopened.get_goal_run(admitted.commit.run.run_id) == admitted.binding
