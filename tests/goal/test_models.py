"""Goal 外部数据与持久快照的解析契约。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from iris.goal import GoalConfig, GoalReason, GoalReport, GoalRunBinding, GoalSnapshot
from iris.lifecycle import AgentRunOptions


def _snapshot(**changes: object) -> GoalSnapshot:
    values: dict[str, object] = {
        "goal_id": "goal-1",
        "session_id": "session-1",
        "revision": 0,
        "objective": "修复问题并通过测试",
        "status": "active",
        "max_rounds": 2,
        "rounds_started": 0,
        "run_options": AgentRunOptions(),
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(changes)
    return GoalSnapshot.model_validate(values)


def test_configuration_defaults_and_positive_rounds() -> None:
    assert GoalConfig().enabled is False
    assert GoalConfig().max_rounds == 20
    with pytest.raises(ValidationError):
        GoalConfig(max_rounds=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"objective": " \n "},
        {"revision": -1},
        {"rounds_started": 3},
        {"max_rounds": 0},
        {"created_at": datetime(2026, 1, 1)},
        {"armed": True},
    ],
)
def test_snapshot_rejects_invalid_persisted_data(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _snapshot(**changes)


def test_snapshot_roundtrip_preserves_objective_and_run_options() -> None:
    goal = _snapshot(objective="  第一行\n第二行  ")
    restored = GoalSnapshot.model_validate_json(goal.model_dump_json())
    assert restored == goal
    assert restored.objective == "  第一行\n第二行  "
    with pytest.raises(ValidationError):
        goal.objective = "new"


def test_reason_and_report_require_nonblank_text() -> None:
    with pytest.raises(ValidationError):
        GoalReason(code="user", text=" ")
    with pytest.raises(ValidationError):
        GoalReport(goal_id="g", revision=1, decision="complete", reason="")
    with pytest.raises(ValidationError):
        GoalReport(goal_id="g", revision=1, decision="paused", reason="wait")
    with pytest.raises(ValidationError):
        GoalReport.model_validate(
            {
                "goal_id": "g",
                "revision": 1,
                "decision": "complete",
                "reason": "done",
                "run_id": "model-cannot-select-run",
            }
        )


def test_binding_requires_positive_round_and_consistent_settlement() -> None:
    binding = GoalRunBinding(run_id="r", goal_id="g", round_no=1, admission_revision=0)
    assert binding.settled_at is None
    with pytest.raises(ValidationError):
        GoalRunBinding(run_id="r", goal_id="g", round_no=0, admission_revision=0)
    with pytest.raises(ValidationError):
        GoalRunBinding(
            run_id="r",
            goal_id="g",
            round_no=1,
            admission_revision=0,
            applied_report_call_id="call-1",
        )


def test_service_validates_raw_fields_before_store_mutation() -> None:
    from iris.exceptions import IrisGoalStateError
    from iris.goal import GoalService
    from iris.store import InMemoryLifecycleStore

    store = InMemoryLifecycleStore()
    service = GoalService(store)
    with pytest.raises(IrisGoalStateError):
        service.create("session-1", " \n ")
    assert store.get_current_goal("session-1") is None
    goal = service.create("session-1", "完成任务", max_rounds=3)
    assert goal.max_rounds == 3
    with pytest.raises(IrisGoalStateError):
        service.edit(goal.ref, objective=" ")
    assert store.get_goal(goal.goal_id) == goal
