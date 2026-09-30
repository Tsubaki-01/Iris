"""GoalDriver 只管理进程内候选，不拥有异步执行。"""

import pytest

from iris.exceptions import IrisGoalStateError
from iris.goal.driver import GoalDriver
from iris.goal.service import GoalService
from iris.store import InMemoryLifecycleStore


def test_driver_starts_disarmed_and_requires_active_matching_goal_with_budget() -> None:
    service = GoalService(InMemoryLifecycleStore())
    goal = service.create("s", "完成目标", max_rounds=2)
    driver = GoalDriver()
    assert driver.armed_goal_id is None and driver.intent is None
    assert not driver.can_continue(goal)
    driver.arm(goal.goal_id)
    assert driver.can_continue(goal)
    assert not driver.can_continue(None)
    assert not driver.can_continue(goal.model_copy(update={"rounds_started": 2}))
    driver.arm("other")
    assert not driver.can_continue(goal)


def test_duplicate_terminal_keeps_one_intent_until_consumed() -> None:
    driver = GoalDriver()
    driver.arm("g")
    first = driver.offer("source", "g", "next")
    assert driver.offer("source", "g", "another-id") is first
    assert first.run_id == "next"
    assert driver.consume() is first
    assert driver.intent is None
    assert driver.armed_goal_id == "g"
    assert driver.consume() is None


def test_invalidation_returns_cleanup_identity_without_disarming() -> None:
    driver = GoalDriver()
    driver.arm("g")
    first = driver.offer(None, "g", "kickoff")
    assert driver.invalidate() is first
    assert driver.armed_goal_id == "g"
    second = driver.offer(None, "g", "retry")
    assert driver.disarm() is second
    assert driver.armed_goal_id is None and driver.intent is None
    assert driver.disarm() is None


def test_replacing_live_candidate_requires_explicit_cleanup() -> None:
    driver = GoalDriver()
    first = driver.offer("a", "g", "b")
    with pytest.raises(IrisGoalStateError):
        driver.offer("other", "new-goal", "c")
    assert driver.intent is first
