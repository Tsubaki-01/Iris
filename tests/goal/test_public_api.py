"""公开 Goal SDK 类型在领域包与 harness 入口保持同一契约。"""

import iris.goal as goal
import iris.harness as harness


def test_public_sdk_types_share_identity_between_entry_points() -> None:
    assert harness.GoalSession is goal.GoalSession
    assert harness.GoalView is goal.GoalView
    assert harness.GoalControlResult is goal.GoalControlResult
    assert harness.GoalChanged is goal.GoalChanged


def test_private_scheduling_and_admission_are_not_top_level_public_api() -> None:
    internal = {
        "GoalDriver",
        "GoalContinuationIntent",
        "GoalAdmission",
        "AdmitGoalRun",
        "GoalCreateInput",
        "GoalEditInput",
        "GoalControlPort",
    }
    assert internal.isdisjoint(goal.__all__)
