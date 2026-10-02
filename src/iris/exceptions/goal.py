"""Goal 状态、版本与持久化领域异常。"""

from .base import IrisError


class IrisGoalError(IrisError):
    """目标操作失败的领域基类。"""

    runtime_error_code = "GOAL_ERROR"


class IrisGoalStateError(IrisGoalError):
    """目标状态或操作参数不允许当前转换。"""

    runtime_error_code = "GOAL_STATE_ERROR"


class IrisGoalConflictError(IrisGoalError):
    """目标身份、版本或当前目标归属已发生变化。"""

    runtime_error_code = "GOAL_CONFLICT"


class IrisGoalNotFoundError(IrisGoalError):
    """指定目标记录不存在。"""

    runtime_error_code = "GOAL_NOT_FOUND"


class IrisGoalPersistenceError(IrisGoalError):
    """目标存储无法可靠读取或提交事实。"""

    runtime_error_source = "persistence"
    runtime_error_code = "GOAL_PERSISTENCE_ERROR"


__all__ = [
    "IrisGoalError",
    "IrisGoalStateError",
    "IrisGoalConflictError",
    "IrisGoalNotFoundError",
    "IrisGoalPersistenceError",
]
