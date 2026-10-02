"""Logical run 生命周期与持久化异常。"""

from .base import IrisError


class IrisRunError(IrisError):
    """Logical run 生命周期领域错误的基类。"""

    runtime_error_source = "lifecycle"
    runtime_error_code = "RUN_ERROR"


class IrisRunNotFoundError(IrisRunError):
    """目标 logical run 或其关联记录不存在。"""

    runtime_error_code = "RUN_NOT_FOUND"


class IrisRunConflictError(IrisRunError):
    """Lifecycle identity、lane 或 compare-and-set 发生冲突。"""

    runtime_error_code = "RUN_CONFLICT"


class IrisRunStateError(IrisRunError):
    """Lifecycle command 不适用于当前 run 状态。"""

    runtime_error_code = "RUN_STATE_ERROR"


class IrisRunPersistenceError(IrisRunError):
    """Lifecycle store 无法可靠读取或提交事务。"""

    runtime_error_source = "persistence"
    runtime_error_code = "RUN_PERSISTENCE_ERROR"


class IrisRunRecoveryError(IrisRunError):
    """Durable recovery facts 无法安全解释。"""

    runtime_error_code = "RUN_RECOVERY_ERROR"


class IrisRunObservationTimeoutError(IrisRunError):
    """等待 logical run 可观察状态变化时超时。"""

    runtime_error_code = "RUN_OBSERVATION_TIMEOUT"


class IrisLifecycleSchemaError(IrisRunPersistenceError):
    """Lifecycle 持久化 schema 与当前契约不兼容。"""

    runtime_error_code = "LIFECYCLE_SCHEMA_ERROR"
