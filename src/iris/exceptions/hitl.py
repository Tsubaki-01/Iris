"""人工交互生命周期与恢复协议异常。"""

from .base import IrisError


class IrisHITLError(IrisError):
    """人工交互生命周期和恢复协议错误的基类。"""

    runtime_error_source = "runtime"
    runtime_error_code = "HITL_ERROR"


class HITLResponseMismatchError(IrisHITLError):
    runtime_error_code = "HITL_RESPONSE_MISMATCH"


class HITLConflictError(IrisHITLError):
    runtime_error_code = "HITL_CONFLICT"


class HITLCheckpointInvalidError(IrisHITLError):
    runtime_error_code = "HITL_CHECKPOINT_INVALID"
