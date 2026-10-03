"""公开 Iris 当前使用的领域异常。"""

from .base import IrisError, IrisValidationError
from .command import IrisCommandCleanupError, IrisCommandError
from .config import IrisConfigError
from .context import IrisContextCompactionError, IrisContextError
from .decision import IrisDecisionError
from .goal import (
    IrisGoalConflictError,
    IrisGoalError,
    IrisGoalNotFoundError,
    IrisGoalPersistenceError,
    IrisGoalStateError,
)
from .hitl import (
    HITLCheckpointInvalidError,
    HITLConflictError,
    HITLResponseMismatchError,
    IrisHITLError,
)
from .hooks import IrisHookError, IrisHookProtocolError
from .image import IrisImageError
from .lifecycle import (
    IrisLifecycleSchemaError,
    IrisRunConflictError,
    IrisRunError,
    IrisRunNotFoundError,
    IrisRunObservationTimeoutError,
    IrisRunPersistenceError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from .mcp import IrisMCPCallError, IrisMCPError, IrisMCPToolError
from .memory import IrisMemoryError
from .provider import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisProviderStreamError,
    IrisProviderStreamInterruptedError,
    IrisProviderStreamProtocolError,
    IrisRateLimitExceededError,
)
from .runtime import IrisCancellationRequestedError
from .sandbox import IrisSandboxError
from .skill import (
    IrisSkillError,
    IrisSkillFormatError,
    IrisSkillNotFoundError,
    IrisSkillPathError,
)
from .template import IrisTemplateError, IrisTemplateNotFoundError
from .todo import IrisTodoError
from .tools import (
    IrisToolError,
    IrisToolExecutionError,
    IrisToolNotFoundError,
    IrisToolOutcomeUnknownError,
    IrisToolValidationError,
)

__all__ = [
    "IrisDecisionError",
    "IrisImageError",
    "IrisHookError",
    "IrisHookProtocolError",
    "IrisTodoError",
    "IrisGoalError",
    "IrisGoalStateError",
    "IrisGoalConflictError",
    "IrisGoalNotFoundError",
    "IrisGoalPersistenceError",
    "IrisError",
    "IrisCancellationRequestedError",
    "IrisConfigError",
    "IrisContextError",
    "IrisContextCompactionError",
    "IrisSkillError",
    "IrisSkillFormatError",
    "IrisSkillPathError",
    "IrisSkillNotFoundError",
    "IrisValidationError",
    "IrisHITLError",
    "HITLResponseMismatchError",
    "HITLConflictError",
    "HITLCheckpointInvalidError",
    "IrisProviderError",
    "IrisProviderStreamError",
    "IrisProviderStreamInterruptedError",
    "IrisProviderStreamProtocolError",
    "IrisAPIConnectionError",
    "IrisRateLimitExceededError",
    "IrisAuthenticationError",
    "IrisToolError",
    "IrisToolNotFoundError",
    "IrisToolExecutionError",
    "IrisToolValidationError",
    "IrisToolOutcomeUnknownError",
    "IrisCommandError",
    "IrisCommandCleanupError",
    "IrisSandboxError",
    "IrisMCPError",
    "IrisMCPToolError",
    "IrisMCPCallError",
    "IrisMemoryError",
    "IrisLifecycleSchemaError",
    "IrisRunConflictError",
    "IrisRunNotFoundError",
    "IrisRunObservationTimeoutError",
    "IrisRunPersistenceError",
    "IrisRunRecoveryError",
    "IrisRunError",
    "IrisRunStateError",
    "IrisTemplateError",
    "IrisTemplateNotFoundError",
]
