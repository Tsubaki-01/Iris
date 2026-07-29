"""Human-in-the-loop 领域模型、协议和惰性 concrete exports。"""

from typing import TYPE_CHECKING

from .models import (
    HumanInteraction,
    HumanInteractionPrompt,
    HumanInteractionRequest,
    HumanInteractionResponse,
    InteractionKind,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
    make_call_fingerprint,
)

if TYPE_CHECKING:
    from .in_memory import InMemoryInteractionStore
    from .service import HumanInteractionService
    from .store import InteractionStore


def __getattr__(name: str) -> object:
    """按需加载旧 HITL service/store，保持纯模型 import 无副作用。"""
    if name == "InMemoryInteractionStore":
        from .in_memory import InMemoryInteractionStore

        value: object = InMemoryInteractionStore
    elif name == "HumanInteractionService":
        from .service import HumanInteractionService

        value = HumanInteractionService
    elif name == "InteractionStore":
        from .store import InteractionStore

        value = InteractionStore
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让惰性公开符号仍可被 IDE 与 introspection 发现。"""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "HumanInteraction",
    "HumanInteractionPrompt",
    "HumanInteractionRequest",
    "HumanInteractionResponse",
    "HumanInteractionService",
    "InMemoryInteractionStore",
    "InteractionKind",
    "InteractionResumePhase",
    "InteractionStatus",
    "InteractionStore",
    "PermissionPrompt",
    "PermissionInteractionResponse",
    "QuestionPrompt",
    "QuestionInteractionResponse",
    "ToolCallSnapshot",
    "make_call_fingerprint",
]
