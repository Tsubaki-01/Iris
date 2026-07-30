"""Human-in-the-loop typed domain models 与无状态服务。"""

from typing import TYPE_CHECKING

from .models import (
    ApprovedToolCall,
    HumanInteraction,
    HumanInteractionPrompt,
    HumanInteractionRequest,
    HumanInteractionResponse,
    InteractionKind,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
    make_call_fingerprint,
)

if TYPE_CHECKING:
    from .service import HumanInteractionService


def __getattr__(name: str) -> object:
    """延迟加载 stateless service，保持 lifecycle contract import 无环。"""
    if name != "HumanInteractionService":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .service import HumanInteractionService

    globals()[name] = HumanInteractionService
    return HumanInteractionService


__all__ = [
    "ApprovedToolCall",
    "HumanInteraction",
    "HumanInteractionPrompt",
    "HumanInteractionRequest",
    "HumanInteractionResponse",
    "HumanInteractionService",
    "InteractionKind",
    "InteractionStatus",
    "PermissionPrompt",
    "PermissionInteractionResponse",
    "QuestionPrompt",
    "QuestionInteractionResponse",
    "ToolCallSnapshot",
    "make_call_fingerprint",
]
