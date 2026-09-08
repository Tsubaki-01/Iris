"""Iris 消息模型公共导出。

消息包导出内部消息、会话和 LLM 调用模型。`LLMRequest` 与 `LLMResponse`
的唯一定义位于 `iris.message.llm`，provider 层应从这里导入。

Example:
    >>> from iris.message import LLMRequest, Msg
    >>> LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]).model
    'gpt-4o'
"""

# region imports
from iris.message.llm import LLMRequest, LLMResponse
from iris.message.message import (
    ContentBlock,
    Conversation,
    Msg,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from .streaming import (
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockKind,
    ModelBlockRef,
    ModelBlockStarted,
    ModelDeltaChannel,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ModelUsageSnapshot,
    ModelUsageUpdated,
    ProviderStreamError,
)

# endregion

__all__ = [
    "ContentBlock",
    "Conversation",
    "LLMRequest",
    "LLMResponse",
    "ModelBlockCompleted",
    "ModelBlockDelta",
    "ModelBlockKind",
    "ModelBlockRef",
    "ModelBlockStarted",
    "ModelDeltaChannel",
    "ModelResponseCancelled",
    "ModelResponseCompleted",
    "ModelResponseFailed",
    "ModelResponseStarted",
    "ModelStreamEvent",
    "ModelStreamScope",
    "ModelUsageSnapshot",
    "ModelUsageUpdated",
    "Msg",
    "ProviderStreamError",
    "Role",
    "TextBlock",
    "ToolResultBlock",
    "ToolUseBlock",
]
