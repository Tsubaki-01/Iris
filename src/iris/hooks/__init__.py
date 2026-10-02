"""Hooks 公共数据与回调接口；派发器和命令适配器不在此处 eager 导入。"""

from .models import (
    HookEvent,
    HookEventName,
    HookHandler,
    HookRegistration,
    HookResult,
    RunFinishedEvent,
    RunStartedEvent,
    ToolAfterEvent,
    ToolAfterResult,
    ToolBeforeEvent,
    ToolBeforeResult,
    event_to_dict,
)

__all__ = [
    "HookEvent",
    "HookEventName",
    "HookHandler",
    "HookRegistration",
    "HookResult",
    "RunFinishedEvent",
    "RunStartedEvent",
    "ToolAfterEvent",
    "ToolAfterResult",
    "ToolBeforeEvent",
    "ToolBeforeResult",
    "event_to_dict",
]
