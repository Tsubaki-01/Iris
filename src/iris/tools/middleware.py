"""工具调用的单次包装扩展点。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ToolResult


@dataclass(frozen=True, slots=True)
class ToolCall:
    """只读调用视图；arguments 是与真实执行参数隔离的独立快照。"""

    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any]
    agent_id: str
    session_id: str
    run_id: str | None
    activation_id: str | None
    workspace_root: Path


type ToolNext = Callable[[], Awaitable[ToolResult]]


class ToolMiddleware(ABC):
    """包装一次工具调用；首个注册项最外层，下游最多执行一次。"""

    @abstractmethod
    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        """返回替代结果或调用一次下游；下游结果只读，修改须返回新结果。"""
        raise NotImplementedError
