"""工具轻量装饰器。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .base import ToolCapability

F = TypeVar("F", bound=Callable[..., Any])


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: set[ToolCapability] | None = None,
    group: str = "core",
    deferred: bool = False,
) -> Callable[[F], F]:
    """为函数附加 Iris 工具元数据，不自动注册。"""

    def decorator(func: F) -> F:
        func.__dict__["iris_tool_name"] = name
        func.__dict__["iris_tool_description"] = description
        func.__dict__["iris_tool_capabilities"] = set(capabilities or set())
        func.__dict__["iris_tool_group"] = group
        func.__dict__["iris_tool_deferred"] = deferred
        return func

    return decorator
