"""工具轻量装饰器。

提供将普通 Python 函数标记为 Iris 工具的非侵入式方法。
本模块的设计目的是允许开发者跨包定义受管方法，而不必在定义时就立即绑定到执行中心的注册表实例。

Example:
    @tool(name="my_tool", description="Does something.")
    def my_tool():
        pass
"""

# region imports
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .base import ToolCapability

# endregion

# ==========================================
#                 Constants
# ==========================================
# region constants
F = TypeVar("F", bound=Callable[..., Any])  # 用于保持被装饰函数签名的泛型类型变量。
# endregion


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: set[ToolCapability] | None = None,
    group: str = "core",
    deferred: bool = False,
) -> Callable[[F], F]:
    """为函数附加 Iris 工具元数据，不自动触发注册。

    将工具属性直接以 `iris_tool_` 前缀的形式注入到函数的 `__dict__` 中，
    这样既不会改变签名的表现形式，又能被 `register_function` 透视读取。
    这满足了“定义即声明，注册可延后”的上下文分离需求。

    Args:
        name (str | None): 重定义的工具名。如未提供，注册时将回退到函数名称。
        description (str | None): 人类与模型可读的具体说明。未提供则回退使用函数的 docstring。
        capabilities (set[ToolCapability] | None): 工具支持的特殊能力标签集合。
        group (str): 权限组划分标签，便于后续执行时实现粗粒度的组级风控拦截。
        deferred (bool): 声明为延迟工具，如果不通过白名单显式指定，模型侧默认不可见此工具。

    Returns:
        Callable[[F], F]: 接收靶函数并回传其原引用的装饰器闭包。

    Example:
        >>> @tool(group="file", deferred=True)
        ... def read_file(path: str) -> str:
        ...     return "content"
    """

    def decorator(func: F) -> F:
        func.__dict__["iris_tool_name"] = name
        func.__dict__["iris_tool_description"] = description
        func.__dict__["iris_tool_capabilities"] = set(capabilities or set())
        func.__dict__["iris_tool_group"] = group
        func.__dict__["iris_tool_deferred"] = deferred
        return func

    return decorator
