"""Agent 工具声明解析。"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ...exceptions import IrisConfigError
from ...tools import ToolRegistry, WorkspaceFileService
from ...tools.builtin.file import (
    EditFileTool,
    GrepSearchTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from .base import ToolsConfig

_BUILTIN_TOOL_CLASSES = {
    "file.read": ReadFileTool,
    "file.list": ListFilesTool,
    "file.grep": GrepSearchTool,
    "file.write": WriteFileTool,
    "file.edit": EditFileTool,
}


def build_tool_registry(config: ToolsConfig) -> ToolRegistry:
    """根据 Agent 工具配置构建工具注册表。

    Args:
        config (ToolsConfig): 已校验的工具配置。

    Returns:
        ToolRegistry: 已注册配置声明工具的注册表。

    Raises:
        IrisConfigError: 工具名称或 Python 引用无法解析时抛出。
    """
    registry = ToolRegistry()
    _register_builtin_tools(registry, config.builtin)
    for ref in config.python.functions:
        registry.register_function(_import_ref(ref))
    for ref in config.python.registrars:
        registrar = _import_ref(ref)
        try:
            registrar(registry)
        except TypeError as exc:
            raise IrisConfigError(
                "Python registrar 必须接收 ToolRegistry 参数",
                ref=ref,
            ) from exc
    return registry


def _register_builtin_tools(registry: ToolRegistry, names: list[str]) -> None:
    """注册 YAML 声明的内置工具。"""
    file_service = WorkspaceFileService()
    for name in names:
        tool_cls = _BUILTIN_TOOL_CLASSES.get(name)
        if tool_cls is None:
            raise IrisConfigError("未知内置工具", tool=name)
        registry.register(tool_cls(file_service=file_service))


def _import_ref(ref: str) -> Callable[..., Any]:
    """导入 `module:function` Python 引用。

    Args:
        ref (str): Python 引用字符串。

    Returns:
        Callable[..., Any]: 导入后的可调用对象。

    Raises:
        IrisConfigError: 引用格式、模块、属性或可调用性不合法时抛出。
    """
    module_name, separator, function_name = ref.partition(":")
    if not separator or not module_name or not function_name:
        raise IrisConfigError("Python 引用必须使用 module:function 格式", ref=ref)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise IrisConfigError("Python 引用模块不存在", ref=ref) from exc
    try:
        target = getattr(module, function_name)
    except AttributeError as exc:
        raise IrisConfigError("Python 引用函数不存在", ref=ref) from exc
    if not callable(target):
        raise IrisConfigError("Python 引用目标不可调用", ref=ref)
    return target


__all__ = ["build_tool_registry"]
