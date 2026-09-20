"""Agent 工具声明解析。"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ...config import get_config
from ...exceptions import IrisConfigError
from ...memory import (
    MEMORY_TOOL_CLASSES,
    MemoryConfig,
    MemoryService,
    default_memory_access_policy_factory,
)
from ...tools import AskQuestionTool, ToolRegistry, WorkspaceFileService
from ...tools.base import BaseTool
from ...tools.builtin.file import (
    EditFileTool,
    GrepSearchTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from ...tools.builtin.web import WebFetchTool, WebSearchTool
from .base import ToolsConfig

_FileToolFactory = Callable[[WorkspaceFileService], BaseTool]
_ToolFactory = Callable[[], BaseTool]

_BUILTIN_FILE_TOOL_FACTORIES: dict[str, _FileToolFactory] = {
    "file.read": lambda service: ReadFileTool(file_service=service),
    "file.list": lambda service: ListFilesTool(file_service=service),
    "file.grep": lambda service: GrepSearchTool(file_service=service),
    "file.write": lambda service: WriteFileTool(file_service=service),
    "file.edit": lambda service: EditFileTool(file_service=service),
}

_BUILTIN_HUMAN_TOOL_FACTORIES: dict[str, _ToolFactory] = {
    "human.ask": AskQuestionTool,
}

_BUILTIN_WEB_TOOL_CLASSES: dict[str, type[WebSearchTool] | type[WebFetchTool]] = {
    "web.search": WebSearchTool,
    "web.fetch": WebFetchTool,
}


def build_tool_registry(
    config: ToolsConfig,
    *,
    memory_service: MemoryService | None = None,
    memory_config: MemoryConfig | None = None,
) -> ToolRegistry:
    """根据 Agent 工具配置构建工具注册表。

    Args:
        config (ToolsConfig): 已校验的工具配置。
        memory_service: 可选 memory service；存在时自动注册三个读取工具。
        memory_config: 绑定工具的读取范围和单个写入 namespace。

    Returns:
        ToolRegistry: 已注册配置声明工具的注册表。

    Raises:
        IrisConfigError: 工具名称或 Python 引用无法解析时抛出。
    """
    registry = ToolRegistry()
    builtin_names = list(config.builtin)
    if memory_service is not None:
        builtin_names.extend(
            name
            for name in ("memory.search", "memory.list", "memory.get")
            if name not in builtin_names
        )
    _register_builtin_tools(
        registry,
        builtin_names,
        memory_service=memory_service,
        memory_config=memory_config or MemoryConfig(),
    )
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


def _register_builtin_tools(
    registry: ToolRegistry,
    names: list[str],
    *,
    memory_service: MemoryService | None,
    memory_config: MemoryConfig,
) -> None:
    """注册 YAML 声明的内置工具。"""
    file_service = WorkspaceFileService(
        memory_view=(
            memory_service.file_access(memory_config.read_namespaces)
            if memory_service is not None
            else None
        )
    )
    for name in names:
        if name.startswith("file."):
            factory = _BUILTIN_FILE_TOOL_FACTORIES.get(name)
            if factory is None:
                raise IrisConfigError("未知内置工具", tool=name)
            registry.register(factory(file_service))
        elif name in _BUILTIN_WEB_TOOL_CLASSES:
            api_key = get_config().provider_api_keys.get("tavily")
            if api_key is None:
                raise IrisConfigError("Web 工具需要配置 IRIS_PROVIDER_API_KEYS__TAVILY", tool=name)
            registry.register(_BUILTIN_WEB_TOOL_CLASSES[name](api_key=api_key))
        elif name in MEMORY_TOOL_CLASSES:
            if memory_service is None:
                raise IrisConfigError("memory 工具需要启用或注入 memory service", tool=name)
            registry.register(
                MEMORY_TOOL_CLASSES[name](
                    service=memory_service,
                    access_policy_factory=default_memory_access_policy_factory(memory_config),
                )
            )
        else:
            human_factory = _BUILTIN_HUMAN_TOOL_FACTORIES.get(name)
            if human_factory is None:
                raise IrisConfigError("未知内置工具", tool=name)
            registry.register(human_factory())


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
