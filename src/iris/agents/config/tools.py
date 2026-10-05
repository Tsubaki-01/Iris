"""Agent 工具声明解析。"""

from __future__ import annotations

from collections.abc import Callable

from ...command.service import CommandBinding
from ...config import get_config
from ...decision import DecisionEvaluator
from ...exceptions import IrisConfigError
from ...memory import (
    MEMORY_TOOL_CLASSES,
    MemoryConfig,
    MemoryFetchTool,
    MemorySearchTool,
    MemoryService,
    default_memory_access_policy_factory,
)
from ...prompts import PromptSnapshot
from ...tools import AskQuestionTool, ToolRegistry, WorkspaceFileService
from ...tools.base import BaseTool
from ...tools.builtin.artifact import PublishArtifactTool
from ...tools.builtin.exec import ExecCommandTool
from ...tools.builtin.file import (
    EditFileTool,
    GrepSearchTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from ...tools.builtin.python import RunPythonTool
from ...tools.builtin.web import WebFetchTool, WebSearchTool
from ._imports import import_ref
from .base import ToolsConfig

_FileToolFactory = Callable[[WorkspaceFileService], BaseTool]
_ToolFactory = Callable[[], BaseTool]

_BUILTIN_COMMAND_TOOL_CLASSES: dict[str, type[ExecCommandTool] | type[RunPythonTool]] = {
    "exec.command": ExecCommandTool,
    "exec.python": RunPythonTool,
}

_BUILTIN_FILE_TOOL_FACTORIES: dict[str, _FileToolFactory] = {
    "file.read": lambda service: ReadFileTool(file_service=service),
    "file.list": lambda service: ListFilesTool(file_service=service),
    "file.grep": lambda service: GrepSearchTool(file_service=service),
    "file.write": lambda service: WriteFileTool(file_service=service),
    "file.edit": lambda service: EditFileTool(file_service=service),
    "file.publish": lambda service: PublishArtifactTool(file_service=service),
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
    memory_decision_client: DecisionEvaluator | None = None,
    command_binding: CommandBinding | None = None,
    prompt_snapshot: PromptSnapshot | None = None,
) -> ToolRegistry:
    """根据 Agent 工具配置构建工具注册表。

    Args:
        config (ToolsConfig): 已校验的工具配置。
        memory_service: 来源工厂已解析的服务，存在时自动绑定双读工具及文件读取范围。
        memory_config: 绑定工具的读取范围和单个写入 namespace。
        memory_decision_client: 仅由 Search 借用的可选判断能力，服务和其他工具不保存它。
        command_binding: root 已装配的命令服务与环境；显式 exec.command/exec.python 消费。
        prompt_snapshot: 构造时固定的项目模板，供可选 Decision 指令使用。

    Returns:
        ToolRegistry: 已注册配置声明工具的注册表。

    Raises:
        IrisConfigError: 工具名称或 Python 引用无法解析时抛出。
    """
    registry = ToolRegistry()
    _register_builtin_tools(
        registry,
        list(config.builtin),
        memory_service=memory_service,
        memory_config=memory_config or MemoryConfig(),
        memory_decision_client=memory_decision_client,
        command_binding=command_binding,
        prompt_snapshot=prompt_snapshot,
    )
    for ref in config.python.functions:
        registry.register_function(import_ref(ref))
    for ref in config.python.registrars:
        registrar = import_ref(ref)
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
    memory_decision_client: DecisionEvaluator | None,
    command_binding: CommandBinding | None,
    prompt_snapshot: PromptSnapshot | None,
) -> None:
    """先绑定有效记忆服务的双读工具，再注册 YAML 声明的其它内置工具。"""
    memory_policy = default_memory_access_policy_factory(memory_config)
    if memory_service is not None:
        registry.register(
            MemorySearchTool(
                service=memory_service,
                access_policy_factory=memory_policy,
                decision_client=memory_decision_client,
                prompt_snapshot=prompt_snapshot,
            )
        )
        registry.register(
            MemoryFetchTool(service=memory_service, access_policy_factory=memory_policy)
        )
    file_service = WorkspaceFileService(
        memory_view=(
            memory_service.file_access(memory_config.read_namespaces)
            if memory_service is not None
            else None
        )
    )
    for name in names:
        if name in _BUILTIN_COMMAND_TOOL_CLASSES:
            if command_binding is None:
                raise IrisConfigError(f"{name} 需要装配层提供 command_binding", tool=name)
            registry.register(_BUILTIN_COMMAND_TOOL_CLASSES[name](command_binding))
        elif name in ("memory.search", "memory.fetch"):
            raise IrisConfigError(
                "memory 读取工具由 memory.enabled 自动启用，请移除手工声明", tool=name
            )
        elif name.startswith("file."):
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
                raise IrisConfigError("memory 写工具需要开启 memory.enabled", tool=name)
            registry.register(
                MEMORY_TOOL_CLASSES[name](
                    service=memory_service,
                    access_policy_factory=memory_policy,
                )
            )
        else:
            human_factory = _BUILTIN_HUMAN_TOOL_FACTORIES.get(name)
            if human_factory is None:
                raise IrisConfigError("未知内置工具", tool=name)
            registry.register(human_factory())


__all__ = ["build_tool_registry"]
