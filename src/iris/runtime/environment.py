"""Runtime 构造期依赖环境。

本模块定义运行时依赖的最小协议和容器。环境只保存同一个 ``AgentRuntime``
生命周期内复用的 live object，不承担配置解析或 checkpoint 序列化职责。

Example:
    environment = RuntimeEnvironment(
        agent_config=config,
        context_input=context_input,
        provider=provider,
    )
"""

# region imports
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..agents import AgentConfig
from ..context import ContextBuilder, ContextBuildInput, ContextSource
from ..memory import MemoryService
from ..message import LLMRequest, ModelStreamEvent
from ..providers.protocols import CompletionProvider
from ..skill import SkillRegistry
from ..tools import ToolExecutor, ToolRegistry
from ..utils import TemplateRenderer
from .assembler import RuntimeMessageAssembler
from .tool_bridge import ToolBridge

if TYPE_CHECKING:
    from ..mcp.manager import MCPManager
    from ..mcp.models import MCPCatalogSnapshot

# endregion


class RuntimeExecutionScope(StrEnum):
    """内部执行范围，child 不拥有自动记忆维护或递归委派。"""

    ROOT = "root"
    CHILD = "child"


class RuntimeMemoryCapturePort(Protocol):
    """Runtime 向 harness 通知可捕获原文的轻量端口。"""

    def request_capture(self, run_id: str, through_count: int) -> None:
        """合并已提交原文范围，不等待模型或持久 IO。"""


@runtime_checkable
class StreamingRuntimeProvider(Protocol):
    """Runtime 可选检测的 provider streaming capability。"""

    def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """返回一次 provider-neutral typed event stream。

        Args:
            request (LLMRequest): 已启用 streaming 的可信请求。

        Returns:
            AsyncIterator[ModelStreamEvent]: 顺序产生的模型流式事件。
        """


def streaming_provider_for(
    provider: CompletionProvider,
) -> StreamingRuntimeProvider | None:
    """返回 provider 的独立 streaming capability。

    Args:
        provider (CompletionProvider): Runtime 当前绑定的 complete capability。

    Returns:
        StreamingRuntimeProvider | None: 可用的 stream capability；缺失时返回 ``None``。
    """
    if isinstance(provider, StreamingRuntimeProvider):
        return provider
    return None


def _default_tool_bridge() -> ToolBridge:
    """构造相互一致的空工具视图与执行器。"""
    registry = ToolRegistry()
    return ToolBridge(
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
    )


@dataclass(slots=True)
class RuntimeEnvironment:
    """一个 runtime 实例的构造期依赖集合。

    该容器只保存 inner engine 的 live dependencies；durable lifecycle store
    由 harness 独占。调用级选项由 ``RuntimeExecutionOptions`` 管理。

    Attributes:
        agent_config (AgentConfig): 已校验的 Agent 配置快照。
        context_input (ContextBuildInput): context 构建输入。
        provider (CompletionProvider): provider-neutral 调用边界。
        context_builder (ContextBuilder): 固定 context 生成器。
        prompt_renderer (TemplateRenderer): runtime 独立指令的模板渲染器。
        assembler (RuntimeMessageAssembler): provider 请求装配器。
        tool_bridge (ToolBridge): 工具可见性、预检与执行边界。
        workspace_root (Path): 工具执行使用的 workspace 根路径。
        memory_service (MemoryService | None): 配置构造或宿主注入的可选 memory 服务。
        skill_registry (SkillRegistry | None): 构造时发现的 Skill 目录元数据快照。
        mcp_manager (MCPManager | None): 当前 runtime 独占的 MCP 资源与目录 owner。
        execution_scope (RuntimeExecutionScope): 明确的 ROOT/CHILD 装配范围。
        memory_capture_port (RuntimeMemoryCapturePort | None): root harness 绑定的原文捕获提示端口。
        context_source (ContextSource | None): 宿主每步采集接口，不缓存快照。
    """

    agent_config: AgentConfig
    context_input: ContextBuildInput
    provider: CompletionProvider
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)
    prompt_renderer: TemplateRenderer = field(default_factory=TemplateRenderer)
    assembler: RuntimeMessageAssembler = field(default_factory=RuntimeMessageAssembler)
    tool_bridge: ToolBridge = field(default_factory=_default_tool_bridge)
    workspace_root: Path = field(default_factory=Path.cwd)
    memory_service: MemoryService | None = None
    skill_registry: SkillRegistry | None = None
    mcp_manager: MCPManager | None = None
    execution_scope: RuntimeExecutionScope = RuntimeExecutionScope.ROOT
    memory_capture_port: RuntimeMemoryCapturePort | None = None
    context_source: ContextSource | None = None

    def __post_init__(self) -> None:
        """归一化工具执行的 workspace 根路径。"""
        self.workspace_root = self.workspace_root.resolve()

    async def aprepare(self) -> MCPCatalogSnapshot | None:
        """准备并发布当前环境的 MCP 工具，直接委托唯一 manager。"""
        if self.mcp_manager is not None:
            return await self.mcp_manager.prepare()
        return None

    async def aclose(self) -> None:
        """关闭自有 MCP 资源；注入的 provider、memory 和 store 由 host 管理。"""
        if self.mcp_manager is not None:
            await self.mcp_manager.aclose()


__all__ = [
    "RuntimeEnvironment",
    "RuntimeExecutionScope",
    "RuntimeMemoryCapturePort",
    "StreamingRuntimeProvider",
    "streaming_provider_for",
]
