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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..agents import AgentConfig
from ..context import ContextBuilder, ContextBuildInput
from ..hitl.in_memory import InMemoryInteractionStore
from ..hitl._legacy_service import HumanInteractionService
from ..memory import MemoryContextBuilder, MemoryService
from ..message import LLMRequest, LLMResponse
from ..session import InMemorySessionStore, SessionStore
from ..tools import ToolExecutor, ToolRegistry
from .assembler import RuntimeMessageAssembler
from .tool_bridge import ToolBridge

# endregion


class RuntimeProvider(Protocol):
    """Runtime 调用的 provider 最小协议。

    Runtime 只依赖 provider-neutral 的请求与响应模型，使真实 client 和测试 fake
    可以在同一运行边界互换。

    Example:
        response = await provider.complete(request)
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """执行一次非流式 LLM 请求。

        Args:
            request (LLMRequest): Runtime 组装完成的 provider-neutral 请求。

        Returns:
            LLMResponse: Provider 归一化后的响应。
        """


def _default_tool_bridge() -> ToolBridge:
    """构造相互一致的空工具视图与执行器。"""
    registry = ToolRegistry()
    return ToolBridge(
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
    )


def _default_interaction_service() -> HumanInteractionService:
    """构造进程内默认 HITL service。"""
    return HumanInteractionService(InMemoryInteractionStore())


@dataclass(slots=True)
class RuntimeEnvironment:
    """一个 runtime 实例的构造期依赖集合。

    该容器把工具和 HITL 的派生依赖收敛到 ``ToolBridge`` 与
    ``HumanInteractionService``，避免调用方构造互不一致的对象图。调用级选项仍由
    ``RuntimeOptions`` 管理。

    Attributes:
        agent_config (AgentConfig): 已校验的 Agent 配置快照。
        context_input (ContextBuildInput): context 构建输入。
        provider (RuntimeProvider): provider-neutral 调用边界。
        session_store (SessionStore): 会话消息、metadata 与事件存储。
        context_builder (ContextBuilder): 固定 context 生成器。
        assembler (RuntimeMessageAssembler): provider 请求装配器。
        tool_bridge (ToolBridge): 工具可见性、预检与执行边界。
        interaction_service (HumanInteractionService): HITL 生命周期服务。
        workspace_root (Path): 工具执行使用的 workspace 根路径。
        memory_service (MemoryService | None): 显式可选 memory 服务。
        memory_context_builder (MemoryContextBuilder): memory context 裁剪器。
    """

    agent_config: AgentConfig
    context_input: ContextBuildInput
    provider: RuntimeProvider
    session_store: SessionStore = field(default_factory=InMemorySessionStore)
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)
    assembler: RuntimeMessageAssembler = field(default_factory=RuntimeMessageAssembler)
    tool_bridge: ToolBridge = field(default_factory=_default_tool_bridge)
    interaction_service: HumanInteractionService = field(
        default_factory=_default_interaction_service
    )
    workspace_root: Path = field(default_factory=Path.cwd)
    memory_service: MemoryService | None = None
    memory_context_builder: MemoryContextBuilder = field(default_factory=MemoryContextBuilder)

    def __post_init__(self) -> None:
        """归一化工具执行的 workspace 根路径。"""
        self.workspace_root = self.workspace_root.resolve()


__all__ = ["RuntimeEnvironment", "RuntimeProvider"]
