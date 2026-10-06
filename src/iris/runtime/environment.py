"""Runtime 构造期依赖环境。

本模块定义运行时依赖的最小协议和容器。环境只保存同一个 ``AgentRuntime``
生命周期内复用的 live object，不承担配置解析或 checkpoint 序列化职责。

Example:
    environment = RuntimeEnvironment(
        agent_config=config,
        context_input=context_input,
        provider=provider,
        prompt_source=prompt_source,
        prompt_snapshot=prompt_source.snapshot(),
    )
"""

# region imports
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..agents import AgentConfig
from ..command.models import CommandEnvironment, CommandStopSlot
from ..command.service import CommandBinding
from ..context import ContextBuilder, ContextBuildInput, ContextSource
from ..memory import MemoryService
from ..observability.service import Observability
from ..prompts import PromptSnapshot, PromptSource
from ..providers.protocols import CompletionProvider
from ..skill import SkillRegistry
from ..tools import ToolExecutor, ToolRegistry
from .assembler import RuntimeMessageAssembler
from .tool_bridge import ToolBridge

if TYPE_CHECKING:
    from ..decision import DecisionEvaluator, JevClient
    from ..goal.service import GoalService
    from ..hooks.dispatcher import HookDispatcher
    from ..mcp.manager import MCPManager
    from ..mcp.models import MCPCatalogSnapshot

# endregion


class RuntimeExecutionScope(StrEnum):
    """内部执行范围，child 不拥有自动记忆维护或递归委派。"""

    ROOT = "root"
    CHILD = "child"


class RuntimeCapturePort(Protocol):
    """Runtime 向 harness 通知可捕获原文的轻量端口。"""

    def request_capture(self, run_id: str, through_count: int) -> None:
        """合并已提交原文范围，不等待模型或持久 IO。"""


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
        prompt_source (PromptSource): root 初始化、child 借用的项目模板来源。
        prompt_snapshot (PromptSnapshot): 当前 runtime 构造期采用的正文快照。
        assembler (RuntimeMessageAssembler): provider 请求装配器。
        tool_bridge (ToolBridge): 工具可见性、预检与执行边界。
        workspace_root (Path): 工具执行使用的 workspace 根路径。
        memory_service (MemoryService | None): 配置构造或宿主注入的可选 memory 服务。
        goal_service (GoalService | None): harness 注入的可选目标服务，不参与执行循环控制。
        skill_registry (SkillRegistry | None): 构造时发现的 Skill 目录元数据快照。
        mcp_manager (MCPManager | None): 当前 runtime 独占的 MCP 资源与目录 owner。
        execution_scope (RuntimeExecutionScope): 明确的 ROOT/CHILD 装配范围。
        command_binding (CommandBinding | None): root 拥有、child 借用的命令服务与配置。
        command_environment (CommandEnvironment | None): 本 Agent 注册命令工具时的环境事实。
        host_os (str): Iris 进程所在宿主操作系统。
        command_stop_slots (dict): root/child 共享的当前调用停止事实槽。
        capture_port (RuntimeCapturePort | None): root harness 绑定的原文捕获提示端口。
        context_source (ContextSource | None): 宿主每步采集接口，不缓存快照。
        hook_dispatcher (HookDispatcher | None): 当前 Agent 的可选进程内 Hook 派发依赖。
        decision_client (DecisionEvaluator | None): 当前已启用接点共同借用的判断能力。
        owned_decision_client (JevClient | None): 本环境自建且负责关闭的判断客户端。
        observability (Observability): 当前 Agent 的固定观测策略，工具执行器借用同一实例。
        owned_observability (Observability | None): 装配自建且在业务资源关闭后收口的服务。
    """

    agent_config: AgentConfig
    context_input: ContextBuildInput
    provider: CompletionProvider
    prompt_source: PromptSource
    prompt_snapshot: PromptSnapshot
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)
    assembler: RuntimeMessageAssembler = field(default_factory=RuntimeMessageAssembler)
    tool_bridge: ToolBridge = field(default_factory=_default_tool_bridge)
    workspace_root: Path = field(default_factory=Path.cwd)
    memory_service: MemoryService | None = None
    goal_service: GoalService | None = None
    skill_registry: SkillRegistry | None = None
    mcp_manager: MCPManager | None = None
    execution_scope: RuntimeExecutionScope = RuntimeExecutionScope.ROOT
    command_binding: CommandBinding | None = None
    command_environment: CommandEnvironment | None = None
    host_os: str = field(default_factory=platform.system)
    command_stop_slots: dict[tuple[str, str], CommandStopSlot] = field(default_factory=dict)
    capture_port: RuntimeCapturePort | None = None
    context_source: ContextSource | None = None
    hook_dispatcher: HookDispatcher | None = None
    decision_client: DecisionEvaluator | None = None
    owned_decision_client: JevClient | None = None
    observability: Observability = field(default_factory=Observability)
    owned_observability: Observability | None = None

    def __post_init__(self) -> None:
        """归一化 workspace，交接当前 Agent 的 Hooks、命令和观测依赖。"""
        self.workspace_root = self.workspace_root.resolve()
        self.tool_bridge.command_stop_slots = self.command_stop_slots
        self.tool_bridge.tool_executor.hook_dispatcher = self.hook_dispatcher
        self.tool_bridge.tool_executor.command_binding = self.command_binding
        self.tool_bridge.tool_executor.observability = self.observability

    async def aprepare(self) -> MCPCatalogSnapshot | None:
        """准备绑定的命令服务和本环境自有 MCP，返回 MCP 目录快照。"""
        if self.command_binding is not None:
            await self.command_binding.service.prepare()
        if self.mcp_manager is not None:
            return await self.mcp_manager.prepare()
        return None

    async def aclose(self) -> None:
        """依次关闭业务资源，最后收口自建观测；借用服务由其宿主关闭。"""
        try:
            try:
                if self.mcp_manager is not None:
                    await self.mcp_manager.aclose()
            finally:
                try:
                    if (
                        self.execution_scope is RuntimeExecutionScope.ROOT
                        and self.command_binding is not None
                    ):
                        await self.command_binding.service.aclose()
                finally:
                    if self.owned_decision_client is not None:
                        await self.owned_decision_client.aclose()
        finally:
            owned, self.owned_observability = self.owned_observability, None
            if owned is not None:
                await owned.aclose()


__all__ = [
    "RuntimeEnvironment",
    "RuntimeExecutionScope",
    "RuntimeCapturePort",
]
