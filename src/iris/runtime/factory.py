"""Runtime 配置装配入口。

本模块把 Agent YAML 或 SDK 构造的 `AgentConfig` 转换为可运行的
`AgentRuntime` 依赖图；真实模型调用仍延迟到 runtime 执行阶段。

Example:
    runtime = RuntimeFactory.from_config_path(
        "agent.yaml", provider=fake_provider, context_access=host_context_access
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..agents import AgentConfig, load_agent_config
from ..context import ContextSource
from ..exceptions import IrisConfigError
from ..providers.protocols import CompletionProvider
from ..tools.context_access import ContextAccessPort
from ._assembly import RuntimeExecutionScope, assemble_runtime, resolve_runtime_boundary
from .runtime import AgentRuntime

if TYPE_CHECKING:
    from ..decision import DecisionEvaluator
    from ..hooks import HookRegistration
    from ..memory import MemoryService
    from ..tools import ToolMiddleware


class RuntimeFactory:
    """从配置构造 `AgentRuntime`。

    Factory 只负责本地依赖装配，不调用 provider 网络接口。显式注入的 provider、
    memory service 优先于配置派生对象，便于测试和 SDK 用户接管边界。

    Example:
        runtime = RuntimeFactory.from_config(
            config, provider=fake_provider, context_access=host_context_access
        )
    """

    @classmethod
    def from_config_path(
        cls,
        path: str | Path,
        *,
        provider: CompletionProvider | None = None,
        memory_service: MemoryService | None = None,
        decision_client: DecisionEvaluator | None = None,
        context_access: ContextAccessPort | None = None,
        context_source: ContextSource | None = None,
        hooks: Sequence[HookRegistration] = (),
        tool_middlewares: Sequence[ToolMiddleware] = (),
        api_key: str | None = None,
    ) -> AgentRuntime:
        """从 `agent.yaml` 路径构造 runtime。

        Args:
            path (str | Path): Agent YAML 配置文件路径。
            provider (CompletionProvider | None): 可选 provider 注入；存在时不创建真实 client。
            memory_service (MemoryService | None): 优先于配置后端的 memory 服务注入。
            decision_client: 可选借用的判断能力，是否挂载由接点配置决定。
            context_access (ContextAccessPort | None): context_policy 启用时必需的宿主回读协议。
            context_source (ContextSource | None): 可选的每步运行态采集接口。
            hooks (Sequence[HookRegistration]): 追加在 YAML 项之后的当前 Agent 处理器。
            tool_middlewares (Sequence[ToolMiddleware]): 追加在 YAML 项之后的工具包装实例。
            api_key (str | None): 创建真实 provider client 时使用的 API key。

        Returns:
            AgentRuntime: 已装配但尚未调用 provider 的 runtime 实例。
        """
        config_path = Path(path)
        config = load_agent_config(config_path)
        return cls.from_config(
            config,
            config_path=config_path,
            provider=provider,
            memory_service=memory_service,
            decision_client=decision_client,
            context_access=context_access,
            context_source=context_source,
            hooks=hooks,
            tool_middlewares=tool_middlewares,
            api_key=api_key,
        )

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        config_path: Path | None = None,
        provider: CompletionProvider | None = None,
        memory_service: MemoryService | None = None,
        decision_client: DecisionEvaluator | None = None,
        context_access: ContextAccessPort | None = None,
        context_source: ContextSource | None = None,
        hooks: Sequence[HookRegistration] = (),
        tool_middlewares: Sequence[ToolMiddleware] = (),
        api_key: str | None = None,
    ) -> AgentRuntime:
        """从已校验的 `AgentConfig` 构造 runtime。

        Args:
            config (AgentConfig): 已校验的 Agent 配置。
            config_path (Path | None): 配置文件路径；相对它解析 workspace、context 和摘要 prompt。
            provider (CompletionProvider | None): 可选 provider 注入；存在时不创建真实 client。
            memory_service (MemoryService | None): 优先于配置后端的 memory 服务注入。
            decision_client: 可选借用的判断能力，是否挂载由接点配置决定。
            context_access (ContextAccessPort | None): context_policy 启用时必需的宿主回读协议。
            context_source (ContextSource | None): 可选的每步运行态采集接口。
            hooks (Sequence[HookRegistration]): 追加在 YAML 项之后的当前 Agent 处理器。
            tool_middlewares (Sequence[ToolMiddleware]): 追加在 YAML 项之后的工具包装实例。
            api_key (str | None): 创建真实 provider client 时使用的 API key。

        Returns:
            AgentRuntime: 已装配的 runtime 实例。
        """
        if config.tools.subagent is not None:
            raise IrisConfigError("tools.subagent 需要通过 AgentRunner.from_config* 构造")
        if config.goal.enabled:
            raise IrisConfigError("goal.enabled 需要通过 AgentRunner.from_config* 构造")
        return assemble_runtime(
            config,
            config_path=config_path,
            provider=provider,
            memory_service=memory_service,
            decision_client=decision_client,
            context_access=context_access,
            context_source=context_source,
            hooks=hooks,
            tool_middlewares=tool_middlewares,
            api_key=api_key,
            execution_scope=RuntimeExecutionScope.ROOT,
            boundary=resolve_runtime_boundary(config, config_path=config_path),
        )


__all__ = ["RuntimeFactory"]
