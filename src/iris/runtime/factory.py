"""Runtime 配置装配入口。

本模块把 Agent YAML 或 SDK 构造的 `AgentConfig` 转换为可运行的
`AgentRuntime` 依赖图；真实模型调用仍延迟到 runtime 执行阶段。

Example:
    runtime = RuntimeFactory.from_config_path("agent.yaml", provider=fake_provider)
"""

# region imports
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..agents import AgentConfig, build_tool_registry, load_agent_config
from ..context import (
    ContextBuildInput,
    ContextSection,
    ContextSlot,
    load_context_build_input,
)
from ..providers import create_provider_client
from ..tools import DefaultPermissionPolicy, ToolExecutor
from .environment import RuntimeEnvironment, RuntimeProvider
from .runtime import AgentRuntime
from .tool_bridge import ToolBridge

if TYPE_CHECKING:
    from ..memory import MemoryService
# endregion


class RuntimeFactory:
    """从配置构造 `AgentRuntime`。

    Factory 只负责本地依赖装配，不调用 provider 网络接口。显式注入的 provider、
    provider 和 memory service 优先于配置派生对象，便于测试和 SDK 用户接管边界。

    Example:
        runtime = RuntimeFactory.from_config(config, provider=fake_provider)
    """

    @classmethod
    def from_config_path(
        cls,
        path: str | Path,
        *,
        provider: RuntimeProvider | None = None,
        memory_service: MemoryService | None = None,
        api_key: str | None = None,
    ) -> AgentRuntime:
        """从 `agent.yaml` 路径构造 runtime。

        Args:
            path (str | Path): Agent YAML 配置文件路径。
            provider (RuntimeProvider | None): 可选 provider 注入；存在时不创建真实 client。
            memory_service (MemoryService | None): 预留给显式 memory 阶段的服务注入。
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
            api_key=api_key,
        )

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        config_path: Path | None = None,
        provider: RuntimeProvider | None = None,
        memory_service: MemoryService | None = None,
        api_key: str | None = None,
    ) -> AgentRuntime:
        """从已校验的 `AgentConfig` 构造 runtime。

        Args:
            config (AgentConfig): 已校验的 Agent 配置。
            config_path (Path | None): 配置文件路径；存在时相对它解析 workspace/context。
            provider (RuntimeProvider | None): 可选 provider 注入；存在时不创建真实 client。
            memory_service (MemoryService | None): 预留给显式 memory 阶段的服务注入。
            api_key (str | None): 创建真实 provider client 时使用的 API key。

        Returns:
            AgentRuntime: 已装配的 runtime 实例。
        """
        base_dir = _base_dir(config_path)
        context_input = _build_context_input(config, base_dir=base_dir)
        tool_registry = build_tool_registry(config.tools)
        tool_view = tool_registry.view()
        permission_policy = DefaultPermissionPolicy(write_mode=config.permissions.writes)
        tool_executor = ToolExecutor(
            tool_registry,
            permission_policy=permission_policy,
        )
        resolved_provider = provider or create_provider_client(
            config.to_model_route(),
            api_key=api_key,
            base_url=config.model.base_url,
            timeout=config.model.timeout,
        )
        workspace_root = _resolve_relative_to_base(
            config.permissions.workspace,
            base_dir=base_dir,
        )

        tool_bridge = ToolBridge(
            tool_view=tool_view,
            tool_executor=tool_executor,
        )
        environment = RuntimeEnvironment(
            agent_config=config,
            context_input=context_input,
            provider=resolved_provider,
            tool_bridge=tool_bridge,
            workspace_root=workspace_root,
            memory_service=memory_service,
        )
        return AgentRuntime(environment)


def _base_dir(config_path: Path | None) -> Path:
    """返回配置相关路径解析基准目录。"""
    if config_path is None:
        return Path.cwd().resolve()
    return Path(config_path).parent.resolve()


def _build_context_input(config: AgentConfig, *, base_dir: Path) -> ContextBuildInput:
    """构造或加载 runtime 使用的 context 输入。"""
    if config.context is not None:
        return load_context_build_input(
            _resolve_relative_to_base(config.context.path, base_dir=base_dir)
        )
    return ContextBuildInput(
        system=ContextSection(
            slots=[
                ContextSlot(
                    name="instructions",
                    content=config.system or "",
                )
            ]
        )
    )


def _resolve_relative_to_base(path: str | Path, *, base_dir: Path) -> Path:
    """按配置基准目录解析路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


__all__ = ["RuntimeFactory"]
