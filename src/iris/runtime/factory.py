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

import httpx

from ..agents import AgentConfig, build_tool_registry, load_agent_config
from ..context import (
    ContextBuildInput,
    ContextSection,
    ContextSlot,
    load_context_build_input,
)
from ..providers import create_provider_client
from ..session import InMemorySessionStore, SessionStore, SQLiteSessionStore
from ..tools import DefaultPermissionPolicy, ToolExecutor
from .runtime import AgentRuntime, RuntimeProvider

if TYPE_CHECKING:
    from ..memory import MemoryService
# endregion


class RuntimeFactory:
    """从配置构造 `AgentRuntime`。

    Factory 只负责本地依赖装配，不调用 provider 网络接口。显式注入的 provider、
    session store 和 memory service 优先于配置派生对象，便于测试和 SDK 用户接管边界。

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
        session_store: SessionStore | None = None,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> AgentRuntime:
        """从 `agent.yaml` 路径构造 runtime。

        Args:
            path (str | Path): Agent YAML 配置文件路径。
            provider (RuntimeProvider | None): 可选 provider 注入；存在时不创建真实 client。
            memory_service (MemoryService | None): 预留给显式 memory 阶段的服务注入。
            session_store (SessionStore | None): 可选 session store 注入。
            api_key (str | None): 创建真实 provider client 时使用的 API key。
            http_client (httpx.AsyncClient | None): 创建真实 provider client 时复用的 HTTP client。

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
            session_store=session_store,
            api_key=api_key,
            http_client=http_client,
        )

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        config_path: Path | None = None,
        provider: RuntimeProvider | None = None,
        memory_service: MemoryService | None = None,
        session_store: SessionStore | None = None,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> AgentRuntime:
        """从已校验的 `AgentConfig` 构造 runtime。

        Args:
            config (AgentConfig): 已校验的 Agent 配置。
            config_path (Path | None): 配置文件路径；存在时相对它解析 workspace/session。
            provider (RuntimeProvider | None): 可选 provider 注入；存在时不创建真实 client。
            memory_service (MemoryService | None): 预留给显式 memory 阶段的服务注入。
            session_store (SessionStore | None): 可选 session store 注入。
            api_key (str | None): 创建真实 provider client 时使用的 API key。
            http_client (httpx.AsyncClient | None): 创建真实 provider client 时复用的 HTTP client。

        Returns:
            AgentRuntime: 已装配的 runtime 实例。
        """
        base_dir = _base_dir(config_path)
        context_input = _build_context_input(config, base_dir=base_dir)
        tool_registry = build_tool_registry(config.tools)
        tool_view = tool_registry.view()
        permission_policy = DefaultPermissionPolicy(
            allow_writes=config.permissions.writes == "allow"
        )
        tool_executor = ToolExecutor(
            tool_registry,
            permission_policy=permission_policy,
        )
        resolved_session_store = session_store or _build_session_store(
            config,
            base_dir=base_dir,
        )
        resolved_provider = provider or create_provider_client(
            config.to_model_route(),
            api_key=api_key,
            base_url=config.model.base_url,
            timeout=config.model.timeout,
            http_client=http_client,
        )
        workspace_root = _resolve_relative_to_base(
            config.permissions.workspace,
            base_dir=base_dir,
        )

        return AgentRuntime(
            agent_config=config,
            context_input=context_input,
            provider=resolved_provider,
            session_store=resolved_session_store,
            tool_registry=tool_registry,
            tool_view=tool_view,
            tool_executor=tool_executor,
            workspace_root=workspace_root,
            permission_policy=permission_policy,
            memory_service=memory_service,
        )


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


def _build_session_store(config: AgentConfig, *, base_dir: Path) -> SessionStore:
    """根据配置创建 session store。"""
    if config.session.backend == "none":
        return InMemorySessionStore()
    session_path = config.session.path or ".iris/session.db"
    return SQLiteSessionStore(
        _resolve_relative_to_base(session_path, base_dir=base_dir)
    )


def _resolve_relative_to_base(path: str | Path, *, base_dir: Path) -> Path:
    """按配置基准目录解析路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


__all__ = ["RuntimeFactory"]
