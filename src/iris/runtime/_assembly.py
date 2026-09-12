"""Runtime 的内部 ROOT/CHILD 依赖装配与唯一边界解析。"""

# region imports
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agents import AgentConfig, build_tool_registry
from ..context import (
    ContextBuildInput,
    ContextSection,
    ContextSlot,
    load_context_build_input,
)
from ..exceptions import IrisConfigError, IrisSkillPathError, IrisToolValidationError
from ..providers import create_provider_client
from ..skill import (
    CATALOG_SLOT_NAME,
    LoadSkillTool,
    SkillCatalog,
    SkillDiscoveryOptions,
    SkillRegistry,
    SkillScope,
    discover_skills,
)
from ..tools import DefaultPermissionPolicy, PermissionPolicy, ToolExecutor
from ..tools.permissions import MostRestrictivePermissionPolicy
from ..tools.subagent import SubagentExecutionPort, SubagentRouteTable, SubagentTool
from .environment import RuntimeEnvironment, RuntimeProvider
from .runtime import AgentRuntime
from .tool_bridge import ToolBridge

if TYPE_CHECKING:
    from ..memory import MemoryService
# endregion

logger = logging.getLogger(__name__)


class RuntimeExecutionScope(StrEnum):
    """内部装配范围，CHILD 不注册递归委派工具。"""

    ROOT = "root"
    CHILD = "child"


@dataclass(frozen=True, slots=True)
class RuntimeAssemblyBoundary:
    """唯一解析后的 effective workspace 与实际权限策略。"""

    workspace_root: Path
    permission_policy: PermissionPolicy


@dataclass(frozen=True, slots=True)
class SubagentAssembly:
    """将路由与执行 port 作为一个完整装配依赖传递。"""

    routes: SubagentRouteTable
    port: SubagentExecutionPort


def resolve_runtime_boundary(
    config: AgentConfig,
    *,
    config_path: Path | None = None,
    permission_policy: PermissionPolicy | None = None,
    parent_boundary: RuntimeAssemblyBoundary | None = None,
) -> RuntimeAssemblyBoundary:
    """解析 ROOT 边界，或将 CHILD 的 workspace/policy 收窄到父边界。"""
    workspace = _resolve_relative_to_base(
        config.permissions.workspace, base_dir=_base_dir(config_path)
    ).resolve()
    if parent_boundary is None:
        policy = (
            permission_policy
            if permission_policy is not None
            else DefaultPermissionPolicy(write_mode=config.permissions.writes)
        )
        return RuntimeAssemblyBoundary(workspace, policy)
    parent_root = parent_boundary.workspace_root
    if workspace.is_relative_to(parent_root):
        effective_root = workspace
    elif parent_root.is_relative_to(workspace):
        effective_root = parent_root
    else:
        raise IrisConfigError(
            "parent 与 child workspace 不相交",
            parent_workspace=str(parent_root),
            child_workspace=str(workspace),
        )
    return RuntimeAssemblyBoundary(
        effective_root,
        MostRestrictivePermissionPolicy(
            parent_boundary.permission_policy,
            DefaultPermissionPolicy(write_mode=config.permissions.writes),
        ),
    )


def assemble_runtime(
    config: AgentConfig,
    *,
    config_path: Path | None,
    provider: RuntimeProvider | None,
    memory_service: MemoryService | None,
    api_key: str | None,
    execution_scope: RuntimeExecutionScope,
    boundary: RuntimeAssemblyBoundary,
    subagent: SubagentAssembly | None = None,
) -> AgentRuntime:
    """消费已解析边界装配 inner engine，不加载 catalog 或创建 store。"""
    base_dir = _base_dir(config_path)
    workspace_root = boundary.workspace_root
    context_input = _build_context_input(config, base_dir=base_dir)
    tool_registry = build_tool_registry(config.tools)
    context_input, skill_registry = _prepare_skills(
        context_input,
        config=config,
        workspace_root=workspace_root,
    )
    if skill_registry is not None:
        try:
            tool_registry.register(LoadSkillTool(skill_registry))
        except IrisToolValidationError as exc:
            raise IrisConfigError(
                "load_skill 与现有工具名称或别名冲突",
                tool="load_skill",
            ) from exc
    if execution_scope is RuntimeExecutionScope.ROOT and subagent is not None:
        try:
            tool_registry.register(SubagentTool(routes=subagent.routes, port=subagent.port))
        except IrisToolValidationError as exc:
            raise IrisConfigError("subagent 与现有工具名称或别名冲突", tool="subagent") from exc
    mcp_manager = None
    if config.mcp is not None:
        from ..mcp.config import load_mcp_config
        from ..mcp.manager import MCPManager

        mcp_manager = MCPManager(
            load_mcp_config(
                _resolve_relative_to_base(config.mcp.path, base_dir=base_dir),
                overrides=config.mcp.overrides,
            ),
            registry=tool_registry,
            workspace_root=workspace_root,
        )
    tool_view = tool_registry.view()
    tool_executor = ToolExecutor(
        tool_registry,
        permission_policy=boundary.permission_policy,
    )
    provider_fingerprint: dict[str, Any] = {}
    resolved_provider: RuntimeProvider
    if provider is None:
        client = create_provider_client(
            config.to_model_route(),
            api_key=api_key,
            base_url=config.model.base_url,
            timeout=config.model.timeout,
        )
        resolved_provider = client
        provider_fingerprint = {
            "provider": client.provider,
            "litellm_provider": client.litellm_provider or client.provider,
            "base_url": client.base_url,
            "headers": dict(client.headers),
        }
    else:
        resolved_provider = provider

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
        skill_registry=skill_registry,
        provider_fingerprint=provider_fingerprint,
        mcp_manager=mcp_manager,
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


def _prepare_skills(
    context_input: ContextBuildInput,
    *,
    config: AgentConfig,
    workspace_root: Path,
) -> tuple[ContextBuildInput, SkillRegistry | None]:
    """发现项目级 Skill，并为非空 registry 追加 catalog slot。"""
    skills_config = config.skills
    if skills_config is None or not skills_config.enabled:
        return context_input, None

    try:
        result = discover_skills(
            SkillDiscoveryOptions(
                workspace_root=workspace_root,
                roots=((SkillScope.PROJECT, Path(skills_config.root)),),
            )
        )
    except IrisSkillPathError as exc:
        raise IrisConfigError(
            "skills.root 不在 workspace 内",
            root=skills_config.root,
            workspace_root=str(workspace_root.resolve()),
        ) from exc
    registry = SkillRegistry(result)
    for diagnostic in registry.diagnostics:
        logger.warning(
            "skill discovery diagnostic %s: %s",
            diagnostic.code,
            diagnostic.message,
            extra={
                "code": diagnostic.code,
                "path": str(diagnostic.path) if diagnostic.path is not None else None,
                "detail": diagnostic.detail,
            },
        )

    missing = registry.missing(skills_config.require)
    if missing:
        raise IrisConfigError(
            "required skills 不存在",
            missing=missing,
            available=registry.names(),
        )
    if len(registry) == 0:
        return context_input, None

    if context_input.system.template is not None:
        logger.warning(
            "custom system template must explicitly consume the skill catalog slot",
            extra={
                "code": "TEMPLATE_SECTION",
                "section": "system",
                "template": str(context_input.system.template),
                "slot": CATALOG_SLOT_NAME,
            },
        )

    catalog = SkillCatalog(registry)
    content_chars = catalog.content_chars()
    logger.info(
        "skill catalog built",
        extra={"count": len(registry), "content_chars": content_chars},
    )
    system = context_input.system.model_copy(
        update={"slots": [*context_input.system.slots, catalog.build_slot()]},
    )
    return context_input.model_copy(update={"system": system}), registry


def _resolve_relative_to_base(path: str | Path, *, base_dir: Path) -> Path:
    """按配置基准目录解析路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()
