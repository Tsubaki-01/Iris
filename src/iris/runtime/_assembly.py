"""Runtime 的内部 ROOT/CHILD 依赖装配与唯一边界解析。"""

# region imports
from __future__ import annotations

import logging
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..agents import AgentConfig, build_tool_registry
from ..command.config import CommandConfig
from ..command.models import CommandEnvironment, CommandMode, CommandStopSlot
from ..command.native import NativeCommandService
from ..command.service import CommandBinding, CommandService
from ..config import get_config
from ..context import (
    ContextBuilder,
    ContextBuildInput,
    ContextSection,
    ContextSlot,
    ContextSource,
    load_context_build_input,
)
from ..decision import (
    DecisionConfig,
    DecisionEvaluator,
    build_decision_client,
    load_decision_config,
)
from ..exceptions import (
    IrisConfigError,
    IrisContextError,
    IrisSkillPathError,
    IrisTemplateError,
    IrisToolValidationError,
)
from ..goal.context import GoalContextSource
from ..goal.tools import GetGoalTool, ReportGoalTool
from ..memory.config import build_memory_service_from_config
from ..observability.provider import observe_provider
from ..observability.service import Observability
from ..prompts import PromptSnapshot, PromptSource
from ..providers import create_provider_client
from ..providers.protocols import CompletionProvider
from ..sandbox import DockerConfig
from ..skill import (
    CATALOG_SLOT_NAME,
    LoadSkillTool,
    SkillCatalog,
    SkillDiscoveryOptions,
    SkillRegistry,
    SkillScope,
    discover_skills,
)
from ..tools import DefaultPermissionPolicy, PermissionPolicy, ToolExecutor, ToolMiddleware
from ..tools.context_access import ContextAccessPort, ContextReadTool, ContextSearchTool
from ..tools.discovery import ToolSearchTool
from ..tools.permissions import MostRestrictivePermissionPolicy
from ..tools.subagent import SubagentExecutionPort, SubagentRouteTable, SubagentTool
from ..utils import TemplateRenderer
from ._extensions import build_extensions
from ._prompts import snapshot_prompts
from .environment import RuntimeEnvironment, RuntimeExecutionScope
from .runtime import AgentRuntime
from .tool_bridge import ToolBridge

if TYPE_CHECKING:
    from ..goal.service import GoalService
    from ..hooks import HookRegistration
    from ..memory import MemoryService
# endregion

logger = logging.getLogger(__name__)
_COMMAND_TOOL_KEYS = frozenset({"exec.command", "exec.python"})


@dataclass(frozen=True, slots=True)
class RuntimeAssemblyBoundary:
    """唯一解析的 workspace/policy 和 root 共享执行依赖。"""

    workspace_root: Path
    permission_policy: PermissionPolicy
    workspace_writable: bool
    command_binding: CommandBinding
    command_stop_slots: dict[tuple[str, str], CommandStopSlot]


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
    if parent_boundary is not None and "command" in config.model_fields_set:
        raise IrisConfigError("child 不能显式声明 command，必须继承 root 执行配置")
    workspace = _resolve_relative_to_base(
        config.permissions.workspace, base_dir=_base_dir(config_path)
    ).resolve()
    writable = config.permissions.writes != "deny"
    if parent_boundary is None:
        policy = (
            permission_policy
            if permission_policy is not None
            else DefaultPermissionPolicy(
                write_mode=config.permissions.writes, execute_mode=config.permissions.execute
            )
        )
        binding = _create_command_binding(config.command, workspace, writable=writable)
        slots: dict[tuple[str, str], CommandStopSlot] = {}
    else:
        parent_root = parent_boundary.workspace_root
        if workspace.is_relative_to(parent_root):
            pass
        elif parent_root.is_relative_to(workspace):
            workspace = parent_root
        else:
            raise IrisConfigError(
                "parent 与 child workspace 不相交",
                parent_workspace=str(parent_root),
                child_workspace=str(workspace),
            )
        writable = writable and parent_boundary.workspace_writable
        policy = MostRestrictivePermissionPolicy(
            parent_boundary.permission_policy,
            DefaultPermissionPolicy(
                write_mode=config.permissions.writes, execute_mode=config.permissions.execute
            ),
        )
        binding = parent_boundary.command_binding
        slots = parent_boundary.command_stop_slots
    if (
        binding.config.mode is CommandMode.NATIVE
        and not writable
        and _COMMAND_TOOL_KEYS.intersection(config.tools.builtin)
    ):
        raise IrisConfigError("Native 只读 workspace 不能注册 exec.command/exec.python")
    return RuntimeAssemblyBoundary(workspace, policy, writable, binding, slots)


def _create_command_binding(
    config: CommandConfig, workspace_root: Path, *, writable: bool
) -> CommandBinding:
    """构造 root 的轻量执行 owner，不连接 Docker 或启动命令。"""
    host_os = platform.system()
    service: CommandService
    if config.mode is CommandMode.NATIVE:
        service = NativeCommandService(workspace_root)
        command_os = host_os
        command_shell = "cmd.exe" if host_os == "Windows" else "/bin/sh"
    else:
        from ..command.docker import DockerCommandService

        service = DockerCommandService(
            workspace_root, cast(DockerConfig, config.docker), workspace_writable=writable
        )
        command_os = "Linux"
        command_shell = "/bin/sh"
    return CommandBinding(
        config=config,
        service=service,
        environment=CommandEnvironment(
            host_os=host_os,
            mode=config.mode,
            command_os=command_os,
            command_shell=command_shell,
        ),
    )


def assemble_runtime(
    config: AgentConfig,
    *,
    config_path: Path | None,
    provider: CompletionProvider | None,
    memory_service: MemoryService | None,
    api_key: str | None,
    execution_scope: RuntimeExecutionScope,
    boundary: RuntimeAssemblyBoundary,
    subagent: SubagentAssembly | None = None,
    context_access: ContextAccessPort | None = None,
    context_source: ContextSource | None = None,
    goal_service: GoalService | None = None,
    hooks: Sequence[HookRegistration] = (),
    tool_middlewares: Sequence[ToolMiddleware] = (),
    decision_client: DecisionEvaluator | None = None,
    prompt_source: PromptSource | None = None,
    observability: Observability | None = None,
) -> AgentRuntime:
    """消费已解析边界装配 inner engine 和可选服务，不创建 lifecycle store。"""
    if config.goal.enabled:
        if execution_scope is RuntimeExecutionScope.CHILD:
            raise IrisConfigError("child 不能启用 goal，Goal 仅由 root AgentRunner 管理")
        if goal_service is None:
            raise IrisConfigError("启用 goal 需要 AgentRunner 注入 GoalService")
    else:
        goal_service = None
    if not config.context_policy.enabled and context_source is not None:
        raise IrisConfigError("context_policy 禁用时不能注入 context_source")
    if config.context_policy.enabled and context_access is None:
        raise IrisConfigError(
            "启用 context_policy 需要注入 context_access；完整运行请使用 AgentRunner"
        )
    # 工厂只构造扩展对象；在 provider、memory、MCP 持有资源前统一失败。
    hook_dispatcher, middlewares = build_extensions(
        config,
        hooks=hooks,
        tool_middlewares=tool_middlewares,
        command_binding=boundary.command_binding,
        workspace_root=boundary.workspace_root,
    )
    base_dir = _base_dir(config_path)
    decision_config = (
        load_decision_config(_resolve_relative_to_base(config.decision.path, base_dir=base_dir))
        if config.decision is not None
        else DecisionConfig()
    )
    if decision_config.tools.discovery and not config.context_policy.deferred_tools:
        raise IrisConfigError("Decision tools.discovery 要求 context_policy.deferred_tools=true")
    if decision_config.memory.recall and not config.memory.enabled:
        raise IrisConfigError("Decision memory.recall 要求 memory.enabled=true")
    workspace_root = boundary.workspace_root
    if prompt_source is None:
        prompt_source = PromptSource.initialize(workspace_root, config.prompts.root)
    prompt_snapshot = snapshot_prompts(prompt_source)
    raw_provider = (
        create_provider_client(
            config.to_model_route(),
            api_key=api_key,
            api_style=config.model.api_style,
            base_url=config.model.base_url,
            timeout=config.model.timeout,
        )
        if provider is None
        else provider
    )
    owned_observability = None
    if observability is None:
        if config.observability.enabled:
            observability = Observability.from_config(
                config.observability, get_config().observability
            )
            owned_observability = observability
        else:
            observability = Observability()
    try:
        memory_service = build_memory_service_from_config(
            config.memory,
            workspace_root,
            memory_service=memory_service,
            overview_provider=raw_provider,
            overview_model=config.model.name,
            observability=observability,
            prompt_source=prompt_source,
        )
        context_input = _build_context_input(config, base_dir=base_dir)
        try:
            context_renderer = TemplateRenderer.freeze_directories(
                section.template.parent
                for section in (
                    context_input.system,
                    context_input.memory,
                    context_input.before_current_input,
                )
                if section is not None and section.template is not None
            )
        except IrisTemplateError as exc:
            raise IrisContextError(exc.message, **exc.context) from exc
        context_input, skill_registry = _prepare_skills(
            context_input,
            config=config,
            workspace_root=workspace_root,
            prompt_snapshot=prompt_snapshot,
        )
        mcp_config = None
        if config.mcp is not None:
            from ..mcp.config import load_mcp_config

            mcp_config = load_mcp_config(
                _resolve_relative_to_base(config.mcp.path, base_dir=base_dir),
                overrides=config.mcp.overrides,
            )
        decision_client, owned_decision_client = build_decision_client(
            decision_config, decision_client=decision_client
        )
        tool_registry = build_tool_registry(
            config.tools,
            memory_service=memory_service,
            memory_config=config.memory,
            memory_decision_client=decision_client if decision_config.memory.recall else None,
            command_binding=boundary.command_binding,
            prompt_snapshot=prompt_snapshot,
        )
        if config.context_policy.enabled:
            access = cast(ContextAccessPort, context_access)
            try:
                tool_registry.register(ContextReadTool(access))
                tool_registry.register(ContextSearchTool(access))
            except IrisToolValidationError as exc:
                raise IrisConfigError("context_read/search 与现有工具名称或别名冲突") from exc
        if goal_service is not None:
            try:
                tool_registry.register_many(
                    (GetGoalTool(goal_service), ReportGoalTool(goal_service))
                )
            except IrisToolValidationError as exc:
                raise IrisConfigError("get_goal/report_goal 与现有工具名称或别名冲突") from exc
            context_source = GoalContextSource(
                goal_service, host_source=context_source, prompt_snapshot=prompt_snapshot
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
        if mcp_config is not None:
            from ..mcp.manager import MCPManager

            mcp_manager = MCPManager(
                mcp_config,
                registry=tool_registry,
                workspace_root=workspace_root,
                defer_tools=config.context_policy.deferred_tools,
            )
        tool_view = tool_registry.view()
        if config.context_policy.deferred_tools:
            try:
                tool_registry.register(
                    ToolSearchTool(
                        tool_view,
                        decision_client=decision_client
                        if decision_config.tools.discovery
                        else None,
                        prompt_snapshot=prompt_snapshot,
                    )
                )
            except IrisToolValidationError as exc:
                raise IrisConfigError("tool_search 与现有工具名称或别名冲突") from exc
        tool_executor = ToolExecutor(
            tool_registry,
            permission_policy=boundary.permission_policy,
            middleware=middlewares,
        )
        tool_bridge = ToolBridge(
            tool_view=tool_view,
            tool_executor=tool_executor,
        )
        environment = RuntimeEnvironment(
            agent_config=config,
            context_input=context_input,
            context_builder=ContextBuilder(template_renderer=context_renderer),
            provider=observe_provider(raw_provider, observability),
            prompt_source=prompt_source,
            prompt_snapshot=prompt_snapshot,
            tool_bridge=tool_bridge,
            workspace_root=workspace_root,
            memory_service=memory_service,
            goal_service=goal_service,
            skill_registry=skill_registry,
            mcp_manager=mcp_manager,
            execution_scope=execution_scope,
            command_binding=boundary.command_binding,
            command_environment=(
                boundary.command_binding.environment
                if _COMMAND_TOOL_KEYS.intersection(config.tools.builtin)
                else None
            ),
            host_os=boundary.command_binding.environment.host_os,
            command_stop_slots=boundary.command_stop_slots,
            context_source=context_source,
            hook_dispatcher=hook_dispatcher,
            decision_client=decision_client,
            owned_decision_client=owned_decision_client,
            observability=observability,
            owned_observability=owned_observability,
        )
        return AgentRuntime(environment)
    except BaseException:
        if owned_observability is not None:
            owned_observability._shutdown()
        raise


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
    prompt_snapshot: PromptSnapshot,
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

    catalog = SkillCatalog(registry, prompt_snapshot=prompt_snapshot)
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
