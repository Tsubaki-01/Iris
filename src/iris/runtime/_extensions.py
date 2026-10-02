"""在资源准备之前构造当前 Agent 的有序扩展实例。"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ..agents.config._imports import import_ref
from ..agents.config.base import AgentConfig
from ..agents.config.hooks import PythonHookHandlerConfig
from ..command.service import CommandBinding
from ..exceptions import IrisConfigError
from ..hooks._dispatch_types import CommandHookRegistration
from ..hooks.command import CommandHookAdapter
from ..hooks.dispatcher import HookDispatcher
from ..hooks.models import HookHandler, HookRegistration
from ..tools.middleware import ToolMiddleware


def build_extensions(
    config: AgentConfig,
    *,
    hooks: Sequence[HookRegistration] = (),
    tool_middlewares: Sequence[ToolMiddleware] = (),
    command_binding: CommandBinding,
    workspace_root: Path,
) -> tuple[HookDispatcher | None, tuple[ToolMiddleware, ...]]:
    """每个 YAML 工厂调用一次，再追加已构造的 SDK 项；不继承父 Agent 扩展。"""
    registrations: list[HookRegistration | CommandHookRegistration] = []
    for hook in config.hooks:
        if isinstance(hook.handler, PythonHookHandlerConfig):
            handler = _create_extension(hook.handler.factory, hook.handler.options)
            if not callable(handler):
                raise IrisConfigError(
                    "Hook 工厂必须返回可调用处理器", name=hook.name, ref=hook.handler.factory
                )
            registrations.append(
                HookRegistration.model_construct(
                    event=hook.event,
                    name=hook.name,
                    handler=cast(HookHandler, handler),
                    tool_names=hook.tools,
                    timeout_seconds=hook.timeout_seconds,
                )
            )
        else:
            registrations.append(
                CommandHookRegistration(
                    event=hook.event,
                    name=hook.name,
                    handler=CommandHookAdapter(
                        binding=command_binding,
                        workspace=workspace_root,
                        command=hook.handler.command,
                        timeout_seconds=hook.timeout_seconds,
                    ),
                    tool_names=hook.tools,
                )
            )
    registrations.extend(hooks)

    middlewares: list[ToolMiddleware] = []
    for declaration in config.middleware.tools:
        middleware = _create_extension(declaration.factory, declaration.options)
        if not isinstance(middleware, ToolMiddleware):
            raise IrisConfigError(
                "Middleware 工厂必须返回 ToolMiddleware 实例", ref=declaration.factory
            )
        middlewares.append(middleware)
    middlewares.extend(tool_middlewares)
    return (HookDispatcher(registrations) if registrations else None, tuple(middlewares))


def _create_extension(ref: str, options: dict[str, Any]) -> object:
    """按唯一 factory(**options) 协议导入并构造对象，不试运行返回的处理器。"""
    try:
        factory = import_ref(ref)
        return factory(**options)
    except Exception as exc:
        raise IrisConfigError("扩展工厂导入或构造失败", ref=ref) from exc
