from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisConfigError
from iris.hooks import HookEvent, HookRegistration, ToolBeforeEvent
from iris.hooks._dispatch_types import CommandHookRegistration
from iris.hooks.command import CommandHookAdapter
from iris.runtime._assembly import resolve_runtime_boundary
from iris.runtime._extensions import build_extensions
from iris.tools import ToolCall, ToolMiddleware, ToolNext, ToolResult


class Passthrough(ToolMiddleware):
    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        return await call_next()


def _config(**fields: object) -> AgentConfig:
    return AgentConfig.model_validate(
        {"name": "agent", "model": "openai/test", "system": "instructions"} | fields
    )


def _event() -> ToolBeforeEvent:
    return ToolBeforeEvent(
        agent_id="agent",
        session_id="session",
        workspace=".",
        call_id="call",
        tool_name="exec_command",
        arguments={},
    )


@pytest.mark.asyncio
async def test_factories_once_ordered_and_reused_with_sdk_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = ModuleType("test_hook_extensions")
    constructed: list[str] = []
    observed: list[str] = []
    middleware = Passthrough()

    def hook_factory(*, label: str) -> object:
        constructed.append(label)

        async def handler(event: HookEvent) -> None:
            observed.append(label)

        return handler

    def middleware_factory(*, label: str) -> ToolMiddleware:
        constructed.append(label)
        return middleware

    module.hook_factory = hook_factory
    module.middleware_factory = middleware_factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = _config(
        hooks=[
            {
                "name": name,
                "event": "tool.before",
                "tools": ["exec_command"],
                "handler": {
                    "type": "python",
                    "factory": "test_hook_extensions:hook_factory",
                    "options": {"label": name},
                },
            }
            for name in ["first", "second"]
        ],
        middleware={
            "tools": [
                {
                    "factory": "test_hook_extensions:middleware_factory",
                    "options": {"label": "middleware"},
                }
            ]
        },
    )
    sdk = HookRegistration(event="tool.before", name="sdk", handler=hook_factory(label="sdk"))
    sdk_middleware = Passthrough()
    boundary = resolve_runtime_boundary(config, config_path=tmp_path / "agent.yaml")

    dispatcher, middlewares = build_extensions(
        config,
        hooks=[sdk],
        tool_middlewares=[sdk_middleware],
        command_binding=boundary.command_binding,
        workspace_root=boundary.workspace_root,
    )

    assert constructed == ["sdk", "first", "second", "middleware"]
    assert observed == []
    assert middlewares == (middleware, sdk_middleware)
    assert dispatcher is not None
    assert dispatcher.registrations[-1] is sdk
    await dispatcher.dispatch(_event())
    await dispatcher.dispatch(_event())
    assert observed == ["first", "second", "sdk"] * 2
    assert constructed == ["sdk", "first", "second", "middleware"]


def test_command_registration_binds_exact_service_workspace_and_hook_timeout(
    tmp_path: Path,
) -> None:
    config = _config(
        hooks=[
            {
                "name": "command",
                "event": "tool.after",
                "timeout_seconds": 3,
                "handler": {"type": "command", "command": "python feedback.py"},
            }
        ]
    )
    boundary = resolve_runtime_boundary(config, config_path=tmp_path / "agent.yaml")

    dispatcher, middlewares = build_extensions(
        config,
        command_binding=boundary.command_binding,
        workspace_root=boundary.workspace_root,
    )

    assert dispatcher is not None
    registration = dispatcher.registrations[0]
    assert isinstance(registration, CommandHookRegistration)
    assert isinstance(registration.handler, CommandHookAdapter)
    assert registration.handler.binding is boundary.command_binding
    assert registration.handler.workspace == boundary.workspace_root
    assert registration.handler.timeout_seconds == 3
    assert middlewares == ()


@pytest.mark.parametrize(
    "target",
    [
        "missing:factory",
        "test_bad_hook:missing",
        "bad-ref",
        "test_bad_hook:not_callable",
        "test_bad_hook:raises",
        "test_bad_hook:not_handler",
    ],
)
def test_factory_failures_are_configuration_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    module = ModuleType("test_bad_hook")
    module.not_callable = 7
    module.not_handler = lambda: 7

    def fails() -> None:
        raise RuntimeError("factory failed")

    module.raises = fails
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = _config(
        hooks=[
            {
                "name": "bad",
                "event": "tool.before",
                "handler": {"type": "python", "factory": target},
            }
        ]
    )
    boundary = resolve_runtime_boundary(config, config_path=tmp_path / "agent.yaml")

    with pytest.raises(IrisConfigError):
        build_extensions(
            config, command_binding=boundary.command_binding, workspace_root=boundary.workspace_root
        )


@pytest.mark.asyncio
async def test_sync_callable_is_not_probed_until_real_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = ModuleType("test_sync_hook")
    called: list[str] = []

    def handler(event: HookEvent) -> None:
        called.append(event.event)

    module.factory = lambda: handler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = _config(
        hooks=[
            {
                "name": "sync",
                "event": "tool.before",
                "handler": {"type": "python", "factory": "test_sync_hook:factory"},
            }
        ]
    )
    boundary = resolve_runtime_boundary(config, config_path=tmp_path / "agent.yaml")
    dispatcher, _ = build_extensions(
        config, command_binding=boundary.command_binding, workspace_root=boundary.workspace_root
    )
    assert called == []
    assert dispatcher is not None
    outcome = await dispatcher.dispatch(_event())
    assert called == ["tool.before"]
    assert outcome.rejection is not None
    assert outcome.rejection.code == "HOOK_ERROR"


def test_middleware_factory_requires_current_abc_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = ModuleType("test_bad_middleware")
    module.factory = lambda: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = _config(middleware={"tools": [{"factory": "test_bad_middleware:factory"}]})
    boundary = resolve_runtime_boundary(config, config_path=tmp_path / "agent.yaml")
    with pytest.raises(IrisConfigError, match="ToolMiddleware"):
        build_extensions(
            config, command_binding=boundary.command_binding, workspace_root=boundary.workspace_root
        )


def test_empty_configuration_keeps_no_dispatcher(tmp_path: Path) -> None:
    config = _config()
    boundary = resolve_runtime_boundary(config, config_path=tmp_path / "agent.yaml")
    assert build_extensions(
        config, command_binding=boundary.command_binding, workspace_root=boundary.workspace_root
    ) == (None, ())
