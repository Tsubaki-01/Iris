"""四个公开装配入口共用扩展实例、顺序及资源准备边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from iris.agents import AgentConfig
from iris.command.docker import DockerCommandService
from iris.exceptions import IrisCommandError, IrisConfigError
from iris.harness import AgentRunner
from iris.hooks import HookEvent, HookHandler, HookRegistration
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import ToolUseBlock
from iris.runtime import RuntimeFactory, _assembly
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolMiddleware, ToolResult
from iris.tools.middleware import ToolCall, ToolNext

from ..harness.fakes import StaticProvider, text_response, tool_response
from .fakes import FakeRuntimeCommitPort, MutableCancellationSignal, start_activation

_constructed: list[str] = []
_events: list[str] = []


class RecordingMiddleware(ToolMiddleware):
    """记录真实调用的 onion 顺序及跨 Run 实例复用。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        """只消费一次 continuation。"""
        self.calls += 1
        _events.append(f"{self.name}-enter")
        result = await call_next()
        _events.append(f"{self.name}-exit")
        return result


def make_hook(*, name: str) -> HookHandler:
    """配置引用的同步纯工厂。"""
    _constructed.append(name)

    async def handler(event: HookEvent) -> None:
        _events.append(name)

    return handler


def make_middleware(*, name: str) -> ToolMiddleware:
    """每次 assembly 创建一个 middleware。"""
    _constructed.append(name)
    return RecordingMiddleware(name)


def failing_factory() -> HookHandler:
    """在框架准备外部资源之前报告工厂失败。"""
    raise RuntimeError("factory failed")


def _payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "name": "extensions",
        "model": "openai/test",
        "system": "test",
        "permissions": {"workspace": str(tmp_path)},
        "context_policy": {"enabled": False},
        "memory": {"enabled": False},
        "tools": {"builtin": ["file.read"]},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["runner", "factory"])
@pytest.mark.parametrize("from_path", [False, True])
async def test_four_public_entries_merge_yaml_then_sdk_once(
    tmp_path: Path, owner: str, from_path: bool
) -> None:
    _constructed.clear()
    _events.clear()
    payload = _payload(tmp_path)
    payload.update(
        hooks=[
            {
                "name": "yaml",
                "event": "tool.before",
                "tools": ["read_file"],
                "handler": {
                    "type": "python",
                    "factory": f"{__name__}:make_hook",
                    "options": {"name": "yaml-hook"},
                },
            }
        ],
        middleware={
            "tools": [
                {"factory": f"{__name__}:make_middleware", "options": {"name": "yaml-middleware"}}
            ]
        },
    )
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    (tmp_path / "payload.txt").write_text("body", encoding="utf-8")
    provider = StaticProvider(
        *[
            response
            for index in range(2)
            for response in (
                tool_response(
                    ToolUseBlock(
                        id=f"read-{index}", name="read_file", input={"file_path": "payload.txt"}
                    )
                ),
                text_response(),
            )
        ]
    )

    async def sdk_hook(event: HookEvent) -> None:
        _events.append("sdk-hook")

    hook = HookRegistration(event="tool.before", name="sdk", handler=sdk_hook)
    middleware = RecordingMiddleware("sdk-middleware")
    entry = AgentRunner if owner == "runner" else RuntimeFactory
    kwargs = {"provider": provider, "hooks": [hook], "tool_middlewares": [middleware]}
    built = (
        entry.from_config_path(path, **kwargs)
        if from_path
        else entry.from_config(AgentConfig.model_validate(payload), **kwargs)
    )
    runner = (
        built
        if isinstance(built, AgentRunner)
        else AgentRunner(runtime=built, store=InMemoryLifecycleStore())
    )
    environment = runner.runtime.environment
    dispatcher = environment.hook_dispatcher
    executor = environment.tool_bridge.tool_executor
    assert dispatcher is not None and executor.hook_dispatcher is dispatcher
    assert dispatcher.registrations[-1] is hook
    assert executor.middleware[-1] is middleware
    assert _constructed == ["yaml-hook", "yaml-middleware"]
    try:
        for index in range(2):
            result = await runner.start(AgentRunRequest(input="read", session_id=str(index)))
            assert result.run.stop_reason is RunStopReason.COMPLETED
    finally:
        await runner.aclose()
    assert _constructed == ["yaml-hook", "yaml-middleware"]
    assert middleware.calls == 2
    assert (
        _events
        == [
            "yaml-hook",
            "sdk-hook",
            "yaml-middleware-enter",
            "sdk-middleware-enter",
            "sdk-middleware-exit",
            "yaml-middleware-exit",
        ]
        * 2
    )


@pytest.mark.parametrize("owner", [AgentRunner, RuntimeFactory])
def test_factory_failure_precedes_framework_resource_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: type[AgentRunner] | type[RuntimeFactory]
) -> None:
    payload = _payload(tmp_path)
    payload["hooks"] = [
        {
            "name": "fails",
            "event": "run.started",
            "handler": {"type": "python", "factory": f"{__name__}:failing_factory"},
        }
    ]
    reached: list[str] = []

    def resource(*args: Any, **kwargs: Any) -> None:
        reached.append("resource")

    monkeypatch.setattr(_assembly, "create_provider_client", resource)
    monkeypatch.setattr(_assembly, "build_memory_service_from_config", resource)
    monkeypatch.setattr(_assembly, "build_decision_client", resource)
    monkeypatch.setattr(_assembly, "build_tool_registry", resource)
    with pytest.raises(IrisConfigError, match="factory"):
        owner.from_config(AgentConfig.model_validate(payload))
    assert reached == []


@pytest.mark.asyncio
async def test_prepare_failure_keeps_original_error_closes_resources_and_runs_no_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    error = IrisCommandError("prepare failed")

    async def prepare(service: DockerCommandService) -> None:
        events.append("prepare")
        raise error

    async def close(service: DockerCommandService) -> None:
        events.append("close")

    async def hook(event: HookEvent) -> None:
        events.append(event.event)

    monkeypatch.setattr(DockerCommandService, "prepare", prepare)
    monkeypatch.setattr(DockerCommandService, "aclose", close)
    payload = _payload(tmp_path)
    payload["command"] = {"mode": "docker"}
    runner = AgentRunner.from_config(
        AgentConfig.model_validate(payload),
        provider=StaticProvider(),
        hooks=[HookRegistration(event="run.started", name="start", handler=hook)],
    )
    with pytest.raises(IrisCommandError) as caught:
        await runner.start(AgentRunRequest(input="input", run_id="run"))
    assert caught.value is error
    assert runner.store.load_run("run") is None
    assert events == ["prepare", "close"]


def test_empty_extensions_do_not_create_dispatcher(tmp_path: Path) -> None:
    runtime = RuntimeFactory.from_config(
        AgentConfig.model_validate(_payload(tmp_path)), provider=StaticProvider()
    )
    assert runtime.environment.hook_dispatcher is None
    assert runtime.environment.tool_bridge.tool_executor.middleware == []


@pytest.mark.asyncio
async def test_factory_run_handlers_require_a_real_runner(tmp_path: Path) -> None:
    """低层 execute 只派发工具事件；同环境交给 Runner 后才产生 Run 事件。"""
    (tmp_path / "payload.txt").write_text("body", encoding="utf-8")
    provider = StaticProvider(
        *[
            response
            for index in range(2)
            for response in (
                tool_response(
                    ToolUseBlock(
                        id=f"read-{index}", name="read_file", input={"file_path": "payload.txt"}
                    )
                ),
                text_response(),
            )
        ]
    )
    events: list[str] = []

    async def record(event: HookEvent) -> None:
        events.append(event.event)

    runtime = RuntimeFactory.from_config(
        AgentConfig.model_validate(_payload(tmp_path)),
        provider=provider,
        hooks=[
            HookRegistration(event=event, name=event, handler=record)
            for event in ("run.started", "tool.before", "run.finished")
        ],
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    await runtime.environment.aprepare()
    await runtime.execute(activation, commits=commits, cancellation=MutableCancellationSignal())
    assert events == ["tool.before"]
    runner = AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())
    try:
        result = await runner.start(AgentRunRequest(input="read"))
        assert result.run.stop_reason is RunStopReason.COMPLETED
    finally:
        await runner.aclose()
    assert events == ["tool.before", "run.started", "tool.before", "run.finished"]
