"""执行绑定由 root 拥有，child 只借用资源并保留自己的工具范围。"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.command.docker import DockerCommandService
from iris.command.models import CommandMode, CommandStopSlot
from iris.command.native import NativeCommandService
from iris.exceptions import IrisCommandError, IrisConfigError, IrisMCPError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, SessionContextWindow
from iris.runtime import RuntimeFactory
from iris.runtime._assembly import (
    RuntimeAssemblyBoundary,
    assemble_runtime,
    resolve_runtime_boundary,
)
from iris.runtime.environment import RuntimeEnvironment, RuntimeExecutionScope
from iris.store import InMemoryLifecycleStore

from ..harness.fakes import StaticProvider, text_response
from ..harness.test_runner_subagent import ChildProviders, _parent_provider, _write_configs


def _config(tmp_path: Path, **changes: object) -> AgentConfig:
    """构造无外部 IO、默认不开命令工具的最小配置。"""
    return AgentConfig.model_validate(
        {
            "name": "command-test",
            "model": "openai/test",
            "system": "test",
            "context_policy": {"enabled": False},
            "permissions": {"workspace": str(tmp_path)},
            **changes,
        }
    )


def _child(config: AgentConfig, parent: RuntimeAssemblyBoundary) -> RuntimeEnvironment:
    """经真实 private assembly 装配借用 root 资源的 child。"""
    return assemble_runtime(
        config,
        config_path=None,
        provider=StaticProvider(),
        memory_service=None,
        api_key=None,
        execution_scope=RuntimeExecutionScope.CHILD,
        boundary=resolve_runtime_boundary(config, parent_boundary=parent),
    ).environment


def test_factory_roots_own_independent_native_bindings_without_optional_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无 exec 也准备轻量 root 资源给 child，但不宣称当前有命令工具。"""
    monkeypatch.setitem(sys.modules, "aiodocker", None)
    config = _config(tmp_path)
    first = RuntimeFactory.from_config(config, provider=StaticProvider()).environment
    second = RuntimeFactory.from_config(config, provider=StaticProvider()).environment
    assert first.command_binding is not None
    assert isinstance(first.command_binding.service, NativeCommandService)
    assert first.command_binding is not second.command_binding
    assert first.command_stop_slots is not second.command_stop_slots
    assert first.tool_bridge.command_stop_slots is first.command_stop_slots
    assert first.command_environment is None
    assert first.host_os == platform.system()


@pytest.mark.parametrize(
    ("host_os", "mode", "command_os", "shell"),
    [
        ("Windows", "native", "Windows", "cmd.exe"),
        ("Linux", "native", "Linux", "/bin/sh"),
        ("Windows", "docker", "Linux", "/bin/sh"),
    ],
)
@pytest.mark.parametrize(
    "builtin,tool_name", [("exec.command", "exec_command"), ("exec.python", "run_python")]
)
def test_registered_command_uses_binding_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_os: str,
    mode: str,
    command_os: str,
    shell: str,
    builtin: str,
    tool_name: str,
) -> None:
    """命令说明使用实际选定 backend，而不是从宿主 OS 猜容器 shell。"""
    monkeypatch.setattr(platform, "system", lambda: host_os)
    runtime = RuntimeFactory.from_config(
        _config(tmp_path, command={"mode": mode}, tools={"builtin": [builtin]}),
        provider=StaticProvider(),
    )
    environment = runtime.environment
    descriptor = environment.command_environment
    assert descriptor is environment.command_binding.environment
    assert descriptor.host_os == environment.host_os == host_os
    assert descriptor.mode is CommandMode(mode)
    assert descriptor.command_os == command_os
    assert descriptor.command_shell == shell
    assert environment.tool_bridge.tool_view.get(tool_name) is not None
    system = runtime._system_addendum(SessionContextWindow())
    assert f"command_mode: {mode}" in system
    assert f"command_os: {command_os}" in system
    if builtin == "exec.python":
        description = environment.tool_bridge.tool_view.get(tool_name).definition.description
        assert ("Iris 当前进程" if mode == "native" else "镜像内的 Python") in description


@pytest.mark.parametrize("root_writes,child_writes", [("allow", "deny"), ("deny", "allow")])
@pytest.mark.parametrize("builtin", ["exec.command", "exec.python"])
def test_docker_child_borrows_root_binding_with_narrow_file_scope(
    tmp_path: Path, root_writes: str, child_writes: str, builtin: str
) -> None:
    """D21 允许只读 child 命令，root bind 的真实写属性不随 child 改变。"""
    config = _config(
        tmp_path,
        command={"mode": "docker"},
        permissions={"workspace": str(tmp_path), "writes": root_writes},
    )
    parent = resolve_runtime_boundary(config)
    child = _child(
        _config(
            tmp_path,
            permissions={"workspace": str(tmp_path / "child"), "writes": child_writes},
            tools={"builtin": [builtin]},
        ),
        parent,
    )
    assert child.workspace_root == tmp_path / "child"
    assert child.command_binding is parent.command_binding
    assert child.command_stop_slots is parent.command_stop_slots
    assert child.tool_bridge.command_stop_slots is parent.command_stop_slots
    assert child.command_environment is parent.command_binding.environment
    assert isinstance(child.command_binding.service, DockerCommandService)
    assert child.command_binding.service._workspace_root == tmp_path
    assert child.command_binding.service._sandbox._workspace_writable is (root_writes != "deny")
    assert not resolve_runtime_boundary(
        child.agent_config, parent_boundary=parent
    ).workspace_writable
    slot = CommandStopSlot()
    parent.command_stop_slots[("run", "call")] = slot
    assert child.command_stop_slots[("run", "call")] is slot


@pytest.mark.parametrize("child_scope", [False, True])
@pytest.mark.parametrize("builtin", ["exec.command", "exec.python"])
def test_native_readonly_command_configuration_is_rejected(
    tmp_path: Path, child_scope: bool, builtin: str
) -> None:
    """Native D12 在唯一有效 workspace 边界拒绝只读命令组合。"""
    parent = resolve_runtime_boundary(
        _config(tmp_path, permissions={"workspace": str(tmp_path), "writes": "deny"})
    )
    config = _config(
        tmp_path,
        tools={"builtin": [builtin]},
        permissions={"workspace": str(tmp_path), "writes": "allow" if child_scope else "deny"},
    )
    with pytest.raises(IrisConfigError, match="Native"):
        resolve_runtime_boundary(config, parent_boundary=parent if child_scope else None)


@pytest.mark.parametrize("command", [{}, {"mode": "native"}, {"mode": "docker"}])
def test_child_rejects_explicit_command_even_if_equal_to_root(
    tmp_path: Path, command: dict[str, object]
) -> None:
    """明确声明与模型默认值区分，child 不能重新选择执行配置。"""
    parent = resolve_runtime_boundary(_config(tmp_path))
    with pytest.raises(IrisConfigError, match="command"):
        resolve_runtime_boundary(_config(tmp_path, command=command), parent_boundary=parent)


@pytest.mark.asyncio
async def test_docker_without_root_command_still_prepares_and_root_alone_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """仅 child 暴露命令时仍 prepare Docker，child close 不关闭共享后端。"""
    prepared: list[DockerCommandService] = []
    closed: list[DockerCommandService] = []

    async def prepare(service: DockerCommandService) -> None:
        prepared.append(service)

    async def close(service: DockerCommandService) -> None:
        closed.append(service)

    monkeypatch.setattr(DockerCommandService, "prepare", prepare)
    monkeypatch.setattr(DockerCommandService, "aclose", close)
    config = _config(tmp_path, command={"mode": "docker"})
    boundary = resolve_runtime_boundary(config)
    runtime = assemble_runtime(
        config,
        config_path=None,
        provider=StaticProvider(),
        memory_service=None,
        api_key=None,
        execution_scope=RuntimeExecutionScope.ROOT,
        boundary=boundary,
    )
    runner = AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())
    assert not runner._prepared
    assert runner.runtime.environment.command_environment is None
    await runner.aprepare()
    assert prepared == [runner.runtime.environment.command_binding.service]
    child = _child(_config(tmp_path, tools={"builtin": ["exec.command"]}), boundary)
    assert child.command_binding is runtime.environment.command_binding
    await child.aclose()
    assert closed == []
    await runner.aclose()
    assert closed == [runner.runtime.environment.command_binding.service]


@pytest.mark.asyncio
async def test_environment_closes_command_resources_after_mcp_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP close 报错仍尝试 root 命令资源关闭，且向调用者保留原错误。"""
    from ..mcp.fixtures.runtime import MCPPeer

    peer = MCPPeer(monkeypatch)
    peer.fail_close = True
    (tmp_path / "mcp.json").write_text('{"servers":{"test":{"command":"fixture"}}}')
    environment = RuntimeFactory.from_config(
        _config(tmp_path, mcp={"path": "mcp.json"}),
        config_path=tmp_path / "agent.yaml",
        provider=StaticProvider(),
    ).environment
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    monkeypatch.setattr(environment.command_binding.service, "aclose", close)
    await environment.aprepare()
    with pytest.raises(IrisMCPError):
        await environment.aclose()
    assert closed == [True]
    assert peer.events == ["open", "list", "close"]


@pytest.mark.asyncio
async def test_native_runner_keeps_no_prepare_start_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """普通 native Agent start 不因新增轻量服务而执行后端 prepare。"""

    async def unexpected_prepare(service: NativeCommandService) -> None:
        raise AssertionError("Native start 无需显式 prepare")

    monkeypatch.setattr(NativeCommandService, "prepare", unexpected_prepare)
    runner = AgentRunner.from_config(_config(tmp_path), provider=StaticProvider(text_response()))
    assert runner._prepared
    await runner.start(AgentRunRequest(input="hello"))
    await runner.aclose()


@pytest.mark.asyncio
async def test_child_command_prepare_error_maps_to_existing_subagent_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """借用服务的 child 准备失败不关闭 root，沿既有 subagent tool 错误返回。"""
    path = _write_configs(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8") + "\ncommand:\n  mode: docker\n", encoding="utf-8"
    )
    prepares = 0
    closed: list[DockerCommandService] = []

    async def prepare(service: DockerCommandService) -> None:
        nonlocal prepares
        prepares += 1
        if prepares == 2:
            raise IrisCommandError("模拟 child prepare 失败")

    async def close(service: DockerCommandService) -> None:
        closed.append(service)

    monkeypatch.setattr(DockerCommandService, "prepare", prepare)
    monkeypatch.setattr(DockerCommandService, "aclose", close)
    runner = AgentRunner.from_config_path(
        path, provider=_parent_provider(), child_provider_factory=ChildProviders(StaticProvider())
    )
    await runner.start(AgentRunRequest(input="delegate", run_id="parent"))
    assert runner.store.load_subagent_link("parent", "delegate") is None
    assert (
        runner.store.load_tool_call("parent", "delegate").result.error.code
        == "SUBAGENT_PREPARE_ERROR"
    )
    assert closed == []
    await runner.aclose()
    assert closed == [runner.runtime.environment.command_binding.service]
