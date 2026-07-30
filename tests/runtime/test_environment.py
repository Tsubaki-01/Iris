from __future__ import annotations

from pathlib import Path

from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.message import LLMResponse, TextBlock
from iris.runtime import RuntimeEnvironment, ToolBridge
from iris.tools import ToolExecutor, ToolRegistry


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="environment-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
    )


def _provider() -> FakeProvider:
    return FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[TextBlock(text="完成")],
                finish_reason="stop",
            )
        ]
    )


def test_runtime_environment_defaults_are_isolated(tmp_path: Path) -> None:
    first = RuntimeEnvironment(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=_provider(),
        workspace_root=tmp_path / "first" / ".." / "workspace",
    )
    second = RuntimeEnvironment(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=_provider(),
    )

    assert first.tool_bridge is not second.tool_bridge
    assert first.tool_bridge.tool_view.registry is first.tool_bridge.tool_executor.registry
    assert first.workspace_root == (tmp_path / "workspace").resolve()
    assert first.memory_service is None
    assert not hasattr(first, "session_store")
    assert not hasattr(first, "interaction_service")


def test_runtime_environment_keeps_explicit_engine_dependencies(tmp_path: Path) -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    bridge = ToolBridge(tool_view=registry.view(), tool_executor=executor)

    environment = RuntimeEnvironment(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=_provider(),
        tool_bridge=bridge,
        workspace_root=tmp_path,
    )

    assert environment.tool_bridge is bridge
    assert environment.tool_bridge.tool_executor is executor
