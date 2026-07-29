from __future__ import annotations

from pathlib import Path

from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.hitl import InMemoryInteractionStore
from iris.hitl._legacy_service import HumanInteractionService
from iris.message import LLMResponse, TextBlock
from iris.runtime import RuntimeEnvironment, ToolBridge
from iris.session import InMemorySessionStore
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
                id="response-1",
                model="gpt-4o-mini",
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

    assert isinstance(first.session_store, InMemorySessionStore)
    assert first.session_store is not second.session_store
    assert first.tool_bridge is not second.tool_bridge
    assert first.tool_bridge.tool_view.registry is first.tool_bridge.tool_executor.registry
    assert first.interaction_service is not second.interaction_service
    assert first.interaction_service.store is not second.interaction_service.store
    assert first.workspace_root == (tmp_path / "workspace").resolve()
    assert first.memory_service is None


def test_runtime_environment_keeps_explicit_dependency_graph(tmp_path: Path) -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    bridge = ToolBridge(tool_view=registry.view(), tool_executor=executor)
    interaction_store = InMemoryInteractionStore()
    interaction_service = HumanInteractionService(interaction_store)
    session_store = InMemorySessionStore()

    environment = RuntimeEnvironment(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=_provider(),
        session_store=session_store,
        tool_bridge=bridge,
        interaction_service=interaction_service,
        workspace_root=tmp_path,
    )

    assert environment.session_store is session_store
    assert environment.tool_bridge is bridge
    assert environment.tool_bridge.tool_executor is executor
    assert environment.interaction_service is interaction_service
    assert environment.interaction_service.store is interaction_store
