from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.message import LLMResponse, TextBlock
from iris.runtime import AgentRuntime, RuntimeFactory
from iris.tools import DefaultPermissionPolicy


def _response() -> LLMResponse:
    return LLMResponse(provider="fake", content=[TextBlock(text="完成")])


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content.strip(), encoding="utf-8")
    return path


def test_from_config_path_loads_relative_context_without_creating_store(
    tmp_path: Path,
) -> None:
    context_path = _write_yaml(
        tmp_path / "context.yaml",
        """
system:
  slots:
    - name: instructions
      content: 来自 context.yaml
""",
    )
    agent_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: context-agent
model: openai/gpt-4o-mini
context:
  path: context.yaml
session:
  backend: sqlite
  path: state/lifecycle.db
""",
    )

    runtime = RuntimeFactory.from_config_path(
        agent_path,
        provider=FakeProvider([_response()]),
    )

    assert isinstance(runtime, AgentRuntime)
    assert runtime.environment.context_input.system.slots[0].content == "来自 context.yaml"
    assert runtime.environment.agent_config.context is not None
    assert runtime.environment.agent_config.context.path == context_path.resolve()
    assert not (tmp_path / "state" / "lifecycle.db").exists()


def test_from_config_keeps_injected_engine_dependencies(tmp_path: Path) -> None:
    memory_service = object()
    provider = FakeProvider([_response()])
    config = AgentConfig(
        name="sdk-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="测试",
        permissions={"workspace": "workspace", "writes": "allow"},
    )

    runtime = RuntimeFactory.from_config(
        config,
        config_path=tmp_path / "agent.yaml",
        provider=provider,
        memory_service=memory_service,
    )

    assert runtime.environment.provider is provider
    assert runtime.environment.memory_service is memory_service
    assert runtime.environment.workspace_root == (tmp_path / "workspace").resolve()
    policy = runtime.environment.tool_bridge.tool_executor.permission_policy
    assert isinstance(policy, DefaultPermissionPolicy)
    assert policy.write_mode == "allow"


def test_from_config_rejects_removed_store_injection_keywords() -> None:
    config = AgentConfig(
        name="sdk-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="测试",
    )

    with pytest.raises(TypeError):
        RuntimeFactory.from_config(
            config,
            provider=FakeProvider([_response()]),
            session_store=object(),
        )
    with pytest.raises(TypeError):
        RuntimeFactory.from_config(
            config,
            provider=FakeProvider([_response()]),
            interaction_store=object(),
        )
