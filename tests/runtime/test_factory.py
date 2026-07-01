from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.message import LLMResponse, TextBlock
from iris.runtime import AgentRuntime, RuntimeFactory
from iris.runtime.models import RuntimeStatus
from iris.session import InMemorySessionStore, SQLiteSessionStore
from iris.tools import DefaultPermissionPolicy, ToolCapability, ToolExecutionContext


def _assistant_response(text: str = "来自 factory。") -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
    )


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content.strip(), encoding="utf-8")
    return path


def test_from_config_path_loads_agent_yaml_and_context_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    context_path = _write_yaml(
        config_dir / "context.yaml",
        """
system:
  slots:
    - name: instructions
      content: 来自 context.yaml
before_current_input:
  slots:
    - name: runtime_state
      content: 来自 before_current_input
""",
    )
    agent_path = _write_yaml(
        config_dir / "agent.yaml",
        """
name: context-agent
model: openai/gpt-4o-mini
context:
  path: context.yaml
session:
  backend: none
""",
    )
    provider = FakeProvider([_assistant_response()])

    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)

    assert isinstance(runtime, AgentRuntime)
    assert runtime.agent_config.name == "context-agent"
    assert runtime.context_input.system.slots[0].content == "来自 context.yaml"
    assert runtime.context_input.before_current_input is not None
    assert runtime.agent_config.context is not None
    assert runtime.agent_config.context.path == context_path.resolve()


def test_from_config_with_config_path_loads_relative_context_yaml(
    tmp_path: Path,
) -> None:
    context_path = _write_yaml(
        tmp_path / "context.yaml",
        """
system:
  slots:
    - name: instructions
      content: SDK context
""",
    )
    config = AgentConfig(
        name="sdk-context-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        context={"path": "context.yaml"},
    )

    runtime = RuntimeFactory.from_config(
        config,
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([_assistant_response()]),
    )

    assert runtime.context_input.system.slots[0].content == "SDK context"
    assert context_path.exists()


@pytest.mark.asyncio
async def test_from_config_path_uses_injected_provider_without_real_api_key(
    tmp_path: Path,
) -> None:
    agent_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: simple-agent
model: openai/gpt-4o-mini
system: 你是本地助手。
session:
  backend: none
""",
    )
    provider = FakeProvider([_assistant_response("已收到。")])

    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_turn("你好")

    assert result.status == RuntimeStatus.OK
    assert result.assistant_message is not None
    assert result.assistant_message.text == "已收到。"
    assert len(provider.requests) == 1


def test_backend_none_uses_in_memory_session_store_and_creates_no_db(
    tmp_path: Path,
) -> None:
    agent_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: memory-agent
model: openai/gpt-4o-mini
system: 你是本地助手。
session:
  backend: none
""",
    )

    runtime = RuntimeFactory.from_config_path(
        agent_path,
        provider=FakeProvider([_assistant_response()]),
    )

    assert isinstance(runtime.session_store, InMemorySessionStore)
    assert not (tmp_path / ".iris").exists()


def test_backend_sqlite_resolves_path_relative_to_config_directory(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    agent_path = _write_yaml(
        config_dir / "agent.yaml",
        """
name: sqlite-agent
model: openai/gpt-4o-mini
system: 你是本地助手。
session:
  backend: sqlite
  path: data/session.db
""",
    )

    runtime = RuntimeFactory.from_config_path(
        agent_path,
        provider=FakeProvider([_assistant_response()]),
    )

    assert isinstance(runtime.session_store, SQLiteSessionStore)
    assert runtime.session_store.path == config_dir / "data" / "session.db"
    assert runtime.session_store.path.exists()


def test_injected_session_store_takes_priority(tmp_path: Path) -> None:
    config = AgentConfig(
        name="sdk-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
        session={"backend": "sqlite", "path": "should-not-exist.db"},
    )
    injected_store = InMemorySessionStore()

    runtime = RuntimeFactory.from_config(
        config,
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([_assistant_response()]),
        session_store=injected_store,
    )

    assert runtime.session_store is injected_store
    assert not (tmp_path / "should-not-exist.db").exists()


def test_injected_memory_service_is_kept_for_later_memory_stage(tmp_path: Path) -> None:
    config = AgentConfig(
        name="memory-ready-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )
    memory_service = object()

    runtime = RuntimeFactory.from_config(
        config,
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([_assistant_response()]),
        memory_service=memory_service,
    )

    assert runtime.memory_service is memory_service


def test_from_config_without_config_path_resolves_workspace_relative_to_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = AgentConfig(
        name="sdk-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
        permissions={"workspace": "workspace", "writes": "allow"},
    )

    runtime = RuntimeFactory.from_config(
        config,
        provider=FakeProvider([_assistant_response()]),
    )

    assert runtime.workspace_root == (tmp_path / "workspace").resolve()


def test_permissions_allow_maps_to_writable_policy(tmp_path: Path) -> None:
    config = AgentConfig(
        name="write-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
        tools={"builtin": ["file.write"]},
        permissions={"workspace": ".", "writes": "allow"},
    )
    runtime = RuntimeFactory.from_config(
        config,
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([_assistant_response()]),
    )

    assert isinstance(runtime.permission_policy, DefaultPermissionPolicy)
    assert runtime.permission_policy.allow_writes is True

    write_tool = runtime.tool_registry.get("write_file")
    context = ToolExecutionContext(
        agent_id="write-agent",
        session_id="session-1",
        workspace_root=tmp_path,
        permissions={"mode": "allow"},
    )
    decision = runtime.permission_policy.check(write_tool, {}, context)

    assert write_tool.definition.capabilities <= {ToolCapability.WRITE}
    assert decision.allowed is True
