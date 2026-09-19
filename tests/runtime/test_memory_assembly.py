"""Root、CLI 与 child 共用 memory 配置装配链。"""

from pathlib import Path

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig, ToolsConfig, build_tool_registry
from iris.exceptions import IrisConfigError, IrisMemoryError, IrisToolValidationError
from iris.harness import AgentRunner
from iris.memory import MemoryConfig, MemoryService, MemoryWriteInput, SQLiteMemoryStore
from iris.runtime import RuntimeFactory
from iris.tools import ToolExecutionContext


def _config(workspace: Path, *, memory: dict[str, object] | None = None) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "name": "memory-agent",
            "model": "openai/test",
            "system": "instructions",
            "permissions": {"workspace": str(workspace)},
            "memory": {} if memory is None else memory,
        }
    )


def test_memory_disabled_creates_no_service_or_tools(tmp_path: Path) -> None:
    runtime = RuntimeFactory.from_config(_config(tmp_path), provider=FakeProvider([]))
    assert runtime.environment.memory_service is None
    assert not runtime.environment.tool_bridge.tool_view.active_tools
    assert not (tmp_path / ".iris").exists()


@pytest.mark.parametrize("recall_mode", ["on_turn", "manual"])
def test_yaml_memory_uses_effective_workspace_and_shared_tool_service(
    tmp_path: Path, recall_mode: str
) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: memory-agent\nmodel: openai/test\nsystem: instructions\n"
        "permissions:\n  workspace: project\nmemory:\n  backend: sqlite\n"
        f"  recall_mode: {recall_mode}\n  read_namespaces: [project, research]\n"
        "  write_namespace: research\n  mirror:\n    enabled: false\n",
        encoding="utf-8",
    )
    runtime = RuntimeFactory.from_config_path(path, provider=FakeProvider([]))
    service = runtime.environment.memory_service
    assert service is not None
    assert service.store.path == tmp_path / "project" / ".iris" / "memory" / "memory.db"
    assert runtime.environment.agent_config.memory.recall_mode == recall_mode
    assert not (tmp_path / ".iris").exists()
    tools = runtime.environment.tool_bridge.tool_view.active_tools
    assert {tool.name for tool in tools} == {"memory_search", "memory_list", "memory_get"}
    for tool in tools:
        assert tool.service is service
        policy = tool.access_policy_factory(ToolExecutionContext(workspace_root=tmp_path))
        assert policy.read_namespaces == ["project", "research"]
        assert policy.write_namespace == "research"


def test_explicit_service_has_priority_over_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris.runtime._assembly as assembly

    service = MemoryService(SQLiteMemoryStore(tmp_path / "explicit.db"))

    def forbidden_config_service(config: MemoryConfig, workspace_root: Path) -> None:
        raise AssertionError("显式 service 不应再构造配置 service")

    monkeypatch.setattr(assembly, "build_memory_service_from_config", forbidden_config_service)
    runtime = RuntimeFactory.from_config(
        _config(tmp_path, memory={"backend": "sqlite"}),
        provider=FakeProvider([]),
        memory_service=service,
    )
    assert runtime.environment.memory_service is service
    assert runtime.environment.tool_bridge.tool_view.get("memory_get").service is service
    assert not (tmp_path / ".iris").exists()


def test_memory_builtins_register_default_reads_once_and_only_selected_writes(
    tmp_path: Path,
) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    registry = build_tool_registry(
        ToolsConfig(builtin=["memory.search", "memory.remember", "file.read"]),
        memory_service=service,
        memory_config=MemoryConfig(write_namespace="research"),
    )
    assert [tool.name for tool in registry.view().active_tools].count("memory_search") == 1
    assert {tool.name for tool in registry.view().active_tools} == {
        "memory_search",
        "memory_list",
        "memory_get",
        "memory_remember",
        "read_file",
    }
    policy = registry.get("memory_remember").access_policy_factory(
        ToolExecutionContext(workspace_root=tmp_path)
    )
    assert policy.write_namespace == "research"


def test_memory_declaration_requires_service_and_preserves_real_name_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris.agents.config.tools as tools_module

    with pytest.raises(IrisConfigError, match="memory"):
        build_tool_registry(ToolsConfig(builtin=["memory.search"]))

    def memory_search(query: str) -> str:
        """用户自定义同名函数。"""
        return query

    monkeypatch.setattr(tools_module, "_import_ref", lambda ref: memory_search)
    with pytest.raises(IrisToolValidationError):
        build_tool_registry(
            ToolsConfig(python={"functions": ["custom:memory_search"]}),
            memory_service=MemoryService(SQLiteMemoryStore(tmp_path / "memory.db")),
        )


def test_project_memory_is_shared_across_agents_and_isolated_by_workspace(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    first = RuntimeFactory.from_config(
        _config(shared, memory={"backend": "sqlite", "mirror": {"enabled": False}}),
        provider=FakeProvider([]),
    )
    second_config = first.environment.agent_config.model_copy(update={"name": "other-agent"})
    second = RuntimeFactory.from_config(second_config, provider=FakeProvider([]))
    isolated = RuntimeFactory.from_config(
        _config(tmp_path / "isolated", memory={"backend": "sqlite", "mirror": {"enabled": False}}),
        provider=FakeProvider([]),
    )
    item = first.environment.memory_service.remember(MemoryWriteInput(text="shared", reason="test"))
    assert second.environment.memory_service.get_item(item.id, ["project"]) == item
    assert isolated.environment.memory_service.get_item(item.id, ["project"]) is None


def test_memory_initialization_error_propagates_from_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris.runtime._assembly as assembly

    def fail_memory(config: MemoryConfig, workspace_root: Path) -> None:
        raise IrisMemoryError("memory 初始化失败")

    monkeypatch.setattr(assembly, "build_memory_service_from_config", fail_memory)
    with pytest.raises(IrisMemoryError, match="初始化失败"):
        RuntimeFactory.from_config(
            _config(tmp_path, memory={"backend": "sqlite"}), provider=FakeProvider([])
        )


def test_cli_uses_the_shared_memory_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris.config as iris_config
    import iris.runtime._assembly as assembly
    from iris.cli.chat import ChatOptions, run_chat

    monkeypatch.setattr(iris_config, "_config", iris_config.Config())
    monkeypatch.setattr(
        assembly, "create_provider_client", lambda *args, **kwargs: FakeProvider([])
    )
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: cli\nmodel: openai/test\nsystem: instructions\n"
        "memory:\n  backend: sqlite\n  mirror:\n    enabled: false\n",
        encoding="utf-8",
    )
    errors: list[str] = []
    code = run_chat(
        ChatOptions(config_path=path),
        input_func=lambda prompt: "/exit",
        output_func=lambda text: None,
        error_func=errors.append,
    )
    assert code == 0
    assert errors == []
    assert (tmp_path / ".iris" / "memory" / "memory.db").exists()


@pytest.mark.parametrize("child_enabled", [False, True])
def test_child_uses_own_memory_config_and_effective_workspace(
    tmp_path: Path, child_enabled: bool
) -> None:
    workspace = tmp_path / "workspace"
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        "name: child\nmodel: openai/test\nsystem: child\n"
        "permissions:\n  workspace: .\n"
        + ("memory:\n  backend: sqlite\n  recall_mode: manual\n" if child_enabled else ""),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "default: child\nagents:\n  child:\n    path: child.yaml\n    description: child\n",
        encoding="utf-8",
    )
    parent_service = MemoryService(SQLiteMemoryStore(tmp_path / "injected-parent.db"))
    parent = AgentRunner.from_config(
        _config(workspace).model_copy(update={"tools": ToolsConfig(subagent=catalog)}),
        config_path=tmp_path / "parent.yaml",
        provider=FakeProvider([]),
        child_provider_factory=lambda config, *, config_path: FakeProvider([]),
        memory_service=parent_service,
    )
    controller = parent._subagent_controller
    assert controller is not None
    child = controller._assemble_child(controller.routes.routes["child"])
    service = child.runtime.environment.memory_service
    assert service is not parent_service
    if child_enabled:
        assert service is not None
        assert service.store.path == workspace / ".iris" / "memory" / "memory.db"
        assert child.runtime.environment.agent_config.memory.recall_mode == "manual"
    else:
        assert service is None
    assert not (tmp_path / ".iris").exists()
