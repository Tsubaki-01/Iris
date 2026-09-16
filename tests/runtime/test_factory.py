from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig, CompactionConfig
from iris.exceptions import IrisConfigError
from iris.message import LLMResponse, TextBlock
from iris.runtime import AgentRuntime, RuntimeFactory


def _response() -> LLMResponse:
    return LLMResponse(provider="fake", content=[TextBlock(text="完成")])


def test_factory_preserves_compaction_in_agent_config() -> None:
    config = AgentConfig.model_validate(
        {
            "name": "agent",
            "model": "openai/test",
            "system": "instructions",
            "compaction": {"input_budget_tokens": 32000, "summary_ratio": 0.1},
        }
    )

    runtime = RuntimeFactory.from_config(config, provider=FakeProvider([]))

    assert runtime.environment.agent_config is config
    assert runtime.environment.agent_config.compaction == CompactionConfig(
        input_budget_tokens=32000, summary_ratio=0.1
    )


def test_from_config_path_loads_relative_context_without_creating_store(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "context.yaml"
    context_path.write_text(
        "system:\n  slots:\n    - name: instructions\n      content: 来自 context.yaml\n",
        encoding="utf-8",
    )
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(
        "\n".join(
            [
                "name: context-agent",
                "model: openai/gpt-4o-mini",
                "context:",
                "  path: context.yaml",
                "session:",
                "  backend: sqlite",
                "  path: state/lifecycle.db",
            ]
        ),
        encoding="utf-8",
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


def test_public_factory_requires_runner_for_subagent_config() -> None:
    config = AgentConfig.model_validate(
        {
            "name": "parent",
            "model": "openai/test",
            "system": "parent",
            "tools": {"subagent": "missing.yaml"},
        }
    )
    with pytest.raises(IrisConfigError, match="AgentRunner"):
        RuntimeFactory.from_config(config, provider=FakeProvider([]))


@pytest.mark.parametrize(
    "parent,child,expected", [(".", ".", "."), (".", "inner", "inner"), ("inner", ".", "inner")]
)
def test_child_boundary_uses_narrower_workspace(
    tmp_path: Path,
    parent: str,
    child: str,
    expected: str,
) -> None:
    from iris.runtime._assembly import resolve_runtime_boundary

    def config(workspace: str) -> AgentConfig:
        return AgentConfig.model_validate(
            {
                "name": "agent",
                "model": "openai/test",
                "system": "agent",
                "permissions": {"workspace": workspace},
            }
        )

    parent_boundary = resolve_runtime_boundary(config(parent), config_path=tmp_path / "agent.yaml")
    boundary = resolve_runtime_boundary(
        config(child),
        config_path=tmp_path / "child.yaml",
        parent_boundary=parent_boundary,
    )
    assert boundary.workspace_root == (tmp_path / expected).resolve()


def test_disjoint_child_workspace_fails_with_both_roots(tmp_path: Path) -> None:
    from iris.runtime._assembly import resolve_runtime_boundary

    config = AgentConfig.model_validate({"name": "a", "model": "openai/test", "system": "a"})
    parent = resolve_runtime_boundary(config, config_path=tmp_path / "parent" / "agent.yaml")
    with pytest.raises(IrisConfigError) as caught:
        resolve_runtime_boundary(
            config, config_path=tmp_path / "child" / "agent.yaml", parent_boundary=parent
        )
    assert caught.value.context["parent_workspace"] == str(tmp_path / "parent")
    assert caught.value.context["child_workspace"] == str(tmp_path / "child")


@pytest.mark.parametrize("scope", ["root", "child"])
def test_private_assembly_registers_subagent_only_in_root(tmp_path: Path, scope: str) -> None:
    from iris.agents.config.subagent import load_subagent_catalog
    from iris.harness import AgentRunner
    from iris.runtime._assembly import (
        RuntimeExecutionScope,
        SubagentAssembly,
        assemble_runtime,
        resolve_runtime_boundary,
    )
    from iris.store import InMemoryLifecycleStore
    from iris.tools.subagent import SubagentExecutionOutcome, SubagentInvocation

    class Port:
        async def execute(self, invocation: SubagentInvocation) -> SubagentExecutionOutcome:
            raise AssertionError("assembly must not execute")

    path = tmp_path / "catalog.yaml"
    path.write_text(
        "default: researcher\nagents:\n  researcher:\n    path: missing.yaml\n"
        "    description: Research\n",
        encoding="utf-8",
    )
    routes = load_subagent_catalog(path)
    config = AgentConfig.model_validate(
        {
            "name": "a",
            "model": "openai/test",
            "system": "a",
            "tools": {"builtin": ["file.read"], "subagent": "missing.yaml"},
        }
    )
    parent = resolve_runtime_boundary(config, config_path=tmp_path / "agent.yaml")
    boundary = (
        resolve_runtime_boundary(
            config, config_path=tmp_path / "child.yaml", parent_boundary=parent
        )
        if scope == "child"
        else parent
    )
    runtime = assemble_runtime(
        config,
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([]),
        memory_service=None,
        api_key=None,
        execution_scope=RuntimeExecutionScope(scope),
        boundary=boundary,
        subagent=SubagentAssembly(routes, Port()),
    )
    view = runtime.environment.tool_bridge.tool_view
    assert view.get("read_file").name == "read_file"
    names = [tool.name for tool in view.active_tools]
    assert ("subagent" in names) == (scope == "root")
    if scope == "root":
        assert view.get("subagent").input_schema["properties"]["agent"]["enum"] == ["researcher"]
    # 普通 Runner fingerprint 必须能够消费 composite policy。
    runner = AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())
    assert runner.runtime.environment.workspace_root == tmp_path


def test_child_executor_uses_custom_parent_policy(tmp_path: Path) -> None:
    from iris.message import ToolUseBlock
    from iris.runtime._assembly import (
        RuntimeExecutionScope,
        assemble_runtime,
        resolve_runtime_boundary,
    )
    from iris.tools import (
        PermissionDecision,
        PermissionEffect,
        PermissionPolicy,
        ToolExecutionContext,
    )

    class DenyReads(PermissionPolicy):
        def check(self, tool: object, params: object, context: object) -> PermissionDecision:
            return PermissionDecision(effect=PermissionEffect.DENY, reason="parent disallows reads")

    config = AgentConfig.model_validate(
        {
            "name": "a",
            "model": "openai/test",
            "system": "a",
            "tools": {"builtin": ["file.read"]},
        }
    )
    parent = resolve_runtime_boundary(
        config, config_path=tmp_path / "a.yaml", permission_policy=DenyReads()
    )
    boundary = resolve_runtime_boundary(
        config, config_path=tmp_path / "b.yaml", parent_boundary=parent
    )
    runtime = assemble_runtime(
        config,
        config_path=tmp_path / "b.yaml",
        provider=FakeProvider([]),
        memory_service=None,
        api_key=None,
        execution_scope=RuntimeExecutionScope.CHILD,
        boundary=boundary,
    )
    plan = runtime.environment.tool_bridge.tool_executor.prepare_many(
        [ToolUseBlock(id="read", name="read_file", input={"file_path": "notes.txt"})],
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert plan.calls[0].preflight_result.error.message == "parent disallows reads"
