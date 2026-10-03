"""Sub Agent 默认权限与最严格组合的可观察契约。"""

from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from iris.tools import BaseTool, CallableTool, ToolCapability, ToolExecutionContext
from iris.tools.permissions import DefaultPermissionPolicy, PermissionDecision, PermissionEffect
from iris.tools.subagent import (
    SubagentExecutionOutcome,
    SubagentInvocation,
    SubagentRoute,
    SubagentRouteTable,
    SubagentTool,
)


class FixedPolicy:
    """记录真实 tool/context 并返回具备诊断字段的决策。"""

    def __init__(self, name: str, effect: PermissionEffect) -> None:
        self.name = name
        self.effect = effect
        self.calls: list[tuple[str, dict[str, Any], Path]] = []

    def check(
        self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
    ) -> PermissionDecision:
        self.calls.append((tool.name, params, context.workspace_root))
        return PermissionDecision(
            effect=self.effect, reason=self.name, metadata={"label": self.name}
        )


@pytest.mark.parametrize(
    "execute,effect",
    [
        ("allow", PermissionEffect.ALLOW),
        ("confirm", PermissionEffect.REQUIRE_HUMAN),
        ("deny", PermissionEffect.DENY),
    ],
)
def test_execute_policy_is_independent_of_file_writes(
    tmp_path: Path, execute: str, effect: PermissionEffect
) -> None:
    command = CallableTool(
        lambda: "ok", name="command", description="command", capabilities={ToolCapability.EXECUTE}
    )
    policy = DefaultPermissionPolicy(execute_mode=execute, write_mode="deny")
    assert policy.check(command, {}, ToolExecutionContext(workspace_root=tmp_path)).effect is effect


@pytest.mark.parametrize("capability", [ToolCapability.MCP, ToolCapability.NETWORK])
def test_execute_allow_does_not_bypass_other_capability_policies(
    tmp_path: Path, capability: ToolCapability
) -> None:
    tool = CallableTool(
        lambda: "ok",
        name="mixed",
        description="mixed",
        capabilities={ToolCapability.EXECUTE, capability},
    )
    policy = DefaultPermissionPolicy(execute_mode="allow")
    assert (
        policy.check(tool, {}, ToolExecutionContext(workspace_root=tmp_path)).effect
        is PermissionEffect.REQUIRE_HUMAN
    )


def test_default_allow_is_specific_to_concrete_subagent_tool(tmp_path: Path) -> None:
    class Port:
        async def execute(self, invocation: SubagentInvocation) -> SubagentExecutionOutcome:
            raise AssertionError("permission must not execute")

    tool = SubagentTool(
        routes=SubagentRouteTable(
            "a",
            MappingProxyType(
                {
                    "a": SubagentRoute("a", tmp_path / "a.yaml", "A"),
                }
            ),
        ),
        port=Port(),
    )
    ordinary = CallableTool(
        lambda: "x",
        name="subagent",
        description="Ordinary agent",
        capabilities={ToolCapability.AGENT},
    )
    policy = DefaultPermissionPolicy()
    context = ToolExecutionContext(workspace_root=tmp_path)
    assert policy.check(tool, {}, context).effect == PermissionEffect.ALLOW
    assert policy.check(ordinary, {}, context).effect == PermissionEffect.REQUIRE_HUMAN


@pytest.mark.parametrize(
    "parent,child,winner",
    [
        ("allow", "deny", "child"),
        ("deny", "allow", "parent"),
        ("allow", "require_human", "child"),
        ("require_human", "allow", "parent"),
        ("require_human", "deny", "child"),
        ("deny", "require_human", "parent"),
        ("allow", "allow", "parent"),
        ("deny", "deny", "parent"),
        ("require_human", "require_human", "parent"),
    ],
)
def test_composite_preserves_strictest_decision_and_checks_actual_tool(
    tmp_path: Path,
    parent: str,
    child: str,
    winner: str,
) -> None:
    from iris.tools.permissions import MostRestrictivePermissionPolicy

    parent_policy = FixedPolicy("parent", PermissionEffect(parent))
    child_policy = FixedPolicy("child", PermissionEffect(child))
    policy = MostRestrictivePermissionPolicy(parent_policy, child_policy)
    tool = CallableTool(lambda: "x", name="child_only", description="Child only")
    for _ in range(2):
        decision = policy.check(tool, {"value": "x"}, ToolExecutionContext(workspace_root=tmp_path))
        assert decision.effect.value == (parent if winner == "parent" else child)
        assert decision.reason == winner
        assert decision.metadata == {"label": winner}
    assert (
        parent_policy.calls == child_policy.calls == [("child_only", {"value": "x"}, tmp_path)] * 2
    )


@pytest.mark.parametrize("name", ["web_search", "web_fetch"])
def test_web_builtins_are_automatic_without_allowing_other_network_tools(
    tmp_path: Path, name: str
) -> None:
    """具体 Web builtin 自动执行，普通 NETWORK 工具仍走原策略。"""
    from iris.tools import WebFetchTool, WebSearchTool

    tools = {
        "web_search": WebSearchTool(api_key="tvly-test"),
        "web_fetch": WebFetchTool(api_key="tvly-test"),
    }
    tool = tools[name]
    ordinary = CallableTool(
        lambda: "x", name=name, description="Network tool", capabilities={ToolCapability.NETWORK}
    )
    policy = DefaultPermissionPolicy()
    context = ToolExecutionContext(workspace_root=tmp_path)

    assert policy.check(tool, {}, context).effect == PermissionEffect.ALLOW
    assert policy.check(ordinary, {}, context).effect == PermissionEffect.REQUIRE_HUMAN
    assert ToolCapability.NETWORK in tool.definition.capabilities
    assert not tool.is_read_only({})


@pytest.mark.asyncio
async def test_web_call_refreshes_parent_policy_before_execution(tmp_path: Path) -> None:
    """父级策略在预检后变化时，Web 默认允许不能绕过执行前刷新。"""
    from iris.message import ToolUseBlock
    from iris.tools import ToolExecutor, ToolRegistry, WebSearchTool
    from iris.tools.permissions import MostRestrictivePermissionPolicy

    parent = FixedPolicy("parent", PermissionEffect.ALLOW)
    registry = ToolRegistry()
    registry.register(WebSearchTool(api_key="tvly-test"))
    executor = ToolExecutor(
        registry,
        permission_policy=MostRestrictivePermissionPolicy(parent, DefaultPermissionPolicy()),
    )
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="web", name="web_search", input={"query": "Python"})], context
    ).calls[0]
    assert prepared.preflight_result is None
    parent.effect = PermissionEffect.DENY

    result = await executor.execute_prepared(prepared, context)

    assert result.error is not None and result.error.code == "PERMISSION_ERROR"
    assert result.error.message == "parent"
    assert len(parent.calls) >= 2


@pytest.mark.parametrize("name", ["tool_search", "memory_search"])
def test_default_allows_concrete_decision_search_not_other_network_tools(
    tmp_path: Path, name: str
) -> None:
    """内置搜索可联网，普通同名工具保持原有权限。"""
    from iris.decision import DecisionRequest, DecisionResponse
    from iris.memory import MemoryAccessPolicy, MemorySearchTool, MemoryService, SQLiteMemoryStore
    from iris.tools import ToolRegistry, ToolSearchTool

    class Port:
        async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
            raise AssertionError("权限检查不执行远程调用")

    search = (
        ToolSearchTool(ToolRegistry().view(), decision_client=Port())
        if name == "tool_search"
        else MemorySearchTool(
            service=MemoryService(SQLiteMemoryStore(tmp_path / "memory.db")),
            access_policy_factory=lambda _: MemoryAccessPolicy(),
            decision_client=Port(),
        )
    )
    ordinary = CallableTool(
        lambda: "ok", name=name, description="other", capabilities={ToolCapability.NETWORK}
    )
    policy = DefaultPermissionPolicy()
    context = ToolExecutionContext(workspace_root=tmp_path)
    assert policy.check(search, {}, context).effect == PermissionEffect.ALLOW
    assert policy.check(ordinary, {}, context).effect == PermissionEffect.REQUIRE_HUMAN
    assert not search.is_read_only({})


@pytest.mark.asyncio
@pytest.mark.parametrize("check_at", ["preflight", "refresh"])
@pytest.mark.parametrize("name", ["tool_search", "memory_search"])
async def test_custom_policy_denial_prevents_decision_search_call(
    tmp_path: Path, check_at: str, name: str
) -> None:
    """自定义策略的预检和执行前刷新均能阻止 Decision 请求。"""
    from iris.decision import DecisionRequest, DecisionResponse
    from iris.memory import (
        MemoryAccessPolicy,
        MemorySearchTool,
        MemoryService,
        MemoryWriteInput,
        SQLiteMemoryStore,
    )
    from iris.message import ToolUseBlock
    from iris.tools import ToolExecutor, ToolRegistry, ToolSearchTool

    class Port:
        async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
            raise AssertionError("拒绝后不得发送请求")

    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="docs", description="docs", deferred=True)
    if name == "tool_search":
        registry.register(ToolSearchTool(registry.view(), decision_client=Port()))
        arguments = {"queries": ["docs"]}
    else:
        service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
        service.remember(MemoryWriteInput(text="docs", reason="test"))
        registry.register(
            MemorySearchTool(
                service=service,
                access_policy_factory=lambda _: MemoryAccessPolicy(),
                decision_client=Port(),
            )
        )
        arguments = {"query": "docs"}
    policy = FixedPolicy(
        "custom", PermissionEffect.DENY if check_at == "preflight" else PermissionEffect.ALLOW
    )
    executor = ToolExecutor(registry, permission_policy=policy)
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="search", name=name, input=arguments)], context
    ).calls[0]
    if check_at == "refresh":
        assert prepared.preflight_result is None
        policy.effect = PermissionEffect.DENY
    result = await executor.execute_prepared(prepared, context)
    assert result.error is not None and result.error.code == "PERMISSION_ERROR"
    assert result.error.message == "custom"
