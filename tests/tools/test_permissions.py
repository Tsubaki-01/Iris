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
