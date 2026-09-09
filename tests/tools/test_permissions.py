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
        self.fingerprint_calls = 0

    def check(
        self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
    ) -> PermissionDecision:
        self.calls.append((tool.name, params, context.workspace_root))
        return PermissionDecision(
            effect=self.effect, reason=self.name, metadata={"label": self.name}
        )

    def fingerprint_payload(self) -> dict[str, object]:
        self.fingerprint_calls += 1
        return {"type": self.name, "effect": self.effect.value}


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
    assert policy.fingerprint_payload() == {
        "type": "most_restrictive",
        "parent": {"type": "parent", "effect": parent},
        "child": {"type": "child", "effect": child},
    }
    assert parent_policy.fingerprint_calls == child_policy.fingerprint_calls == 1
