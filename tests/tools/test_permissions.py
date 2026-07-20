from __future__ import annotations

import pytest

from iris.tools import (
    DefaultPermissionPolicy,
    PermissionDecision,
    PermissionEffect,
    ToolCapability,
    ToolExecutionContext,
    ToolRegistry,
)


def test_default_permission_policy_distinguishes_confirm_allow_and_deny() -> None:
    registry = ToolRegistry()
    registry.register_function(
        lambda: "written",
        name="write_note",
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    write_tool = registry.get("write_note")
    context = ToolExecutionContext(workspace_root=".")

    assert (
        DefaultPermissionPolicy(write_mode="confirm").check(write_tool, {}, context).effect
        is PermissionEffect.REQUIRE_HUMAN
    )
    assert (
        DefaultPermissionPolicy(write_mode="allow").check(write_tool, {}, context).effect
        is PermissionEffect.ALLOW
    )
    assert (
        DefaultPermissionPolicy(write_mode="deny").check(write_tool, {}, context).effect
        is PermissionEffect.DENY
    )


def test_permission_decision_only_exposes_effect() -> None:
    decision = PermissionDecision(effect=PermissionEffect.ALLOW)

    assert not hasattr(decision, "allowed")
    assert not hasattr(decision, "require_confirmation")


def test_default_permission_policy_rejects_removed_allow_writes_argument() -> None:
    with pytest.raises(TypeError):
        DefaultPermissionPolicy(allow_writes=True)  # type: ignore[call-arg]


def test_default_permission_policy_only_exposes_write_mode() -> None:
    policy = DefaultPermissionPolicy(write_mode="allow")

    assert policy.write_mode == "allow"
    assert not hasattr(policy, "allow_writes")
