from __future__ import annotations

import pytest

from iris.exceptions import IrisToolValidationError
from iris.tools import (
    DefaultPermissionPolicy,
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


def test_default_permission_policy_keeps_allow_writes_compatibility() -> None:
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
        DefaultPermissionPolicy(allow_writes=True).check(write_tool, {}, context).effect
        is PermissionEffect.ALLOW
    )
    assert (
        DefaultPermissionPolicy(allow_writes=False).check(write_tool, {}, context).effect
        is PermissionEffect.REQUIRE_HUMAN
    )


def test_default_permission_policy_rejects_conflicting_compatibility_arguments() -> None:
    with pytest.raises(IrisToolValidationError):
        DefaultPermissionPolicy(write_mode="deny", allow_writes=True)
