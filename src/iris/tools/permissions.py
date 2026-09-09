"""工具权限与 workspace 边界策略。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..exceptions import IrisConfigError, IrisToolValidationError
from .base import BaseTool, ToolCapability, ToolExecutionContext
from .subagent import SubagentTool


class PermissionEffect(StrEnum):
    """权限策略的三态裁决。"""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HUMAN = "require_human"


class PermissionDecision(BaseModel):
    """权限策略裁决结果。"""

    effect: PermissionEffect
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reason(self) -> PermissionDecision:
        """拒绝或需要人工确认时要求提供原因。"""
        if (
            self.effect in {PermissionEffect.DENY, PermissionEffect.REQUIRE_HUMAN}
            and not self.reason.strip()
        ):
            raise ValueError("权限拒绝必须包含原因")
        return self


class WorkspacePolicy:
    """统一解析路径并拒绝 workspace 外访问。"""

    def resolve_path(self, path: str, *, workspace_root: Path) -> Path:
        """解析用户路径为 workspace 内绝对路径。"""
        root = workspace_root.resolve()
        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else root / raw_path
        resolved = candidate.resolve(strict=False)
        if not _is_resolved_within(resolved, root):
            raise IrisToolValidationError(
                "PATH_OUTSIDE_WORKSPACE: 路径不在 workspace 内",
                path=path,
                workspace_root=str(root),
            )
        return resolved

    def is_within_workspace(self, path: Path, workspace_root: Path) -> bool:
        """判断路径是否在 workspace 内。"""
        return _is_resolved_within(
            path.resolve(strict=False),
            workspace_root.resolve(strict=False),
        )


def _is_resolved_within(path: Path, root: Path) -> bool:
    """判断两个已 resolve 路径的包含关系。"""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class PermissionPolicy:
    """权限策略接口。"""

    def check(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PermissionDecision:
        """返回工具调用权限裁决。"""
        raise NotImplementedError

    def fingerprint_payload(self) -> dict[str, object]:
        """返回决定恢复兼容性的确定性 JSON-safe 策略状态。"""
        raise IrisConfigError(
            "自定义权限策略必须实现 fingerprint_payload()",
            policy_type=type(self).__qualname__,
        )


class DefaultPermissionPolicy(PermissionPolicy):
    """保守默认权限策略。"""

    def __init__(
        self,
        *,
        workspace_policy: WorkspacePolicy | None = None,
        write_mode: Literal["confirm", "allow", "deny"] = "confirm",
    ) -> None:
        """初始化默认策略。"""
        self.workspace_policy = workspace_policy or WorkspacePolicy()
        self.write_mode = write_mode

    def check(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PermissionDecision:
        """只读允许，写入依 write_mode，其他高风险能力需要人工确认。"""
        del context
        if isinstance(tool, SubagentTool):
            return PermissionDecision(effect=PermissionEffect.ALLOW)
        if tool.definition.capabilities <= {ToolCapability.READ}:
            return PermissionDecision(effect=PermissionEffect.ALLOW)
        if tool.definition.capabilities <= {
            ToolCapability.READ,
            ToolCapability.WRITE,
        }:
            if self.write_mode == "allow":
                return PermissionDecision(effect=PermissionEffect.ALLOW)
            if self.write_mode == "deny":
                return PermissionDecision(
                    effect=PermissionEffect.DENY,
                    reason="工具写入权限被策略拒绝",
                    metadata={"tool": tool.name, "params": params},
                )
        return PermissionDecision(
            effect=PermissionEffect.REQUIRE_HUMAN,
            reason="工具调用需要用户确认",
            metadata={"tool": tool.name, "params": params},
        )

    def fingerprint_payload(self) -> dict[str, object]:
        """返回默认权限策略影响工具执行的稳定状态。"""
        return {
            "type": "default",
            "version": 1,
            "write_mode": self.write_mode,
        }


class MostRestrictivePermissionPolicy(PermissionPolicy):
    """对实际 child 工具分别裁决，两侧同级时保留 parent 决策。"""

    def __init__(self, parent: PermissionPolicy, child: PermissionPolicy) -> None:
        self.parent = parent
        self.child = child

    def check(
        self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
    ) -> PermissionDecision:
        """取 DENY、REQUIRE_HUMAN、ALLOW 顺序中更严格的原始决策。"""
        parent = self.parent.check(tool, params, context)
        child = self.child.check(tool, params, context)
        priority = {
            PermissionEffect.ALLOW: 0,
            PermissionEffect.REQUIRE_HUMAN: 1,
            PermissionEffect.DENY: 2,
        }
        return child if priority[child.effect] > priority[parent.effect] else parent

    def fingerprint_payload(self) -> dict[str, object]:
        """将两侧策略状态交给既有 environment fingerprint owner。"""
        return {
            "type": "most_restrictive",
            "parent": self.parent.fingerprint_payload(),
            "child": self.child.fingerprint_payload(),
        }
