"""工具权限与 workspace 边界策略。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..exceptions import IrisToolValidationError
from .base import BaseTool, ToolCapability, ToolExecutionContext


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

    @property
    def allowed(self) -> bool:
        """兼容旧调用方的 allow 判断。"""
        return self.effect is PermissionEffect.ALLOW

    @property
    def require_confirmation(self) -> bool:
        """兼容旧调用方的人工确认判断。"""
        return self.effect is PermissionEffect.REQUIRE_HUMAN


class ReadFileRecord(BaseModel):
    """已读取文件的乐观锁记录。"""

    path: Path
    mtime_ns: int
    size_bytes: int
    digest: str | None = None


class ReadFileState(BaseModel):
    """当前上下文中已读取文件状态。"""

    files: dict[str, ReadFileRecord] = Field(default_factory=dict)

    def update(self, path: Path) -> None:
        """用当前文件 stat 刷新记录。"""
        stat = path.stat()
        resolved = path.resolve()
        self.files[str(resolved)] = ReadFileRecord(
            path=resolved,
            mtime_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
        )

    def get(self, path: Path) -> ReadFileRecord | None:
        """按 resolve 后路径获取读取记录。"""
        return self.files.get(str(path.resolve()))


class WorkspacePolicy:
    """统一解析路径并拒绝 workspace 外访问。"""

    def resolve_path(self, path: str, *, workspace_root: Path) -> Path:
        """解析用户路径为 workspace 内绝对路径。"""
        root = workspace_root.resolve()
        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else root / raw_path
        resolved = candidate.resolve(strict=False)
        if not self.is_within_workspace(resolved, root):
            raise IrisToolValidationError(
                "PATH_OUTSIDE_WORKSPACE: 路径不在 workspace 内",
                path=path,
                workspace_root=str(root),
            )
        return resolved

    def is_within_workspace(self, path: Path, workspace_root: Path) -> bool:
        """判断路径是否在 workspace 内。"""
        try:
            path.resolve(strict=False).relative_to(workspace_root.resolve())
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


class DefaultPermissionPolicy(PermissionPolicy):
    """保守默认权限策略。"""

    def __init__(
        self,
        *,
        workspace_policy: WorkspacePolicy | None = None,
        write_mode: Literal["confirm", "allow", "deny"] = "confirm",
        allow_writes: bool | None = None,
    ) -> None:
        """初始化默认策略。"""
        if allow_writes is not None:
            compatibility_mode: Literal["allow", "confirm"] = (
                "allow" if allow_writes else "confirm"
            )
            if write_mode != "confirm" and write_mode != compatibility_mode:
                raise IrisToolValidationError("write_mode 与 allow_writes 配置冲突")
            write_mode = compatibility_mode
        self.workspace_policy = workspace_policy or WorkspacePolicy()
        self.write_mode = write_mode
        self.allow_writes = write_mode == "allow"

    def check(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PermissionDecision:
        """只读允许，写入依 write_mode，其他高风险能力需要人工确认。"""
        del context
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
