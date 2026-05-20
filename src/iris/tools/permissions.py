"""工具权限与 workspace 边界策略。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..exceptions import IrisToolValidationError
from .base import BaseTool, ToolCapability, ToolExecutionContext


class PermissionDecision(BaseModel):
    """权限策略裁决结果。"""

    allowed: bool
    reason: str = ""
    require_confirmation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str, info: Any) -> str:
        """拒绝时要求提供原因。"""
        if info.data.get("allowed") is False and not value:
            raise ValueError("权限拒绝必须包含原因")
        return value


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
        allow_writes: bool = False,
    ) -> None:
        """初始化默认策略。"""
        self.workspace_policy = workspace_policy or WorkspacePolicy()
        self.allow_writes = allow_writes

    def check(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PermissionDecision:
        """只读默认允许，写入默认需要后续 CLI 确认 UX。"""
        del context
        if tool.definition.capabilities <= {ToolCapability.READ}:
            return PermissionDecision(allowed=True)
        if self.allow_writes and tool.definition.capabilities <= {
            ToolCapability.READ,
            ToolCapability.WRITE,
        }:
            return PermissionDecision(allowed=True)
        # TODO(cli-confirmation-ux): 后续接入 CLI 用户确认流程。
        return PermissionDecision(
            allowed=False,
            reason="TODO(cli-confirmation-ux): 工具调用需要用户确认",
            require_confirmation=True,
            metadata={"tool": tool.name, "params": params},
        )
