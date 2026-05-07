"""工具执行入口。"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

from pydantic import ValidationError

from ..exceptions import (
    IrisToolExecutionError,
    IrisToolNotFoundError,
    IrisToolValidationError,
)
from ..message import ToolUseBlock
from .artifacts import ToolArtifactStore
from .base import ToolErrorInfo, ToolExecutionContext, ToolResult
from .permissions import DefaultPermissionPolicy, PermissionPolicy
from .registry import ToolRegistry


class ToolExecutor:
    """执行 `ToolUseBlock` 的工具执行器。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_policy: PermissionPolicy | None = None,
        artifact_preview_chars: int = 8000,
    ) -> None:
        """初始化执行器。"""
        self.registry = registry
        self.permission_policy = permission_policy or DefaultPermissionPolicy()
        self.artifact_preview_chars = artifact_preview_chars

    async def execute_one(
        self,
        tool_use: ToolUseBlock,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行单个工具调用并将错误归一化为 `ToolResult`。"""
        context.call_id = tool_use.id
        context.tool_name = tool_use.name
        execution_context = context
        try:
            tool = self.registry.get(tool_use.name)
            params = tool.validate_input(tool_use.input)
            raw_params = params.model_dump() if hasattr(params, "model_dump") else dict(params)
            try:
                decision = self.permission_policy.check(tool, raw_params, execution_context)
            except Exception as exc:
                return self._error_result(tool_use, "PERMISSION_ERROR", str(exc))
            if not decision.allowed:
                return self._error_result(
                    tool_use,
                    "PERMISSION_ERROR",
                    decision.reason,
                    details={
                        "require_confirmation": decision.require_confirmation,
                        **decision.metadata,
                    },
                )
            result = await tool.arun(params, execution_context)
            normalized = result.model_copy(
                update={
                    "tool_use_id": result.tool_use_id or tool_use.id,
                    "tool_name": result.tool_name or tool_use.name,
                }
            )
            artifact_store = self._artifact_store(execution_context)
            return artifact_store.persist_if_large(
                normalized,
                max_chars=tool.definition.max_result_chars,
            )
        except IrisToolNotFoundError:
            return self._error_result(tool_use, "NOT_FOUND", f"工具不存在: {tool_use.name}")
        except (IrisToolValidationError, ValidationError) as exc:
            return self._error_result(tool_use, "VALIDATION_ERROR", str(exc))
        except IrisToolExecutionError as exc:
            code, message = _tool_error_code_and_message(
                exc.message,
                allow_structured=(
                    tool.definition.group == "file"
                    or exc.message.startswith("ARTIFACT_ERROR:")
                ),
            )
            return self._error_result(tool_use, code, message)
        except Exception as exc:
            return self._error_result(tool_use, "EXECUTION_ERROR", str(exc))

    async def execute_many(
        self,
        tool_uses: Sequence[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """执行多个工具调用，只读且并发安全的连续批次并发执行。"""
        results: list[ToolResult] = []
        batch: list[ToolUseBlock] = []
        for tool_use in tool_uses:
            if self._is_read_only_concurrency_safe(tool_use):
                batch.append(tool_use)
                continue
            if batch:
                results.extend(await self._execute_read_batch(batch, context))
                batch = []
            results.append(await self.execute_one(tool_use, context))
        if batch:
            results.extend(await self._execute_read_batch(batch, context))
        return results

    def _error_result(
        self,
        tool_use: ToolUseBlock,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> ToolResult:
        """构造错误工具结果。"""
        return ToolResult(
            tool_use_id=tool_use.id,
            tool_name=tool_use.name,
            is_error=True,
            error=ToolErrorInfo(code=code, message=message, details=details or {}),
        )

    async def _execute_read_batch(
        self,
        tool_uses: list[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """并发执行连续只读批次。"""
        tasks = (self.execute_one(tool_use, context) for tool_use in tool_uses)
        return list(await asyncio.gather(*tasks))

    def _is_read_only_concurrency_safe(self, tool_use: ToolUseBlock) -> bool:
        """判断工具调用是否可进入只读并发批次。"""
        try:
            tool = self.registry.get(tool_use.name)
            params = tool.validate_input(tool_use.input)
            raw_params = params.model_dump() if hasattr(params, "model_dump") else dict(params)
            return tool.is_read_only(raw_params) and tool.is_concurrency_safe(raw_params)
        except (IrisToolNotFoundError, IrisToolValidationError, ValidationError):
            return False

    def _artifact_store(self, context: ToolExecutionContext) -> ToolArtifactStore:
        """为当前上下文创建 artifact store。"""
        session_id = _safe_path_segment(context.session_id or "default")
        root = context.workspace_root / ".iris" / "tool-results" / session_id
        return ToolArtifactStore(root=root, preview_chars=self.artifact_preview_chars)


def _tool_error_code_and_message(
    message: str,
    *,
    allow_structured: bool,
) -> tuple[str, str]:
    """从工具异常消息中提取稳定错误码。"""
    if not allow_structured:
        return "EXECUTION_ERROR", message
    match = re.match(r"^([A-Z][A-Z0-9_]+):\s*(.*)$", message)
    if match is None:
        return "EXECUTION_ERROR", message
    return match.group(1), match.group(2)


def _safe_path_segment(value: str) -> str:
    """将外部 ID 转为单个安全路径段。"""
    segment = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return segment.strip("_") or "default"
