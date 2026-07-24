"""Runtime 错误分类与失败结果构造。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..exceptions import IrisError
from ..message import Msg
from .models import (
    RuntimeErrorInfo,
    RuntimeErrorSource,
    RuntimeStatus,
    RuntimeTurnResult,
    ToolBridgeResult,
)


def normalize_runtime_error(error: Exception) -> RuntimeErrorInfo:
    """将 runtime 边界异常归一化为稳定错误信息。"""
    code, source = _classify_runtime_error(error)
    details: dict[str, Any] = {}
    if isinstance(error, IrisError):
        details.update(error.context)
    return RuntimeErrorInfo(
        code=code,
        message=str(error),
        source=source,
        details=details,
    )


def error_result(
    *,
    session_id: str,
    run_id: str,
    error: RuntimeErrorInfo,
    assistant_message: Msg | None = None,
    steps: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> RuntimeTurnResult:
    """构造统一失败结果。"""
    return RuntimeTurnResult(
        session_id=session_id,
        run_id=run_id,
        status=RuntimeStatus.ERROR,
        assistant_message=assistant_message,
        steps=steps,
        error=error,
        metadata=dict(metadata or {}),
    )


def tool_error_info(bridge_result: ToolBridgeResult) -> RuntimeErrorInfo:
    """从第一个工具错误构造 runtime 错误信息。"""
    for result in bridge_result.results:
        if result.is_error and result.error is not None:
            return RuntimeErrorInfo(
                code=result.error.code,
                message=result.error.message,
                source="tool",
                details=result.error.details,
            )
    return RuntimeErrorInfo(
        code="TOOL_ERROR",
        message="工具执行失败",
        source="tool",
    )


def _classify_runtime_error(error: Exception) -> tuple[str, RuntimeErrorSource]:
    """从 Iris 异常实例读取 runtime 错误映射。"""
    if isinstance(error, IrisError):
        return error.runtime_code, error.runtime_source
    return "RUNTIME_ERROR", "runtime"


__all__ = ["normalize_runtime_error"]
