"""公共工具 unknown 与命令执行错误的领域契约。"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.exceptions import (
    IrisExecutionCleanupError,
    IrisExecutionError,
    IrisMCPError,
    IrisToolError,
    IrisToolOutcomeUnknownError,
)
from iris.execution.models import ExecutionStopReceipt
from iris.message import ToolUseBlock
from iris.runtime.runtime import _normalize_run_error
from iris.tools import (
    BaseTool,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (IrisExecutionError, "EXECUTION_ERROR"),
        (IrisExecutionCleanupError, "EXECUTION_CLEANUP_FAILED"),
    ],
)
def test_execution_errors_keep_tool_source(error_type: type[IrisExecutionError], code: str) -> None:
    """执行与清理异常沿用 tool 来源，保留各自错误码和普通诊断。"""
    error = error_type("执行环境失败", operation="stop")

    normalized = _normalize_run_error(error)

    assert isinstance(error, IrisToolError)
    assert normalized.source == "tool"
    assert normalized.code == code
    assert normalized.details == {"operation": "stop"}


def test_unknown_stop_receipt_is_separate_from_error_context() -> None:
    """停止证明保持对象身份，但不进入会被归档的通用错误详情。"""
    receipt = ExecutionStopReceipt(service_id="service", stop_id="stop")
    error = IrisToolOutcomeUnknownError("命令结果无法确认", stop_receipt=receipt, operation="exec")

    normalized = _normalize_run_error(error)

    assert isinstance(error, IrisToolError)
    assert not isinstance(error, IrisMCPError)
    assert error.stop_receipt is receipt
    assert normalized.source == "tool"
    assert normalized.code == "TOOL_OUTCOME_UNKNOWN"
    assert normalized.details == {"operation": "exec"}
    assert IrisToolOutcomeUnknownError("无停止证明").stop_receipt is None


@pytest.mark.asyncio
async def test_non_mcp_unknown_bypasses_middleware_and_executor(tmp_path: Path) -> None:
    """普通 BaseTool 同样原样透传 unknown，不让错误 middleware 吞掉收据。"""
    receipt = ExecutionStopReceipt(service_id="service", stop_id="stop")
    unknown = IrisToolOutcomeUnknownError("结果未知", stop_receipt=receipt)

    class UnknownTool(BaseTool):
        """模拟没有 MCP 依赖的未知执行结果。"""

        definition = ToolDefinition(
            name="unknown", description="返回未知结果", input_schema={"type": "object"}
        )

        async def arun(
            self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            """抛出同一个控制异常供执行器透传。"""
            raise unknown

    class SwallowErrors(ToolMiddleware):
        """若进入普通错误处理，就会掩盖执行结果未知。"""

        async def on_error(
            self, tool: BaseTool, error: Exception, context: ToolExecutionContext
        ) -> ToolResult | None:
            """任何调用都说明控制异常进入了错误转换路径。"""
            raise AssertionError("unknown must bypass middleware conversion")

    registry = ToolRegistry()
    registry.register(UnknownTool())
    executor = ToolExecutor(registry, middleware=[SwallowErrors()])

    with pytest.raises(IrisToolOutcomeUnknownError) as caught:
        await executor.execute_one(
            ToolUseBlock(id="call", name="unknown", input={}),
            ToolExecutionContext(workspace_root=tmp_path),
        )

    assert caught.value is unknown
    assert caught.value.stop_receipt is receipt
