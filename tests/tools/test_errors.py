"""公共工具 unknown 与命令执行错误的领域契约。"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.command.models import (
    CommandMode,
    CommandOutcome,
    CommandOutputStats,
    CommandStatus,
    CommandStopReceipt,
)
from iris.exceptions import (
    IrisCommandCleanupError,
    IrisCommandError,
    IrisMCPError,
    IrisToolError,
    IrisToolOutcomeUnknownError,
)
from iris.message import TextBlock, ToolUseBlock
from iris.runtime.runtime import _normalize_run_error
from iris.tools import (
    BaseTool,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
)


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (IrisCommandError, "COMMAND_ERROR"),
        (IrisCommandCleanupError, "COMMAND_CLEANUP_FAILED"),
    ],
)
def test_command_errors_keep_tool_source(error_type: type[IrisCommandError], code: str) -> None:
    """执行与清理异常沿用 tool 来源，保留各自错误码和普通诊断。"""
    error = error_type("执行环境失败", operation="stop")

    normalized = _normalize_run_error(error)

    assert isinstance(error, IrisToolError)
    assert normalized.source == "tool"
    assert normalized.code == code
    assert normalized.details == {"operation": "stop"}


def test_unknown_stop_receipt_is_separate_from_error_context() -> None:
    """停止证明保持对象身份，但不进入会被归档的通用错误详情。"""
    receipt = CommandStopReceipt(service_id="service", stop_id="stop")
    error = IrisToolOutcomeUnknownError("命令结果无法确认", stop_receipt=receipt, operation="exec")

    normalized = _normalize_run_error(error)

    assert isinstance(error, IrisToolError)
    assert not isinstance(error, IrisMCPError)
    assert error.stop_receipt is receipt
    assert normalized.source == "tool"
    assert normalized.code == "TOOL_OUTCOME_UNKNOWN"
    assert normalized.details == {"operation": "exec"}
    assert IrisToolOutcomeUnknownError("无停止证明").stop_receipt is None


def test_cleanup_known_outcome_is_separate_from_error_details() -> None:
    """已知命令事实供结算使用，输出与收据不能混入通用错误详情。"""
    outcome = CommandOutcome(
        mode=CommandMode.DOCKER,
        status=CommandStatus.EXITED,
        exit_code=7,
        stdout="known stdout",
        stderr="known stderr",
        output_stats=CommandOutputStats(12, 12, 12, 12, frozenset()),
        duration_seconds=0.5,
        cwd=".",
    )
    error = IrisCommandCleanupError(
        "已知命令结束但环境清理失败", command_outcome=outcome, operation="stop"
    )
    assert error.command_outcome is outcome
    assert error.context == {"operation": "stop"}
    assert _normalize_run_error(error).details == {"operation": "stop"}
    assert IrisCommandCleanupError("没有已知结果", started=False).command_outcome is None


@pytest.mark.asyncio
async def test_non_mcp_unknown_survives_middleware_replacement(tmp_path: Path) -> None:
    """普通 BaseTool 的 unknown 不因包装器合成成功而丢失。"""
    receipt = CommandStopReceipt(service_id="service", stop_id="stop")
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
        """尝试将下游控制异常转为普通成功。"""

        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            """执行器必须保留已锁存的原控制异常。"""
            try:
                return await call_next()
            except Exception:
                return ToolResult(
                    tool_use_id=call.tool_use_id,
                    tool_name=call.tool_name,
                    content=[TextBlock(text="swallowed")],
                )

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
