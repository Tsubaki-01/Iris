"""Python 工具借用命令内核、权限与结果投影，不拥有另一套执行循环。"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from iris.command import (
    CommandOutputStats,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
    PythonCode,
)
from iris.exceptions import IrisCommandCleanupError, IrisCommandError, IrisToolOutcomeUnknownError
from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    BaseTool,
    DefaultPermissionPolicy,
    RunPythonTool,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
    ToolTimeoutOwner,
)

from .test_exec_command import FakeCommandService, binding, outcome


def executor_for(service: FakeCommandService) -> tuple[ToolExecutor, RunPythonTool]:
    """只注册 Python 入口并显式允许执行。"""
    registry = ToolRegistry()
    tool = RunPythonTool(binding(service))
    registry.register(tool)
    return ToolExecutor(
        registry, permission_policy=DefaultPermissionPolicy(execute_mode="allow")
    ), tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested,runtime_limit,expected",
    [(None, None, 120), (20, None, 20), (200, 30, 30), (2, 30, 2)],
)
async def test_python_preserves_source_and_uses_shared_scope_cwd_and_deadline(
    tmp_path: Path, requested: float | None, runtime_limit: float | None, expected: float
) -> None:
    """Python 载荷原样传递，调用身份、目录和最短期限由共享入口构造。"""
    child = tmp_path / "child"
    child.mkdir()
    source = "print('中文与 \\\" 引号')\nvalue = 1\n"
    service = FakeCommandService(outcome())
    executor, tool = executor_for(service)
    context = ToolExecutionContext(
        workspace_root=child,
        session_id="session",
        metadata={"run_id": "run"},
        tool_timeout_seconds=runtime_limit,
    )
    arguments: dict[str, Any] = {"code": source}
    if requested is not None:
        arguments["timeout_seconds"] = requested
    call = ToolUseBlock(id="python-call", name="run_python", input=arguments)
    prepared = executor.prepare_many([call], context).calls[0]
    assert prepared.timeout_owner is ToolTimeoutOwner.TOOL
    result = await executor.execute_prepared(prepared, context)
    scope, request = service.calls[0]
    assert scope == CommandScope("run", "session")
    assert request.payload == PythonCode(source)
    assert request.cwd == child.resolve()
    assert request.timeout_seconds == expected
    assert result.tool_name == "run_python" and result.tool_use_id == "python-call"
    assert tool.definition.capabilities == {ToolCapability.EXECUTE}
    assert tool.definition.group == "exec"
    assert tool.definition.context_retention == "keep"
    assert tool.definition.preview_mode == "head_tail"
    assert not tool.is_read_only(arguments)
    assert "新进程" in tool.definition.description and "print" in tool.definition.description


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"code": " "},
        {"code": "x", "cwd": ".."},
        {"code": "x", "cwd": "missing"},
        {"code": "x", "timeout_seconds": float("inf")},
        {"code": "x", "file": "script.py"},
        {"code": "x", "interpreter": "python3"},
    ],
)
async def test_invalid_python_input_never_reaches_service(
    tmp_path: Path, arguments: dict[str, Any]
) -> None:
    """非法原始参数或目录在运行用户代码前失败。"""
    service = FakeCommandService(outcome())
    executor, _tool = executor_for(service)
    result = await executor.execute_one(
        ToolUseBlock(id="python", name="run_python", input=arguments),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.error is not None and result.error.code == "VALIDATION_ERROR"
    assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("execute_mode", ["confirm", "deny"])
async def test_python_execute_permission_precedes_backend(
    tmp_path: Path, execute_mode: Literal["confirm", "deny"]
) -> None:
    """确认或拒绝都先于服务调用，不能借 Python 入口跳过 execute 权限。"""
    service = FakeCommandService(outcome())
    registry = ToolRegistry()
    registry.register(RunPythonTool(binding(service)))
    executor = ToolExecutor(
        registry, permission_policy=DefaultPermissionPolicy(execute_mode=execute_mode)
    )
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="python", name="run_python", input={"code": "print(1)"})], context
    ).calls[0]
    if execute_mode == "confirm":
        assert prepared.human_request is not None
    result = await executor.execute_prepared(prepared, context)
    assert result.error is not None and result.error.code == "PERMISSION_ERROR"
    assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,exit_code,error_code",
    [
        (CommandStatus.EXITED, 1, "COMMAND_FAILED"),
        (CommandStatus.TIMED_OUT, None, "COMMAND_TIMEOUT"),
        (CommandStatus.CANCELLED, None, "COMMAND_CANCELLED"),
        (CommandStatus.ENVIRONMENT_INTERRUPTED, None, "COMMAND_ENVIRONMENT_INTERRUPTED"),
    ],
)
async def test_python_results_keep_diagnostics_and_stop_receipt(
    tmp_path: Path, status: CommandStatus, exit_code: int | None, error_code: str
) -> None:
    """业务结果保留 Python 诊断，并按原协议交接实际停止收据。"""
    receipt = CommandStopReceipt("service", "stop")
    service = FakeCommandService(
        outcome(status, exit_code, receipt=receipt, stderr="Traceback\nValueError: invalid")
    )
    executor, _tool = executor_for(service)
    context = ToolExecutionContext(workspace_root=tmp_path)
    result = await executor.execute_one(
        ToolUseBlock(id="python", name="run_python", input={"code": "raise ValueError('invalid')"}),
        context,
    )
    assert result.error is not None and result.error.code == error_code
    assert "ValueError: invalid" in result.model_content
    assert context.command_stop_slot.receipt is receipt
    assert "stdout" not in result.data and "stderr" not in result.data


@pytest.mark.asyncio
async def test_python_unknown_and_cleanup_follow_existing_stop_slot(tmp_path: Path) -> None:
    """未知效果透传，已知结果的清理失败保留在现有调用槽。"""
    receipt = CommandStopReceipt("service", "stop")
    unknown = IrisToolOutcomeUnknownError("lost", stop_receipt=receipt)
    executor, _tool = executor_for(FakeCommandService(unknown))
    context = ToolExecutionContext(workspace_root=tmp_path)
    call = ToolUseBlock(id="python", name="run_python", input={"code": "print(1)"})
    with pytest.raises(IrisToolOutcomeUnknownError) as caught:
        await executor.execute_one(call, context)
    assert caught.value is unknown and context.command_stop_slot.receipt is receipt
    cleanup = IrisCommandCleanupError(
        "cleanup", command_outcome=outcome(exit_code=7, receipt=receipt)
    )
    executor, _tool = executor_for(FakeCommandService(cleanup))
    result = await executor.execute_one(call, context)
    assert result.error is not None and result.error.code == "COMMAND_FAILED"
    assert context.command_stop_slot.cleanup_error is cleanup


@pytest.mark.asyncio
async def test_python_unavailable_does_not_invent_output_stats(tmp_path: Path) -> None:
    """明确未启动沿用原 unavailable 结果，不伪造零输出的执行。"""
    executor, _tool = executor_for(
        FakeCommandService(IrisCommandError("prepare failed", started=False))
    )
    result = await executor.execute_one(
        ToolUseBlock(id="python", name="run_python", input={"code": "print(1)"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.error is not None and result.error.code == "COMMAND_UNAVAILABLE"
    assert result.data == {"mode": "docker", "started": False}


@pytest.mark.asyncio
async def test_python_statistics_have_one_serializable_shape_and_incomplete_notice(
    tmp_path: Path,
) -> None:
    """统计只投影一次，原因排序固定，未读到 EOF 时不推测剩余输出量。"""
    stats = CommandOutputStats(
        1000,
        700,
        200,
        300,
        frozenset({"stream_closed", "byte_limit", "drain_timeout", "stream_error"}),
    )
    executor, _tool = executor_for(FakeCommandService(replace(outcome(), output_stats=stats)))
    result = await executor.execute_one(
        ToolUseBlock(id="python", name="run_python", input={"code": "print(1)"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.data == {
        "mode": "docker",
        "status": "exited",
        "exit_code": 0,
        "cwd": "child",
        "duration_seconds": 0.2,
        "output_truncated": True,
        "output_stats": {
            "stdout_bytes": 1000,
            "stderr_bytes": 700,
            "stdout_retained_bytes": 200,
            "stderr_retained_bytes": 300,
            "truncation_reasons": ["byte_limit", "drain_timeout", "stream_error", "stream_closed"],
        },
    }
    assert "已读取 1700 bytes，后续未知" in result.model_content
    assert "output_stats" in result.model_dump_json()


@pytest.mark.asyncio
async def test_python_middleware_replacement_cannot_erase_receipt(tmp_path: Path) -> None:
    """中间件替换完整结果也不能丢失独立调用槽中的停止事实。"""

    class ReplaceResult(ToolMiddleware):
        """模拟完全替换正文的正常结果后处理。"""

        async def after_call(
            self, tool: BaseTool, result: ToolResult, context: ToolExecutionContext
        ) -> ToolResult:
            """返回不携带原对象信息的新结果。"""
            return ToolResult(tool_use_id="", tool_name="", content=[TextBlock(text="replacement")])

    receipt = CommandStopReceipt("service", "stop")
    registry = ToolRegistry()
    registry.register(RunPythonTool(binding(FakeCommandService(outcome(receipt=receipt)))))
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
        middleware=[ReplaceResult()],
    )
    context = ToolExecutionContext(workspace_root=tmp_path)
    result = await executor.execute_one(
        ToolUseBlock(id="python", name="run_python", input={"code": "print(1)"}), context
    )
    assert result.model_content == "replacement"
    assert context.command_stop_slot.receipt is receipt
