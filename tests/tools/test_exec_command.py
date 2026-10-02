"""命令工具、业务期限和进程内清理事实的集成契约。"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from iris.command import (
    CommandBinding,
    CommandConfig,
    CommandEnvironment,
    CommandMode,
    CommandOutcome,
    CommandOutputStats,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
    ShellCommand,
    StopOperation,
)
from iris.command._output import OutputBuffer
from iris.exceptions import (
    IrisCommandCleanupError,
    IrisCommandError,
    IrisToolOutcomeUnknownError,
)
from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    DefaultPermissionPolicy,
    ExecCommandTool,
    RunPythonTool,
    ToolCall,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
    ToolTimeoutOwner,
)


def test_command_context_keeps_live_slot_identity_and_excludes_serialization(
    tmp_path: Path,
) -> None:
    """深复制 context 也不能复制 live 收据槽或把控制异常放入 JSON。"""
    context = ToolExecutionContext(workspace_root=tmp_path, tool_timeout_seconds=4)
    slot = context.command_stop_slot
    slot.receipt = CommandStopReceipt("service", "stop")
    slot.cleanup_error = IrisCommandCleanupError("等待重试")
    copied = context.model_copy(deep=True)
    assert copied.command_stop_slot is slot
    assert copied.tool_timeout_seconds == 4
    assert "command_stop_slot" not in context.model_dump()
    assert "command_stop_slot" not in context.model_dump_json()
    assert ToolTimeoutOwner.RUNTIME.value == "runtime"
    assert (
        CommandEnvironment("Windows", CommandMode.DOCKER, "Linux", "/bin/sh").command_os == "Linux"
    )


class FakeCommandService:
    """只记录已授权命令请求，返回给定的后端事实。"""

    def __init__(self, result: CommandOutcome | Exception) -> None:
        self.result = result
        self.calls: list[tuple[CommandScope, CommandRequest]] = []

    async def prepare(self) -> None:
        """命令工具不负责服务准备。"""
        raise AssertionError("tool must not prepare service")

    async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        """记录请求并提供可控后端结果。"""
        self.calls.append((scope, request))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def stop(self, scope: CommandScope) -> StopOperation:
        """命令工具不另发一次停止。"""
        raise AssertionError("tool must not repeat backend stop")

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        """工具 body 不等待包括自身的排空。"""
        raise AssertionError("tool must not wait for drained")

    async def aclose(self) -> None:
        """命令工具不拥有关闭生命周期。"""
        raise AssertionError("tool must not close service")


def outcome(
    status: CommandStatus = CommandStatus.EXITED,
    exit_code: int | None = 0,
    *,
    receipt: CommandStopReceipt | None = None,
    stdout: str = "hello",
    stderr: str = "diagnostic",
) -> CommandOutcome:
    """构造已验证的后端事实。"""
    return CommandOutcome(
        CommandMode.DOCKER,
        status,
        exit_code,
        stdout,
        stderr,
        CommandOutputStats(
            len(stdout.encode()),
            len(stderr.encode()),
            len(stdout.encode()),
            len(stderr.encode()),
            frozenset(),
        ),
        0.2,
        "child",
        receipt,
    )


def binding(service: FakeCommandService) -> CommandBinding:
    """工具说明与 service 使用同一启动配置。"""
    return CommandBinding(
        CommandConfig(mode="docker", timeout_seconds=120),
        service,
        CommandEnvironment("Windows", CommandMode.DOCKER, "Linux", "/bin/sh"),
    )


def executor_for(
    service: FakeCommandService, *, middleware: list[ToolMiddleware] | None = None
) -> tuple[ToolExecutor, ExecCommandTool]:
    """绑定命令工具并显式允许执行。"""
    tool = ExecCommandTool(binding(service))
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
        middleware=middleware,
    ), tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "runtime_limit", "expected"),
    [(None, None, 120), (20, None, 20), (200, 30, 30), (2, 30, 2)],
)
async def test_exec_resolves_child_cwd_and_owns_shortest_business_deadline(
    tmp_path: Path, requested: float | None, runtime_limit: float | None, expected: float
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    service = FakeCommandService(outcome())
    executor, tool = executor_for(service)
    context = ToolExecutionContext(
        workspace_root=child,
        session_id="session",
        metadata={"run_id": "run"},
        tool_timeout_seconds=runtime_limit,
    )
    arguments: dict[str, Any] = {"command": "echo hello"}
    if requested is not None:
        arguments["timeout_seconds"] = requested
    call = ToolUseBlock(id="call", name="exec_command", input=arguments)
    prepared = executor.prepare_many([call], context).calls[0]
    assert prepared.timeout_owner is ToolTimeoutOwner.TOOL
    assert tool.definition.capabilities == {ToolCapability.EXECUTE}
    assert tool.definition.context_retention == "keep"
    assert not tool.is_read_only(arguments)
    assert "/bin/sh" in tool.definition.description
    assert "Linux" in tool.definition.description
    result = await executor.execute_prepared(prepared, context)
    scope, request = service.calls[0]
    assert scope == CommandScope("run", "session")
    assert request.cwd == child.resolve()
    assert request.timeout_seconds == expected
    assert request.call_id == "call"
    assert request.payload == ShellCommand("echo hello")
    assert tool.definition.preview_mode == "head_tail"
    assert result.is_error is False
    assert "hello" in result.model_content
    assert "stdout" not in result.data and "stderr" not in result.data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"command": " "},
        {"command": "x", "timeout_seconds": 0},
        {"command": "x", "image": "other"},
        {"command": "x", "cwd": ".."},
    ],
)
async def test_invalid_exec_arguments_never_reach_service(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    service = FakeCommandService(outcome())
    executor, _tool = executor_for(service)
    result = await executor.execute_one(
        ToolUseBlock(id="call", name="exec_command", input=params),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.is_error
    assert result.error is not None and result.error.code == "VALIDATION_ERROR"
    assert not service.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "exit_code", "code"),
    [
        (CommandStatus.EXITED, 124, "COMMAND_FAILED"),
        (CommandStatus.TIMED_OUT, None, "COMMAND_TIMEOUT"),
        (CommandStatus.CANCELLED, None, "COMMAND_CANCELLED"),
        (CommandStatus.ENVIRONMENT_INTERRUPTED, None, "COMMAND_ENVIRONMENT_INTERRUPTED"),
    ],
)
async def test_exec_errors_are_model_visible_and_stop_fact_stays_in_slot(
    tmp_path: Path, status: CommandStatus, exit_code: int | None, code: str
) -> None:
    receipt = CommandStopReceipt("service", "stop")
    service = FakeCommandService(outcome(status, exit_code, receipt=receipt))
    executor, _tool = executor_for(service)
    context = ToolExecutionContext(workspace_root=tmp_path)
    result = await executor.execute_one(
        ToolUseBlock(id="call", name="exec_command", input={"command": "x"}), context
    )
    assert result.error is not None and result.error.code == code
    assert "diagnostic" in result.model_content
    if exit_code is not None:
        assert str(exit_code) in result.model_content
    assert context.command_stop_slot.receipt is receipt
    assert "stop_receipt" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_exec_unknown_records_receipt_before_executor_propagates(tmp_path: Path) -> None:
    receipt = CommandStopReceipt("service", "stop")
    error = IrisToolOutcomeUnknownError("connection lost", stop_receipt=receipt)
    executor, _tool = executor_for(FakeCommandService(error))
    context = ToolExecutionContext(workspace_root=tmp_path)
    with pytest.raises(IrisToolOutcomeUnknownError) as caught:
        await executor.execute_one(
            ToolUseBlock(id="call", name="exec_command", input={"command": "x"}), context
        )
    assert caught.value is error
    assert context.command_stop_slot.receipt is receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("known", ["outcome", "not-started", "unknown"])
async def test_exec_cleanup_failure_keeps_known_fact_and_excluded_error(
    tmp_path: Path, known: str
) -> None:
    receipt = CommandStopReceipt("service", "stop")
    error = IrisCommandCleanupError(
        "cleanup failed",
        command_outcome=outcome(exit_code=7, receipt=receipt) if known == "outcome" else None,
        **({"started": False} if known == "not-started" else {}),
    )

    class SwallowError(ToolMiddleware):
        """若控制异常误入错误 middleware，就会被转换成普通成功。"""

        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            try:
                return await call_next()
            except Exception:
                return ToolResult(
                    tool_use_id="", tool_name="", content=[TextBlock(text="swallowed")]
                )

    executor, _tool = executor_for(FakeCommandService(error), middleware=[SwallowError()])
    context = ToolExecutionContext(workspace_root=tmp_path)
    call = ToolUseBlock(id="call", name="exec_command", input={"command": "x"})
    if known == "unknown":
        with pytest.raises(IrisCommandCleanupError) as caught:
            await executor.execute_one(call, context)
        assert caught.value is error
        return
    result = await executor.execute_one(call, context)
    assert context.command_stop_slot.cleanup_error is error
    assert result.error is not None
    assert result.error.code == ("COMMAND_FAILED" if known == "outcome" else "COMMAND_UNAVAILABLE")
    if known == "outcome":
        assert context.command_stop_slot.receipt is receipt
        assert "7" in result.model_content
    assert "cleanup_error" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_middleware_replacement_cannot_erase_command_stop_slot(tmp_path: Path) -> None:
    class ReplaceResult(ToolMiddleware):
        """用全新结果替换命令输出。"""

        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            await call_next()
            return ToolResult(tool_use_id="", tool_name="", content=[TextBlock(text="replacement")])

    receipt = CommandStopReceipt("service", "stop")
    service = FakeCommandService(
        outcome(CommandStatus.ENVIRONMENT_INTERRUPTED, None, receipt=receipt)
    )
    executor, _tool = executor_for(service, middleware=[ReplaceResult()])
    context = ToolExecutionContext(workspace_root=tmp_path)
    result = await executor.execute_one(
        ToolUseBlock(id="call", name="exec_command", input={"command": "x"}), context
    )
    assert result.model_content == "replacement"
    assert context.command_stop_slot.receipt is receipt


@pytest.mark.asyncio
async def test_unavailable_command_is_known_error_not_unknown(tmp_path: Path) -> None:
    executor, _tool = executor_for(
        FakeCommandService(IrisCommandError("image not ready", started=False))
    )
    result = await executor.execute_one(
        ToolUseBlock(id="call", name="exec_command", input={"command": "x"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.error is not None and result.error.code == "COMMAND_UNAVAILABLE"
    assert "image not ready" in result.model_content


@pytest.mark.asyncio
async def test_exec_confirmation_and_claim_precede_backend_execution(tmp_path: Path) -> None:
    """preflight 与 effect claim 未通过时，不进入命令服务。"""
    service = FakeCommandService(outcome())
    _allowed, tool = executor_for(service)
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)
    call = ToolUseBlock(id="call", name="exec_command", input={"command": "x"})
    prepared = executor.prepare_many([call], context).calls[0]
    assert prepared.human_request is not None
    assert not service.calls

    class RefuseClaim:
        """模拟 durable claim 拒绝，阻止副作用开始。"""

        def before_effect(self, prepared: Any) -> None:
            raise IrisCommandError("claim refused")

    with pytest.raises(IrisCommandError, match="claim refused"):
        await executor.execute_prepared(
            prepared, context, approved_tool_call_id="call", effect_guard=RefuseClaim()
        )
    assert not service.calls


@pytest.mark.asyncio
async def test_exec_artifact_keeps_bounded_model_text_and_stop_slot(tmp_path: Path) -> None:
    receipt = CommandStopReceipt("service", "stop")
    service = FakeCommandService(
        outcome(CommandStatus.ENVIRONMENT_INTERRUPTED, None, receipt=receipt, stdout="x" * 60_000)
    )
    executor, tool = executor_for(service)
    context = ToolExecutionContext(workspace_root=tmp_path)
    result = await executor.execute_one(
        ToolUseBlock(id="call", name="exec_command", input={"command": "x"}), context
    )
    assert result.artifact is not None
    assert "x" * 60_000 in result.artifact.path.read_text(encoding="utf-8")
    assert len(result.model_content) <= tool.definition.max_result_chars
    assert "stdout" not in result.data
    assert context.command_stop_slot.receipt is receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", [ExecCommandTool, RunPythonTool])
@pytest.mark.parametrize("failed", [False, True])
async def test_backend_and_artifact_truncation_preserve_status_and_final_diagnostic(
    tmp_path: Path, tool_type: type[ExecCommandTool] | type[RunPythonTool], failed: bool
) -> None:
    """两条流先经真实字节头尾缓存，再经 executor 字符预览，错误尾部仍可见。"""
    buffer = OutputBuffer(limit=4096)
    buffer.append(1, b"STDOUT_HEAD\n" + b"o" * 20_000 + b"\nSTDOUT_TAIL")
    buffer.append(2, b"STDERR_HEAD\n" + b"e" * 20_000 + b"\nFINAL_TRACEBACK")
    response = replace(
        outcome(exit_code=7 if failed else 0),
        stdout=buffer.stdout,
        stderr=buffer.stderr,
        output_stats=buffer.stats,
    )
    registry = ToolRegistry()
    tool = tool_type(binding(FakeCommandService(response)))
    tool.definition = tool.definition.model_copy(
        update={"max_result_chars": 1200, "preview_chars": 1000}
    )
    registry.register(tool)
    executor = ToolExecutor(
        registry, permission_policy=DefaultPermissionPolicy(execute_mode="allow")
    )
    argument = "code" if tool.name == "run_python" else "command"
    result = await executor.execute_one(
        ToolUseBlock(id="two-limits", name=tool.name, input={argument: "x"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.is_error is failed
    assert len(result.model_content) <= 1200
    assert "status: exited" in result.model_content
    assert f"exit_code: {7 if failed else 0}" in result.model_content
    assert "FINAL_TRACEBACK" in result.model_content
    assert result.data["output_truncated"] is True
    assert result.data["output_stats"]["truncation_reasons"] == ["byte_limit"]
    assert "后续未知" not in result.model_content
    assert result.artifact is not None
    saved = result.artifact.path.read_text(encoding="utf-8")
    for marker in ("STDOUT_HEAD", "STDOUT_TAIL", "STDERR_HEAD", "FINAL_TRACEBACK"):
        assert marker in saved
    assert "FINAL_TRACEBACK" in result.artifact.preview
    assert "o" * 20_000 not in saved


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before", "after", "recovery"])
async def test_cleanup_control_error_from_middleware_is_not_rewritten(
    tmp_path: Path, stage: str
) -> None:
    error = IrisCommandCleanupError("cleanup remains pending")

    class CleanupMiddleware(ToolMiddleware):
        """在当前阶段报告未完成资源清理。"""

        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            if stage == "before":
                raise error
            try:
                result = await call_next()
            except IrisCommandError:
                raise error from None
            if stage == "after":
                raise error
            return result

    service = FakeCommandService(
        IrisCommandError("body failure") if stage == "recovery" else outcome()
    )
    executor, _tool = executor_for(service, middleware=[CleanupMiddleware()])
    with pytest.raises(IrisCommandCleanupError) as caught:
        await executor.execute_one(
            ToolUseBlock(id="call", name="exec_command", input={"command": "x"}),
            ToolExecutionContext(workspace_root=tmp_path),
        )
    assert caught.value is error
