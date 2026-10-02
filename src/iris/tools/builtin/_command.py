"""Shell 与 Python 工具共用的请求边界、停止事实交接与结果投影。"""

from ...command.models import (
    CommandMode,
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
    PythonCode,
    ShellCommand,
)
from ...command.service import CommandBinding
from ...exceptions import (
    IrisCommandCleanupError,
    IrisCommandError,
    IrisToolOutcomeUnknownError,
    IrisToolValidationError,
)
from ...message import TextBlock
from ..base import ToolErrorInfo, ToolExecutionContext, ToolResult
from ..permissions import WorkspacePolicy


def command_boundary_description(mode: CommandMode) -> str:
    """返回两种工具共同承担的命令环境边界。"""
    if mode is CommandMode.NATIVE:
        return "代码或命令以宿主用户权限运行；cwd 仅指定起始目录，不限制其他宿主路径访问。"
    return (
        "所有 session/child 共用 root 的 /workspace 挂载与容器文件层。"
        "child workspace 只确定默认 cwd；child writes=deny 只限制原生文件工具，"
        "不保证 Docker 执行只读，写入取决于 root 挂载。"
        "整体停止后依赖文件保留，后台服务不会自动恢复。"
    )


async def run_command(
    payload: ShellCommand | PythonCode,
    *,
    binding: CommandBinding,
    cwd: str,
    timeout_seconds: float | None,
    context: ToolExecutionContext,
    tool_name: str,
) -> ToolResult:
    """消费已解析载荷，执行前解析目录/期限，返回前交接停止事实。"""
    resolved_cwd = WorkspacePolicy().resolve_path(cwd, workspace_root=context.workspace_root)
    if not resolved_cwd.is_dir():
        raise IrisToolValidationError(f"{tool_name} cwd 必须是已存在的目录", cwd=cwd)
    limits = [binding.config.timeout_seconds]
    if timeout_seconds is not None:
        limits.append(timeout_seconds)
    if context.tool_timeout_seconds is not None:
        limits.append(context.tool_timeout_seconds)
    request = CommandRequest(context.call_id, payload, resolved_cwd, min(limits))
    scope = CommandScope(str(context.metadata.get("run_id", "")), context.session_id)
    try:
        outcome = await binding.service.execute(scope, request)
    except IrisToolOutcomeUnknownError as error:
        context.command_stop_slot.record(receipt=error.stop_receipt)
        raise
    except IrisCommandCleanupError as error:
        if error.command_outcome is not None:
            context.command_stop_slot.record(cleanup_error=error)
            return _known_result(error.command_outcome, context, tool_name)
        if error.context.get("started") is False:
            context.command_stop_slot.record(cleanup_error=error)
            return _unavailable(error, binding, context, tool_name)
        raise
    except IrisCommandError as error:
        if error.context.get("started") is False:
            return _unavailable(error, binding, context, tool_name)
        raise
    return _known_result(outcome, context, tool_name)


def _known_result(
    outcome: CommandOutcome, context: ToolExecutionContext, tool_name: str
) -> ToolResult:
    """状态和采集事实置首，stderr 置尾，供统一 artifact 层产生头尾预览。"""
    context.command_stop_slot.record(status=outcome.status, receipt=outcome.stop_receipt)
    descriptions = {
        CommandStatus.EXITED: f"前台命令已退出，退出码 {outcome.exit_code}",
        CommandStatus.TIMED_OUT: "命令业务期限已到，本次命令已停止",
        CommandStatus.CANCELLED: "调用已中断，后端已确认本次停止范围",
        CommandStatus.ENVIRONMENT_INTERRUPTED: "共享命令环境已停止，本次命令被连带中断",
    }
    stats = outcome.output_stats
    reasons = [
        reason
        for reason in ("byte_limit", "drain_timeout", "stream_error", "stream_closed")
        if reason in stats.truncation_reasons
    ]
    text = (
        f"{descriptions[outcome.status]}\nmode: {outcome.mode.value}\n"
        f"status: {outcome.status.value}\nexit_code: {outcome.exit_code}\n"
        f"cwd: {outcome.cwd}\nduration_seconds: {outcome.duration_seconds:.3f}\n"
        f"stdout_bytes: {stats.stdout_bytes}, retained: {stats.stdout_retained_bytes}\n"
        f"stderr_bytes: {stats.stderr_bytes}, retained: {stats.stderr_retained_bytes}\n"
        f"truncation_reasons: {', '.join(reasons) if reasons else 'none'}"
    )
    if stats.truncation_reasons - {"byte_limit"}:
        text += (
            f"\n输出采集不完整：已读取 {stats.stdout_bytes + stats.stderr_bytes} bytes，后续未知。"
        )
    elif outcome.output_truncated:
        text += "\n输出达到保留额度，仅保留各流头尾；省略部分未保存。"
    if outcome.stdout:
        text += f"\n\nstdout:\n{outcome.stdout}"
    if outcome.stderr:
        text += f"\n\nstderr:\n{outcome.stderr}"
    error_code = {
        CommandStatus.TIMED_OUT: "COMMAND_TIMEOUT",
        CommandStatus.CANCELLED: "COMMAND_CANCELLED",
        CommandStatus.ENVIRONMENT_INTERRUPTED: "COMMAND_ENVIRONMENT_INTERRUPTED",
    }.get(outcome.status)
    if outcome.status is CommandStatus.EXITED and outcome.exit_code != 0:
        error_code = "COMMAND_FAILED"
    return ToolResult(
        tool_use_id=context.call_id,
        tool_name=tool_name,
        content=[] if error_code else [TextBlock(text=text)],
        is_error=error_code is not None,
        error=ToolErrorInfo(code=error_code, message=text) if error_code else None,
        data={
            "mode": outcome.mode.value,
            "status": outcome.status.value,
            "exit_code": outcome.exit_code,
            "cwd": outcome.cwd,
            "duration_seconds": outcome.duration_seconds,
            "output_truncated": outcome.output_truncated,
            "output_stats": {
                "stdout_bytes": stats.stdout_bytes,
                "stderr_bytes": stats.stderr_bytes,
                "stdout_retained_bytes": stats.stdout_retained_bytes,
                "stderr_retained_bytes": stats.stderr_retained_bytes,
                "truncation_reasons": reasons,
            },
        },
    )


def _unavailable(
    error: IrisCommandError, binding: CommandBinding, context: ToolExecutionContext, tool_name: str
) -> ToolResult:
    """明确未启动的请求不生成已执行 outcome 或采集统计。"""
    return ToolResult(
        tool_use_id=context.call_id,
        tool_name=tool_name,
        is_error=True,
        error=ToolErrorInfo(code="COMMAND_UNAVAILABLE", message=f"命令尚未启动：{error}"),
        data={"mode": binding.environment.mode.value, "started": False},
    )
