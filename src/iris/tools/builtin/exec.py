"""显式命令工具：参数边界、业务期限与后端事实投影。"""

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...command.models import (
    CommandMode,
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
)
from ...command.service import CommandBinding
from ...exceptions import (
    IrisCommandCleanupError,
    IrisCommandError,
    IrisToolOutcomeUnknownError,
    IrisToolValidationError,
)
from ...message import TextBlock
from ..base import (
    BaseTool,
    ToolCapability,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
    ToolTimeoutOwner,
)
from ..permissions import WorkspacePolicy
from ..schema import schema_from_pydantic_model


class ExecCommandInput(BaseModel):
    """模型可决定命令、起始目录和更短的前台业务期限。"""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, pattern=r"\S", description="使用当前环境 shell 语法的命令。")
    cwd: str = Field(default=".", description="当前 Agent workspace 内的起始目录。")
    timeout_seconds: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="可选前台期限，只能缩短 root 期限。"
    )


class ExecCommandTool(BaseTool):
    """借用 root 已装配的执行服务，不拥有环境准备、停止或关闭。"""

    timeout_owner = ToolTimeoutOwner.TOOL

    def __init__(self, binding: CommandBinding) -> None:
        """绑定同源配置、环境说明和已有服务。"""
        self.binding = binding
        self._workspace_policy = WorkspacePolicy()
        environment = binding.environment
        boundary = (
            "命令以宿主用户权限运行；cwd 仅指定起始目录，不限制其他宿主路径访问。"
            if environment.mode is CommandMode.NATIVE
            else "所有 session/child 共用 root 的 /workspace 挂载与容器文件层。"
            "child workspace 只确定默认 cwd；child writes=deny 只限制原生文件工具，"
            "不保证 Docker 命令只读，命令写入取决于 root 挂载。"
            "整体停止后依赖文件保留，后台服务不会自动恢复。"
        )
        self.definition = ToolDefinition(
            name="exec_command",
            description=(
                f"在 {environment.mode.value} 环境执行命令："
                f"系统 {environment.command_os}，shell {environment.command_shell}。"
                "每次新 shell；cd/export 不延续。无交互 stdin、无 TTY。"
                f"{boundary} 单命令超时只停止本次命令，停止不回滚工作文件。"
            ),
            input_schema=schema_from_pydantic_model(ExecCommandInput),
            capabilities={ToolCapability.EXECUTE},
            group="exec",
            context_retention="keep",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回唯一的原始参数解析模型。"""
        return ExecCommandInput

    def validate_input(self, params: dict[str, Any]) -> ExecCommandInput:
        """在模型参数首次进入时解析一次，不改写 shell 命令内容。"""
        try:
            return ExecCommandInput.model_validate(params)
        except ValidationError as error:
            raise IrisToolValidationError(
                "exec_command 参数校验失败", errors=error.errors()
            ) from error

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """投影已授权请求，并先保存停止事实再返回工具结果。"""
        inputs = cast(ExecCommandInput, params)
        cwd = self._workspace_policy.resolve_path(inputs.cwd, workspace_root=context.workspace_root)
        if not cwd.is_dir():
            raise IrisToolValidationError("exec_command cwd 必须是已存在的目录", cwd=inputs.cwd)
        limits = [self.binding.config.timeout_seconds]
        if inputs.timeout_seconds is not None:
            limits.append(inputs.timeout_seconds)
        if context.tool_timeout_seconds is not None:
            limits.append(context.tool_timeout_seconds)
        request = CommandRequest(context.call_id, inputs.command, cwd, min(limits))
        scope = CommandScope(str(context.metadata.get("run_id", "")), context.session_id)
        try:
            result = await self.binding.service.execute(scope, request)
        except IrisToolOutcomeUnknownError as error:
            context.command_stop_slot.receipt = error.stop_receipt
            raise
        except IrisCommandCleanupError as error:
            if error.command_outcome is not None:
                context.command_stop_slot.cleanup_error = error
                return self._known_result(error.command_outcome, context)
            if error.context.get("started") is False:
                context.command_stop_slot.cleanup_error = error
                return self._unavailable(error, context)
            raise
        except IrisCommandError as error:
            if error.context.get("started") is False:
                return self._unavailable(error, context)
            raise
        return self._known_result(result, context)

    def _known_result(self, outcome: CommandOutcome, context: ToolExecutionContext) -> ToolResult:
        context.command_stop_slot.receipt = outcome.stop_receipt
        descriptions = {
            CommandStatus.EXITED: f"前台命令已退出，退出码 {outcome.exit_code}",
            CommandStatus.TIMED_OUT: "命令业务期限已到，本次命令已停止",
            CommandStatus.CANCELLED: "调用已中断，后端已确认本次停止范围",
            CommandStatus.ENVIRONMENT_INTERRUPTED: "共享命令环境已停止，本次命令被连带中断",
        }
        text = (
            f"{descriptions[outcome.status]}\nmode: {outcome.mode.value}\n"
            f"cwd: {outcome.cwd}\nduration_seconds: {outcome.duration_seconds:.3f}"
        )
        if outcome.stdout:
            text += f"\n\nstdout:\n{outcome.stdout}"
        if outcome.stderr:
            text += f"\n\nstderr:\n{outcome.stderr}"
        if outcome.output_truncated:
            text += "\n\n输出已截断：达到输出额度或前台退出后的排空期限。"
        error_code = {
            CommandStatus.TIMED_OUT: "COMMAND_TIMEOUT",
            CommandStatus.CANCELLED: "COMMAND_CANCELLED",
            CommandStatus.ENVIRONMENT_INTERRUPTED: "COMMAND_ENVIRONMENT_INTERRUPTED",
        }.get(outcome.status)
        if outcome.status is CommandStatus.EXITED and outcome.exit_code != 0:
            error_code = "COMMAND_FAILED"
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
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
            },
        )

    def _unavailable(self, error: IrisCommandError, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            is_error=True,
            error=ToolErrorInfo(code="COMMAND_UNAVAILABLE", message=f"命令尚未启动：{error}"),
            data={"mode": self.binding.environment.mode.value, "started": False},
        )


__all__ = ["ExecCommandInput", "ExecCommandTool"]
