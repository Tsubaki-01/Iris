"""显式 shell 命令工具；与 Python 入口共用命令调用及结果投影。"""

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...command.models import ShellCommand
from ...command.service import CommandBinding
from ...exceptions import IrisToolValidationError
from ..base import (
    BaseTool,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    ToolTimeoutOwner,
)
from ..schema import schema_from_pydantic_model
from ._command import command_boundary_description, run_command


class ExecCommandInput(BaseModel):
    """模型可决定命令、起始目录和更短的前台业务期限。"""

    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, pattern=r"\S", description="使用当前环境 shell 语法的命令。")
    cwd: str = Field(default=".", description="当前 Agent workspace 内的起始目录。")
    timeout_seconds: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="可选前台期限，只能缩短 root 期限。"
    )


class ExecCommandTool(BaseTool):
    """借用 root 已装配的命令服务，不拥有环境准备、停止或关闭。"""

    timeout_owner = ToolTimeoutOwner.TOOL

    def __init__(self, binding: CommandBinding) -> None:
        """绑定同源配置、环境说明和已有服务。"""
        self.binding = binding
        environment = binding.environment
        self.definition = ToolDefinition(
            name="exec_command",
            description=(
                f"在 {environment.mode.value} 环境执行命令："
                f"系统 {environment.command_os}，shell {environment.command_shell}。"
                "每次新 shell；cd/export 不延续。无交互 stdin、无 TTY。"
                f"{command_boundary_description(environment.mode)}"
                "单命令超时只停止本次命令，停止不回滚工作文件。"
            ),
            input_schema=schema_from_pydantic_model(ExecCommandInput),
            capabilities={ToolCapability.EXECUTE},
            group="exec",
            context_retention="keep",
            preview_mode="head_tail",
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
        """把已解析 shell 载荷交给共享命令入口。"""
        inputs = cast(ExecCommandInput, params)
        return await run_command(
            ShellCommand(inputs.command),
            binding=self.binding,
            cwd=inputs.cwd,
            timeout_seconds=inputs.timeout_seconds,
            context=context,
            tool_name=self.name,
        )


__all__ = ["ExecCommandInput", "ExecCommandTool"]
