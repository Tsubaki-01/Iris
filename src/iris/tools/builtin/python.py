"""显式 Python 源码工具；借用与 shell 相同的命令环境和生命周期。"""

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...command.models import CommandMode, PythonCode
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


class RunPythonInput(BaseModel):
    """只接收完整源码、起始目录和可缩短的前台期限。"""

    model_config = ConfigDict(extra="forbid")
    code: str = Field(
        min_length=1, pattern=r"\S", description="要执行的完整 Python 代码；用 print 输出结果。"
    )
    cwd: str = Field(default=".", description="当前 Agent workspace 内的起始目录。")
    timeout_seconds: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="可选前台期限，只能缩短 root 期限。"
    )


class RunPythonTool(BaseTool):
    """显式执行 Python 源码，不自动安装依赖或创建独立模型循环。"""

    timeout_owner = ToolTimeoutOwner.TOOL

    def __init__(self, binding: CommandBinding) -> None:
        """借用 root 选定环境，描述本次真正使用的解释器来源。"""
        self.binding = binding
        environment = binding.environment
        interpreter = (
            "解释器与 Iris 当前进程相同。"
            if environment.mode is CommandMode.NATIVE
            else "使用所选镜像内的 Python。"
        )
        self.definition = ToolDefinition(
            name="run_python",
            description=(
                f"在 {environment.mode.value} 环境执行完整 Python 代码，"
                f"系统 {environment.command_os}。"
                f"{interpreter}每次新进程，变量和导入状态不跨调用延续；"
                "工作文件和已准备依赖沿用当前环境。UTF-8、非缓冲输出，无交互 stdin 或 TTY。"
                "用 print 输出结果，末尾表达式不自动显示；不自动安装依赖。"
                f"{command_boundary_description(environment.mode)}"
                "单次超时只停止本次执行，停止不回滚工作文件。"
            ),
            input_schema=schema_from_pydantic_model(RunPythonInput),
            capabilities={ToolCapability.EXECUTE},
            group="exec",
            context_retention="keep",
            preview_mode="head_tail",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回 Python 参数的唯一解析模型。"""
        return RunPythonInput

    def validate_input(self, params: dict[str, Any]) -> RunPythonInput:
        """解析原始输入，不在宿主编译或改写用户代码。"""
        try:
            return RunPythonInput.model_validate(params)
        except ValidationError as error:
            raise IrisToolValidationError(
                "run_python 参数校验失败", errors=error.errors()
            ) from error

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """把已解析 Python 载荷交给共享命令入口。"""
        inputs = cast(RunPythonInput, params)
        return await run_command(
            PythonCode(inputs.code),
            binding=self.binding,
            cwd=inputs.cwd,
            timeout_seconds=inputs.timeout_seconds,
            context=context,
            tool_name=self.name,
        )


__all__ = ["RunPythonInput", "RunPythonTool"]
