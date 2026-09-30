"""主模型读取目标与提交结果申报的普通工具。"""

from __future__ import annotations

import json
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from ..exceptions import IrisGoalError
from ..message import TextBlock
from ..tools import (
    BaseTool,
    ToolCapability,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
    schema_from_pydantic_model,
)
from .models import GoalReport, GoalView
from .service import GoalService

_VIEW_ADAPTER = TypeAdapter(GoalView)


class _GetGoalInput(BaseModel):
    """目标读取不接受模型指定会话或目标身份。"""

    model_config = ConfigDict(extra="forbid")


class _GoalTool(BaseTool):
    """共享工具 schema 和一次输入解析，不接管执行或结算。"""

    _name: ClassVar[str]
    _description: ClassVar[str]
    _input_type: ClassVar[type[BaseModel]]

    def __init__(self, service: GoalService) -> None:
        """绑定目标服务并声明普通、可见且保留历史的工具。"""
        self.service = service
        self.definition = ToolDefinition(
            name=self._name,
            description=self._description,
            input_schema=schema_from_pydantic_model(self._input_type),
            capabilities={ToolCapability.READ},
            group="goal",
            deferred=False,
            context_retention="keep",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """声明工具执行器的唯一原始输入解析模型。"""
        return self._input_type

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """解析原始工具参数一次，下游只消费 typed model。"""
        return self._input_type.model_validate(params)


class GetGoalTool(_GoalTool):
    """读取所在 session 的当前目标与执行状态。"""

    _name = "get_goal"
    _description = "读取当前目标、版本、状态、轮数及是否等待人工输入；不创建或恢复目标。"
    _input_type = _GetGoalInput

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """返回统一 GoalView，读取不会触发补结算或自动继续。"""
        payload = _VIEW_ADAPTER.dump_python(self.service.get_view(context.session_id), mode="json")
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[
                TextBlock(text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            ],
            data=payload,
        )


class ReportGoalTool(_GoalTool):
    """提交本轮的结构化目标申报，由 Run 正常结束后的结算决定状态。"""

    _name = "report_goal"
    _description = (
        "申报本轮目标结果：complete 已完成、blocked 无法继续、continue 继续或撤回旧申报。"
        "使用最新 goal_id/revision 并说明证据；单独调用，不与工作工具同一步。"
        "后续有新工作或新输入时须重新申报。申报不立即完成目标，正常结束后才结算。"
    )
    _input_type = GoalReport

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """使用真实执行 run_id 核对报告，输出交给原工具提交路径。"""
        try:
            report = self.service.report(
                cast(str, context.metadata.get("run_id", "")),
                cast(GoalReport, params),
            )
        except IrisGoalError as exc:
            return ToolResult(
                tool_use_id=context.call_id,
                tool_name=self.name,
                is_error=True,
                error=ToolErrorInfo(
                    code=exc.runtime_code, message=exc.message, details=exc.context
                ),
            )
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[
                TextBlock(text="已记录本轮申报，正常结束后结算；新工作或新输入后请重新申报。")
            ],
            data={"goal_report": report.model_dump(mode="json")},
        )


__all__ = ["GetGoalTool", "ReportGoalTool"]
