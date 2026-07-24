"""面向人工交互的内置工具定义。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...exceptions import IrisHITLError
from ...hitl.models import QuestionPrompt
from ..base import BaseTool, ToolDefinition, ToolExecutionContext, ToolResult
from ..schema import schema_from_pydantic_model


class AskQuestionInput(BaseModel):
    """请求人类回答的单个问题。"""

    question: str
    options: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: list[str]) -> list[str]:
        normalized = [option.strip() for option in value]
        if any(not option for option in normalized):
            raise ValueError("options 不能包含空值")
        if len(normalized) != len(set(normalized)):
            raise ValueError("options 不能包含重复值")
        return normalized


class AskQuestionTool(BaseTool):
    """将模型问题标识为需要 runtime 创建的 human interaction。"""

    def __init__(self) -> None:
        self.definition = self._definition()

    @staticmethod
    def _definition() -> ToolDefinition:
        return ToolDefinition(
            name="ask_question",
            description="向人类提出一个问题，并通过 HITL 返回回答。",
            input_schema=schema_from_pydantic_model(AskQuestionInput),
            capabilities=set(),
            group="human",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return AskQuestionInput

    def validate_input(self, params: dict[str, Any]) -> AskQuestionInput:
        return AskQuestionInput.model_validate(params)

    def build_interaction_prompt(
        self,
        *,
        params: AskQuestionInput | dict[str, Any],
    ) -> QuestionPrompt:
        """将已验证输入转换为向人展示的问题。"""
        input_data = AskQuestionInput.model_validate(params)
        return QuestionPrompt(
            question=input_data.question,
            options=input_data.options,
        )

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """拒绝绕过 runtime 直接执行 human question。"""
        del context
        AskQuestionInput.model_validate(params)
        raise IrisHITLError("HITL_PROTOCOL_ERROR: ask_question 必须由 runtime 处理")


__all__ = ["AskQuestionInput", "AskQuestionTool"]
