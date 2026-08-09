"""Runtime 专属数据模型。

本模块只定义 activation inner engine 的输入、cursor 与结果模型。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic_core import PydanticSerializationError

from ..hitl import HumanInteraction
from ..lifecycle.models import (
    RunErrorInfo,
    RuntimeExecutionOptions,
    validate_json_safe,
)
from ..message import Msg, ToolUseBlock
from ..tools import ToolResult


class _FrozenRuntimeModel(BaseModel):
    """Phase 2 runtime facts 的不可变模型基类。"""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class RuntimeActivationOutcome(StrEnum):
    """一次 activation inner engine 的结束事实。"""

    COMPLETED = "completed"
    SUSPENDED = "suspended"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class RuntimeCursor(_FrozenRuntimeModel):
    """可持久化的 inner engine 精确位置。"""

    position: Literal["before_model", "tool_batch", "outcome_ready"]
    step_index: int = Field(ge=0)
    next_tool_index: int = Field(default=0, ge=0)
    tool_calls: tuple[ToolUseBlock, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    assistant_message: Msg | None = None
    read_state: dict[str, Any] | None = None

    @field_validator("read_state")
    @classmethod
    def _validate_read_state(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return validate_json_safe(value, field_name="read_state")

    @model_validator(mode="after")
    def _validate_position(self) -> RuntimeCursor:
        call_ids = [call.id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("同一 model step 的 tool call ID 不能重复")
        if self.next_tool_index > len(self.tool_calls):
            raise ValueError("next_tool_index 不能超过 tool batch 长度")
        if len(self.tool_results) != self.next_tool_index:
            raise ValueError("tool_results 必须精确覆盖已提交调用前缀")
        committed_calls = self.tool_calls[: self.next_tool_index]
        for call, result in zip(committed_calls, self.tool_results, strict=True):
            if result.tool_use_id != call.id or result.tool_name != call.name:
                raise ValueError("tool result identity 与已提交调用前缀不匹配")

        if self.position == "before_model":
            if (
                self.next_tool_index != 0
                or self.tool_calls
                or self.tool_results
                or self.assistant_message is not None
            ):
                raise ValueError("before_model cursor 不能包含 model/tool 结果")
        elif self.position == "tool_batch":
            if self.assistant_message is None or not self.tool_calls:
                raise ValueError("tool_batch cursor 必须包含 assistant message 和 tool calls")
            if self.next_tool_index >= len(self.tool_calls):
                raise ValueError("tool_batch cursor 必须保留至少一个未提交调用")
        else:
            if self.assistant_message is None:
                raise ValueError("outcome_ready cursor 必须包含 assistant message")
            if self.next_tool_index != len(self.tool_calls):
                raise ValueError("outcome_ready cursor 不能包含未提交工具调用")
        try:
            serialized = self.model_dump(mode="json")
        except (PydanticSerializationError, TypeError, ValueError) as exc:
            raise ValueError("runtime cursor 必须是 JSON-safe") from exc
        validate_json_safe(serialized, field_name="runtime cursor")
        return self


class RuntimeApprovedToolCall(_FrozenRuntimeModel):
    """Lifecycle owner 提供给 engine 的精确权限批准投影。"""

    interaction_id: str
    tool_call_id: str
    tool_name: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("interaction_id", "tool_call_id", "tool_name")
    @classmethod
    def _validate_text(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized


class RuntimeActivationInput(_FrozenRuntimeModel):
    """一次 start、resume 或 recover activation 的 engine 输入。"""

    run_id: str
    activation_id: str
    session_id: str
    kind: Literal["start", "resume", "recover"]
    input: str | None
    cursor: RuntimeCursor
    options: RuntimeExecutionOptions
    interaction_projection: ToolResult | RuntimeApprovedToolCall | None = None

    @field_validator("run_id", "activation_id", "session_id")
    @classmethod
    def _validate_identity(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_activation_shape(self) -> RuntimeActivationInput:
        if self.kind == "start":
            if self.input is None or not self.input.strip():
                raise ValueError("start activation 必须包含非空 input")
            if self.cursor.position != "before_model" or self.cursor.step_index != 0:
                raise ValueError("start activation 必须从 step 0 before_model 开始")
            if self.interaction_projection is not None:
                raise ValueError("start activation 不能携带 interaction projection")
        elif self.kind == "resume":
            if self.input is not None:
                raise ValueError("resume activation 不能携带用户 input")
        elif self.cursor.position == "before_model" and self.cursor.step_index == 0:
            if self.input is None or not self.input.strip():
                raise ValueError("初始 recover activation 必须包含非空 input")
        elif self.input is not None:
            raise ValueError("非初始 recover activation 不能重复携带用户 input")
        if self.interaction_projection is not None:
            if self.cursor.position != "tool_batch":
                raise ValueError("interaction projection 必须绑定 tool_batch cursor")
            try:
                projection = self.interaction_projection.model_dump(mode="json")
            except (PydanticSerializationError, TypeError, ValueError) as exc:
                raise ValueError("interaction projection 必须是 JSON-safe") from exc
            validate_json_safe(projection, field_name="interaction projection")
        return self


class RuntimeActivationResult(_FrozenRuntimeModel):
    """Inner engine 返回给 lifecycle owner 的 activation 事实。"""

    outcome: RuntimeActivationOutcome
    cursor: RuntimeCursor
    assistant_message: Msg | None = None
    suspension: HumanInteraction | None = None
    error: RunErrorInfo | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> RuntimeActivationResult:
        is_suspended = self.outcome is RuntimeActivationOutcome.SUSPENDED
        if is_suspended != (self.suspension is not None):
            raise ValueError("suspended outcome 必须且只能包含 suspension")
        if (
            self.outcome
            in {
                RuntimeActivationOutcome.FAILED,
                RuntimeActivationOutcome.OUTCOME_UNKNOWN,
            }
            and self.error is None
        ):
            raise ValueError("failed/outcome_unknown 必须包含结构化错误")
        if self.outcome is RuntimeActivationOutcome.COMPLETED and self.error is not None:
            raise ValueError("completed outcome 不能包含错误")
        return self


__all__ = [
    "RuntimeActivationInput",
    "RuntimeActivationOutcome",
    "RuntimeActivationResult",
    "RuntimeApprovedToolCall",
    "RuntimeCursor",
]
