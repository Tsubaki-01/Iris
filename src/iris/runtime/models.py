"""Runtime 专属数据模型。

本模块只定义 runtime 阶段共享的配置、快照和结果模型，不执行 provider、
工具、memory 或 session 逻辑。
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic_core import PydanticSerializationError

from ..context import ContextBuildOutput
from ..hitl import HumanInteraction
from ..lifecycle.models import (
    RunErrorInfo,
    RuntimeExecutionOptions,
    ToolErrorPolicy,
    validate_json_safe,
)
from ..memory import MemoryQuery, MemorySearchResult
from ..message import Conversation, LLMRequest, Msg, ToolUseBlock
from ..tools import ToolResult

RuntimeErrorSource = Literal[
    "config",
    "context",
    "provider",
    "tool",
    "memory",
    "session",
    "runtime",
]


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
        elif self.input is not None:
            raise ValueError("resume/recover activation 不能重复携带用户 input")
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
        if self.outcome in {
            RuntimeActivationOutcome.FAILED,
            RuntimeActivationOutcome.OUTCOME_UNKNOWN,
        } and self.error is None:
            raise ValueError("failed/outcome_unknown 必须包含结构化错误")
        if self.outcome is RuntimeActivationOutcome.COMPLETED and self.error is not None:
            raise ValueError("completed outcome 不能包含错误")
        return self


def _new_run_id() -> str:
    """生成一次 runtime run 的本地唯一 ID。"""
    return f"run_{uuid.uuid4().hex[:12]}"


class RuntimeStatus(StrEnum):
    """Runtime 单轮或 loop 的结束状态。"""

    OK = "ok"
    ERROR = "error"
    MAX_STEPS = "max_steps"
    WAITING_HUMAN = "waiting_human"


class BoundedLoopOptions(BaseModel):
    """有界 loop 的基础控制参数。"""

    max_steps: int = Field(default=20, gt=0)
    tool_error_policy: ToolErrorPolicy = ToolErrorPolicy.RETURN_TO_MODEL

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class RuntimeErrorInfo(BaseModel):
    """Runtime 对外返回的结构化错误信息。"""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source: RuntimeErrorSource
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ProviderResponseSnapshot(BaseModel):
    """可持久化的 provider 响应最小快照。"""

    provider: str
    response_id: str = ""
    model: str = ""
    content: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning: str = ""

    model_config = ConfigDict(extra="forbid")


class RuntimeOptionsSnapshot(BaseModel):
    """HITL 恢复所需的调用级选项快照。"""

    options: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class RuntimeContinuationClaim(BaseModel):
    """恢复后 continuation 的 fail-closed 执行标记。"""

    kind: Literal["tool", "loop"]
    next_tool_index: int = Field(ge=0)
    tool_call_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_subject(self) -> RuntimeContinuationClaim:
        if self.kind == "tool" and not self.tool_call_id:
            raise ValueError("tool continuation claim 必须包含 tool_call_id")
        if self.kind == "loop" and self.tool_call_id is not None:
            raise ValueError("loop continuation claim 不能包含 tool_call_id")
        return self


class RuntimeHITLCheckpoint(BaseModel):
    """第一次人工等待前保存的 runtime 恢复快照。"""

    checkpoint_version: Literal[2] = 2
    run_mode: Literal["turn", "loop"]
    agent_name: str
    session_id: str
    run_id: str
    step_index: int = Field(ge=0)
    runtime_options: RuntimeOptionsSnapshot
    assistant_message: dict[str, Any]
    provider_response: ProviderResponseSnapshot
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    next_tool_index: int = Field(ge=0)
    batch_results: list[dict[str, Any]] = Field(default_factory=list)
    all_tool_results: list[dict[str, Any]] = Field(default_factory=list)
    read_state: dict[str, Any] | None = None
    pending_result: dict[str, Any] | None = None
    continuation_claim: RuntimeContinuationClaim | None = None
    continuation_complete: bool = False

    model_config = ConfigDict(extra="forbid")


class RuntimeTurnInput(BaseModel):
    """单次 runtime 调用的用户输入包。"""

    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ToolResultCommit(BaseModel):
    """一次工具结果提交的结果快照。

    committer 会同时生成面向程序、模型和 session 的三份视图，调用方按使用场景选择
    对应字段，而不需要重新转换工具结果。

    Attributes:
        results (list[ToolResult]): 程序侧可读取的结构化工具执行结果。
        messages (list[Msg]): 可回灌给模型的 tool result 消息。
        events (list[dict[str, Any]]): 已写入 session store 的 JSON-safe 工具事件快照。
    """

    results: list[ToolResult] = Field(default_factory=list)
    messages: list[Msg] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Runstate(BaseModel):
    """Runtime 内部测试和调试用的单步运行快照。

    Attributes:
        session_id (str): 当前运行绑定的会话标识，用于关联 session history。
        run_id (str): 当前 runtime run 的唯一标识，用于串联日志、事件和调试信息。
        step_index (int): 当前 loop 步骤序号，从 0 开始。
        context_output (ContextBuildOutput): 本步请求使用的 context 构建结果。
        history (list[Msg]): 从 session 层读取的历史消息，不包含本轮 context 注入。
        current_input (Msg | None): 本步新增输入；loop 后续步骤可为空。
        conversation (Conversation): assembler 生成的完整 provider 请求消息序列。
        tools_schema (list[dict[str, Any]]): 本步挂载到请求上的工具 schema 快照。
        request (LLMRequest): 本步发送给 provider 的 provider-neutral 请求。
        metadata (dict[str, Any]): 仅用于调试和追踪的运行态附加信息。
    """

    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    context_output: ContextBuildOutput
    history: list[Msg] = Field(default_factory=list)
    current_input: Msg | None = None
    conversation: Conversation
    tools_schema: list[dict[str, Any]] = Field(default_factory=list)
    request: LLMRequest
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RuntimeOptions(BaseModel):
    """Runtime 调用级选项。

    Attributes:
        session_id (str): 本次调用使用的会话标识，默认使用 `"default"`。
        run_id (str): 本次调用的唯一运行标识，默认自动生成。
        include_tools (bool): 是否在 provider 请求中包含可用工具 schema。
        request_options (dict[str, Any]): 透传给 `LLMRequest` 的请求级覆盖项， provider 专属选项。
        metadata (dict[str, Any]): 运行态追踪信息，不直接进入 prompt。
        memory_query (MemoryQuery | None): 显式触发 memory recall 的查询条件。
        memory_results (list[MemorySearchResult] | None): 调用方预先提供的 memory 结果。
        memory_max_chars (int): memory 注入 context 时允许使用的字符预算。
        loop (BoundedLoopOptions): 有界 loop 的步数和工具错误处理配置。
    """

    session_id: str = "default"
    run_id: str = Field(default_factory=_new_run_id)
    include_tools: bool = True
    request_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_query: MemoryQuery | None = None
    memory_results: list[MemorySearchResult] | None = None
    memory_max_chars: int = Field(default=4000, gt=0)
    loop: BoundedLoopOptions = Field(default_factory=BoundedLoopOptions)

    model_config = ConfigDict(extra="forbid")


class RuntimeTurnResult(BaseModel):
    """Runtime 对外返回的单轮或 loop 执行结果。

    Attributes:
        session_id (str): 本次结果所属的会话标识。
        run_id (str): 本次结果所属的 runtime run 标识。
        status (RuntimeStatus): 本次运行的最终状态。
        assistant_message (Msg | None): 最终对外返回的 assistant 消息。
        tool_result_messages (list[Msg]): 工具执行后可回灌给模型的消息。
        tool_results (list[ToolResult]): 程序侧可读取的结构化工具执行结果。
        steps (int): 本次运行实际完成的 provider 调用步数。
        error (RuntimeErrorInfo | None): 失败时返回的归一化错误信息。
        pending_interaction (HumanInteraction | None): 等待人工响应时返回的持久化请求。
        metadata (dict[str, Any]): 运行摘要、追踪字段或调试附加信息。
    """

    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: RuntimeStatus
    assistant_message: Msg | None = None
    tool_result_messages: list[Msg] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    steps: int = Field(default=1, gt=0)
    error: RuntimeErrorInfo | None = None
    pending_interaction: HumanInteraction | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @model_validator(mode="after")
    def _validate_pending_interaction(self) -> RuntimeTurnResult:
        """确保 pending interaction 只随等待状态返回。"""
        if (self.status is RuntimeStatus.WAITING_HUMAN) != (self.pending_interaction is not None):
            raise ValueError("waiting_human 状态必须且只能包含 pending_interaction")
        return self


__all__ = [
    "BoundedLoopOptions",
    "ProviderResponseSnapshot",
    "Runstate",
    "RuntimeActivationInput",
    "RuntimeActivationOutcome",
    "RuntimeActivationResult",
    "RuntimeApprovedToolCall",
    "RuntimeContinuationClaim",
    "RuntimeCursor",
    "RuntimeErrorInfo",
    "RuntimeHITLCheckpoint",
    "RuntimeOptions",
    "RuntimeOptionsSnapshot",
    "RuntimeStatus",
    "RuntimeTurnInput",
    "RuntimeTurnResult",
    "ToolErrorPolicy",
    "ToolResultCommit",
]
