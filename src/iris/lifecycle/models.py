"""Logical run 生命周期的纯数据契约。

本模块只定义可持久化模型与投影函数，不拥有运行控制流或具体存储实现。

Example:
    request = AgentRunRequest(input="hello", run_id="run-1")
    options = AgentRunOptions()
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..exceptions import IrisRunStateError
from ..hitl.models import HumanInteraction
from ..message.message import Msg
from ..tools.base import ToolResult

RunErrorSource = Literal[
    "config",
    "context",
    "provider",
    "tool",
    "memory",
    "session",
    "runtime",
    "lifecycle",
    "persistence",
]


class RunPhase(StrEnum):
    """Logical run 的外部可见阶段。"""

    ACTIVE = "active"
    WAITING = "waiting"
    TERMINAL = "terminal"


class RunStopReason(StrEnum):
    """Terminal run 的稳定停止原因。"""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERACTION_EXPIRED = "interaction_expired"
    BUDGET_EXHAUSTED = "budget_exhausted"
    OUTCOME_UNKNOWN = "outcome_unknown"


class ToolErrorPolicy(StrEnum):
    """工具错误进入下一模型步或立即停止的策略。"""

    RETURN_TO_MODEL = "return_to_model"
    STOP = "stop"


class ActivationKind(StrEnum):
    """Activation 的创建来源。"""

    START = "start"
    RESUME = "resume"
    RECOVER = "recover"


class ActivationStatus(StrEnum):
    """Activation 的 durable 状态。"""

    ACTIVE = "active"
    SETTLED = "settled"
    ABANDONED = "abandoned"


class ActivationOutcome(StrEnum):
    """已结束 activation 的事实结果。"""

    COMPLETED = "completed"
    SUSPENDED = "suspended"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERED = "recovered"
    OUTCOME_UNKNOWN = "outcome_unknown"


class CheckpointResumability(StrEnum):
    """Checkpoint 是否允许安全恢复。"""

    SAFE = "safe"
    OUTCOME_READY = "outcome_ready"
    BLOCKED_UNKNOWN = "blocked_unknown"


class ToolCallPhase(StrEnum):
    """工具调用的 durable effect 状态。"""

    PREPARED = "prepared"
    CLAIMED = "claimed"
    COMMITTED = "committed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class RecoveryDisposition(StrEnum):
    """活跃 run 恢复时对旧 activation 的处置。"""

    RESUME = "resume"
    FINALIZE = "finalize"
    OUTCOME_UNKNOWN = "outcome_unknown"


class RunEventKind(StrEnum):
    """Lifecycle store 发布的 durable 事件类别。"""

    RUN_STARTED = "run.started"
    ACTIVATION_STARTED = "activation.started"
    MODEL_STEP_RESERVED = "model_step.reserved"
    MODEL_STEP_COMMITTED = "model_step.committed"
    TOOL_CALL_CLAIMED = "tool_call.claimed"
    TOOL_CALL_COMMITTED = "tool_call.committed"
    TOOL_CALL_OUTCOME_UNKNOWN = "tool_call.outcome_unknown"
    INTERACTION_SUSPENDED = "interaction.suspended"
    INTERACTION_RESOLVED = "interaction.resolved"
    CANCELLATION_REQUESTED = "run.cancellation_requested"
    ACTIVATION_ABANDONED = "activation.abandoned"
    RUN_TERMINAL = "run.terminal"


def validate_json_safe(value: Any, field_name: str) -> Any:
    """验证值能以严格 JSON 形式持久化，并原样返回。"""
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是 JSON-safe 数据") from exc
    return value


def _trim_required(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _aware_utc(value: datetime | None, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} 必须包含时区")
    return value.astimezone(UTC)


def _required_aware_utc(value: datetime, *, field_name: str) -> datetime:
    validated = _aware_utc(value, field_name=field_name)
    if validated is None:
        raise ValueError(f"{field_name} 不能为空")
    return validated


def _validate_phase_facts(
    *,
    phase: RunPhase,
    stop_reason: RunStopReason | None,
    current_activation_id: str | None,
    pending_interaction_id: str | None,
    finished_at: datetime | None,
) -> None:
    if phase is RunPhase.ACTIVE:
        if stop_reason is not None or current_activation_id is None or finished_at is not None:
            raise ValueError("active run 的 phase facts 不一致")
        if pending_interaction_id is not None:
            raise ValueError("active run 不能包含 pending interaction")
        return
    if phase is RunPhase.WAITING:
        if stop_reason is not None or current_activation_id is not None or finished_at is not None:
            raise ValueError("waiting run 的 phase facts 不一致")
        if pending_interaction_id is None:
            raise ValueError("waiting run 必须包含 pending interaction")
        return
    if stop_reason is None or finished_at is None:
        raise ValueError("terminal run 必须包含 stop reason 与 finished time")
    if current_activation_id is not None or pending_interaction_id is not None:
        raise ValueError("terminal run 不能保留 activation 或 interaction")


class _FrozenModel(BaseModel):
    """Lifecycle 数据模型的统一不可变配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class RunLimits(_FrozenModel):
    """一次 logical run 的固定预算与截止约束。"""

    max_model_steps: int = Field(default=20, gt=0)
    deadline_at: datetime | None = None
    interaction_timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("deadline_at")
    @classmethod
    def _validate_deadline(cls, value: datetime | None) -> datetime | None:
        return _aware_utc(value, field_name="deadline_at")


class AgentRunRequest(_FrozenModel):
    """调用方提交的一次 logical run 请求。"""

    input: str
    session_id: str = "default"
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input", "session_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("run_id")
    @classmethod
    def _validate_optional_run_id(cls, value: str | None) -> str | None:
        return None if value is None else _trim_required(value, field_name="run_id")

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_safe(value, field_name="metadata")


class RuntimeExecutionOptions(_FrozenModel):
    """每次 activation 固定使用的 runtime 行为选项。"""

    include_tools: bool = True
    request_options: dict[str, Any] = Field(default_factory=dict)
    tool_timeout_seconds: float | None = Field(default=None, gt=0)
    tool_error_policy: ToolErrorPolicy = ToolErrorPolicy.RETURN_TO_MODEL
    memory_query: dict[str, Any] | None = None
    memory_results: list[dict[str, Any]] | None = None
    memory_max_chars: int = Field(default=4000, gt=0)

    @field_validator("request_options")
    @classmethod
    def _validate_request_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_safe(value, field_name="request_options")

    @field_validator("memory_query", "memory_results")
    @classmethod
    def _validate_memory_snapshots(cls, value: Any, info: ValidationInfo) -> Any:
        return validate_json_safe(value, field_name=str(info.field_name))


class AgentRunOptions(_FrozenModel):
    """Logical run 的完整固定选项。"""

    limits: RunLimits = Field(default_factory=RunLimits)
    runtime: RuntimeExecutionOptions = Field(default_factory=RuntimeExecutionOptions)


class RunUsage(_FrozenModel):
    """Logical run 已持久化的预算和 token 计数。"""

    model_steps_reserved: int = Field(default=0, ge=0)
    model_steps_committed: int = Field(default=0, ge=0)
    tool_calls_committed: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_counters(self) -> RunUsage:
        if self.model_steps_committed > self.model_steps_reserved:
            raise ValueError("committed model steps 不能超过 reserved model steps")
        if self.model_steps_reserved - self.model_steps_committed > 1:
            raise ValueError("single owner 不能存在多个未提交 model-step reservation")
        return self


class RunErrorInfo(_FrozenModel):
    """Logical run 对外返回的结构化错误。"""

    code: str
    message: str
    source: RunErrorSource
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("details")
    @classmethod
    def _validate_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_safe(value, field_name="details")


class RunSnapshot(_FrozenModel):
    """调用方可观察的 logical run 不可变快照。"""

    run_id: str
    session_id: str
    agent_id: str
    phase: RunPhase
    stop_reason: RunStopReason | None = None
    revision: int = Field(ge=1)
    current_activation_id: str | None = None
    pending_interaction_id: str | None = None
    cancellation_requested_at: datetime | None = None
    cancellation_reason: str | None = None
    limits: RunLimits
    usage: RunUsage
    environment_fingerprint: str
    checkpoint_sequence: int = Field(ge=0)
    last_event_sequence: int = Field(ge=1)
    created_at: datetime
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None

    @field_validator("run_id", "session_id", "agent_id", "environment_fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("current_activation_id", "pending_interaction_id", "cancellation_reason")
    @classmethod
    def _validate_optional_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        return None if value is None else _trim_required(value, field_name=str(info.field_name))

    @field_validator(
        "cancellation_requested_at",
        "created_at",
        "started_at",
        "updated_at",
        "finished_at",
    )
    @classmethod
    def _validate_times(cls, value: datetime | None, info: ValidationInfo) -> datetime | None:
        return _aware_utc(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_state(self) -> RunSnapshot:
        _validate_phase_facts(
            phase=self.phase,
            stop_reason=self.stop_reason,
            current_activation_id=self.current_activation_id,
            pending_interaction_id=self.pending_interaction_id,
            finished_at=self.finished_at,
        )
        if (self.cancellation_requested_at is None) != (self.cancellation_reason is None):
            raise ValueError("cancellation time 与 reason 必须同时存在")
        if self.usage.model_steps_reserved > self.limits.max_model_steps:
            raise ValueError("run usage 不能超过固定 model-step 预算")
        return self


class RunResult(_FrozenModel):
    """Waiting 或 terminal run 的 durable 对外结果。"""

    run: RunSnapshot
    assistant_message: Msg | None = None
    pending_interaction: HumanInteraction | None = None
    error: RunErrorInfo | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> RunResult:
        if self.run.phase is RunPhase.ACTIVE:
            raise ValueError("active run 不产生 RunResult")
        if self.run.phase is RunPhase.WAITING:
            if self.pending_interaction is None or self.error is not None:
                raise ValueError("waiting result 必须且只能包含 pending interaction")
            if (
                self.pending_interaction.interaction_id != self.run.pending_interaction_id
                or self.pending_interaction.run_id != self.run.run_id
                or self.pending_interaction.session_id != self.run.session_id
            ):
                raise ValueError("waiting result 的 interaction identity 不匹配")
            return self
        if self.pending_interaction is not None:
            raise ValueError("terminal result 不能包含 pending interaction")
        if (
            self.run.stop_reason
            in {
                RunStopReason.FAILED,
                RunStopReason.OUTCOME_UNKNOWN,
            }
            and self.error is None
        ):
            raise ValueError("failed/outcome_unknown result 必须包含 error")
        if self.run.stop_reason is RunStopReason.COMPLETED and self.error is not None:
            raise ValueError("completed result 不能包含 error")
        return self


class RunRecord(_FrozenModel):
    """Store 内 logical run aggregate 的权威记录。"""

    run_id: str
    session_id: str
    agent_id: str
    request: AgentRunRequest
    options: AgentRunOptions
    phase: RunPhase
    stop_reason: RunStopReason | None = None
    revision: int = Field(ge=1)
    current_activation_id: str | None = None
    pending_interaction_id: str | None = None
    cancellation_requested_at: datetime | None = None
    cancellation_reason: str | None = None
    usage: RunUsage = Field(default_factory=RunUsage)
    environment_fingerprint: str
    assistant_message: Msg | None = None
    error: RunErrorInfo | None = None
    checkpoint_sequence: int = Field(ge=0)
    last_event_sequence: int = Field(ge=1)
    created_at: datetime
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None

    @field_validator("run_id", "session_id", "agent_id", "environment_fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("current_activation_id", "pending_interaction_id", "cancellation_reason")
    @classmethod
    def _validate_optional_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        return None if value is None else _trim_required(value, field_name=str(info.field_name))

    @field_validator(
        "cancellation_requested_at",
        "created_at",
        "started_at",
        "updated_at",
        "finished_at",
    )
    @classmethod
    def _validate_times(cls, value: datetime | None, info: ValidationInfo) -> datetime | None:
        return _aware_utc(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_state(self) -> RunRecord:
        if self.request.run_id != self.run_id or self.request.session_id != self.session_id:
            raise ValueError("stored request 必须绑定 record identity")
        _validate_phase_facts(
            phase=self.phase,
            stop_reason=self.stop_reason,
            current_activation_id=self.current_activation_id,
            pending_interaction_id=self.pending_interaction_id,
            finished_at=self.finished_at,
        )
        if (self.cancellation_requested_at is None) != (self.cancellation_reason is None):
            raise ValueError("cancellation time 与 reason 必须同时存在")
        if self.usage.model_steps_reserved > self.options.limits.max_model_steps:
            raise ValueError("run usage 不能超过固定 model-step 预算")
        if self.phase is not RunPhase.TERMINAL and self.checkpoint_sequence < 1:
            raise ValueError("non-terminal run 必须包含 current checkpoint")
        return self


class SessionSnapshot(_FrozenModel):
    """一个 session 的消息历史与 CAS revision。"""

    session_id: str
    revision: int = Field(default=0, ge=0)
    messages: list[Msg] = Field(default_factory=list)

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str) -> str:
        return _trim_required(value, field_name="session_id")


class ActivationRecord(_FrozenModel):
    """一次 activation fence 的 durable 记录。"""

    activation_id: str
    run_id: str
    ordinal: int = Field(ge=1)
    kind: ActivationKind
    status: ActivationStatus
    outcome: ActivationOutcome | None = None
    started_at: datetime
    ended_at: datetime | None = None

    @field_validator("activation_id", "run_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("started_at", "ended_at")
    @classmethod
    def _validate_times(cls, value: datetime | None, info: ValidationInfo) -> datetime | None:
        return _aware_utc(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_status(self) -> ActivationRecord:
        if self.status is ActivationStatus.ACTIVE:
            if self.outcome is not None or self.ended_at is not None:
                raise ValueError("active activation 不能包含结束事实")
        elif self.outcome is None or self.ended_at is None:
            raise ValueError("已结束 activation 必须包含 outcome 与 ended_at")
        return self


class RunCheckpoint(_FrozenModel):
    """Logical run 当前可恢复位置。"""

    checkpoint_version: Literal[1] = 1
    run_id: str
    sequence: int = Field(ge=1)
    activation_id: str
    engine_cursor: dict[str, Any]
    session_revision: int = Field(ge=0)
    model_steps_reserved: int = Field(ge=0)
    model_steps_committed: int = Field(ge=0)
    environment_fingerprint: str
    resumability: CheckpointResumability = CheckpointResumability.SAFE

    @field_validator("run_id", "activation_id", "environment_fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("engine_cursor")
    @classmethod
    def _validate_cursor(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_safe(value, field_name="engine_cursor")

    @model_validator(mode="after")
    def _validate_counters(self) -> RunCheckpoint:
        if self.model_steps_committed > self.model_steps_reserved:
            raise ValueError("checkpoint committed steps 不能超过 reserved steps")
        if self.model_steps_reserved - self.model_steps_committed > 1:
            raise ValueError("checkpoint 不能包含多个未提交 model-step reservation")
        return self


class RunToolCallRecord(_FrozenModel):
    """工具调用从 prepared 到 durable result 的状态记录。"""

    run_id: str
    step_index: int = Field(ge=0)
    ordinal: int = Field(ge=1)
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    interaction_id: str | None = None
    phase: ToolCallPhase
    claim_activation_id: str | None = None
    result: ToolResult | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    claimed_at: datetime | None = None
    committed_at: datetime | None = None

    @field_validator("run_id", "tool_call_id", "tool_name", "fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("interaction_id", "claim_activation_id")
    @classmethod
    def _validate_optional_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        return None if value is None else _trim_required(value, field_name=str(info.field_name))

    @field_validator("arguments")
    @classmethod
    def _validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_safe(value, field_name="arguments")

    @field_validator("created_at", "updated_at", "claimed_at", "committed_at")
    @classmethod
    def _validate_times(cls, value: datetime | None, info: ValidationInfo) -> datetime | None:
        return _aware_utc(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_phase(self) -> RunToolCallRecord:
        if (self.claim_activation_id is None) != (self.claimed_at is None):
            raise ValueError("tool call claim identity 与时间必须同时存在")
        if self.phase is ToolCallPhase.PREPARED:
            if any(
                value is not None
                for value in (
                    self.claim_activation_id,
                    self.result,
                    self.claimed_at,
                    self.committed_at,
                )
            ):
                raise ValueError("prepared tool call 包含了后续阶段事实")
        elif self.phase is ToolCallPhase.CLAIMED:
            if (
                self.claim_activation_id is None
                or self.claimed_at is None
                or self.result is not None
            ):
                raise ValueError("claimed tool call 的 phase facts 不一致")
        elif self.phase is ToolCallPhase.COMMITTED:
            if self.result is None or self.committed_at is None:
                raise ValueError("committed tool call 必须包含 result 与 committed_at")
        elif self.claim_activation_id is None or self.claimed_at is None:
            raise ValueError("outcome_unknown tool call 必须来自 durable claim")
        return self


class RunEvent(_FrozenModel):
    """与 aggregate mutation 同事务追加的 durable 事件。"""

    run_id: str
    session_id: str
    sequence: int = Field(ge=1)
    kind: RunEventKind
    occurred_at: datetime
    activation_id: str | None = None
    step_index: int | None = Field(default=None, ge=0)
    correlation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "session_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("activation_id", "correlation_id")
    @classmethod
    def _validate_optional_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        return None if value is None else _trim_required(value, field_name=str(info.field_name))

    @field_validator("occurred_at")
    @classmethod
    def _validate_occurred_at(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="occurred_at")

    @field_validator("payload")
    @classmethod
    def _validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_safe(value, field_name="payload")


def snapshot_run(record: RunRecord) -> RunSnapshot:
    """从权威 record 生成不可变 public snapshot。"""
    return RunSnapshot(
        run_id=record.run_id,
        session_id=record.session_id,
        agent_id=record.agent_id,
        phase=record.phase,
        stop_reason=record.stop_reason,
        revision=record.revision,
        current_activation_id=record.current_activation_id,
        pending_interaction_id=record.pending_interaction_id,
        cancellation_requested_at=record.cancellation_requested_at,
        cancellation_reason=record.cancellation_reason,
        limits=record.options.limits,
        usage=record.usage,
        environment_fingerprint=record.environment_fingerprint,
        checkpoint_sequence=record.checkpoint_sequence,
        last_event_sequence=record.last_event_sequence,
        created_at=record.created_at,
        started_at=record.started_at,
        updated_at=record.updated_at,
        finished_at=record.finished_at,
    )


def project_result(
    record: RunRecord,
    interaction: HumanInteraction | None = None,
) -> RunResult:
    """从 waiting/terminal record 投影 durable result。

    Raises:
        IrisRunStateError: 当 record 仍 active 或 interaction 与状态不一致时。
    """
    if record.phase is RunPhase.ACTIVE:
        raise IrisRunStateError("active run 不产生 RunResult", run_id=record.run_id)
    try:
        return RunResult(
            run=snapshot_run(record),
            assistant_message=record.assistant_message,
            pending_interaction=interaction,
            error=record.error,
        )
    except ValueError as exc:
        raise IrisRunStateError("run result facts 不一致", run_id=record.run_id) from exc


__all__ = [
    "ActivationKind",
    "ActivationOutcome",
    "ActivationRecord",
    "ActivationStatus",
    "AgentRunOptions",
    "AgentRunRequest",
    "CheckpointResumability",
    "RecoveryDisposition",
    "RunCheckpoint",
    "RunErrorInfo",
    "RunErrorSource",
    "RunEvent",
    "RunEventKind",
    "RunLimits",
    "RunPhase",
    "RunRecord",
    "RunResult",
    "RunSnapshot",
    "RunStopReason",
    "RunToolCallRecord",
    "RunUsage",
    "RuntimeExecutionOptions",
    "SessionSnapshot",
    "ToolCallPhase",
    "ToolErrorPolicy",
    "project_result",
    "snapshot_run",
    "validate_json_safe",
]
