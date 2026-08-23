"""Lifecycle store 的 command 与同步协议。

所有 mutation 只接收一个显式 command，使 CAS、activation fence 与历史 revision
成为可验证输入。

Example:
    store.create_run(command)
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from ..hitl.models import HumanInteraction, HumanInteractionResponse
from ..message.message import Msg
from ..tools.base import ToolResult
from .models import (
    ActivationKind,
    AgentRunOptions,
    AgentRunRequest,
    RecoveryDisposition,
    RunCheckpoint,
    RunControlSnapshot,
    RunErrorInfo,
    RunEvent,
    RunRecord,
    RunResult,
    RunStopReason,
    RunToolCallRecord,
    RunUsage,
    SessionSnapshot,
    _required_aware_utc,
    _trim_required,
)


class _Command(BaseModel):
    """Mutation command 的统一不可变配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class CreateRun(_Command):
    """原子创建 logical run、session lane 与初始 activation。"""

    request: AgentRunRequest
    options: AgentRunOptions
    agent_id: str
    environment_fingerprint: str
    start_activation_id: str
    initial_checkpoint: RunCheckpoint
    now: datetime

    @field_validator("agent_id", "environment_fingerprint", "start_activation_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")

    @model_validator(mode="after")
    def _validate_initial_facts(self) -> CreateRun:
        if self.request.run_id is None:
            raise ValueError("CreateRun request 必须包含最终 run_id")
        if self.initial_checkpoint.run_id != self.request.run_id:
            raise ValueError("initial checkpoint 必须绑定 request run_id")
        if self.initial_checkpoint.activation_id != self.start_activation_id:
            raise ValueError("initial checkpoint 必须绑定 start activation")
        if self.initial_checkpoint.sequence != 1:
            raise ValueError("initial checkpoint sequence 必须为 1")
        if self.initial_checkpoint.environment_fingerprint != self.environment_fingerprint:
            raise ValueError("initial checkpoint environment fingerprint 不匹配")
        return self


class ResumeWaitingRun(_Command):
    """为 resolved waiting run 创建新的 activation fence。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    new_activation_id: str
    kind: ActivationKind
    expected_checkpoint_sequence: int = Field(ge=0)
    now: datetime

    @field_validator("run_id", "new_activation_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class ReserveModelStep(_Command):
    """在 provider effect 前 durable 预留一个模型步。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    activation_id: str
    now: datetime

    @field_validator("run_id", "activation_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class CommitModelStep(_Command):
    """原子提交模型响应、history、tool intents 与 checkpoint。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    activation_id: str
    expected_session_revision: int = Field(ge=0)
    message_delta: list[Msg] = Field(default_factory=list)
    usage: RunUsage
    prepared_tool_calls: list[RunToolCallRecord] = Field(default_factory=list)
    checkpoint: RunCheckpoint
    assistant_message: Msg | None = None
    now: datetime

    @field_validator("run_id", "activation_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class ClaimToolCall(_Command):
    """在工具副作用前 durable claim 一次 prepared 调用。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    activation_id: str
    tool_call_id: str
    fingerprint: str
    expected_tool_version: int = Field(ge=0)
    now: datetime

    @field_validator("run_id", "activation_id", "tool_call_id", "fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class CommitToolResult(_Command):
    """原子提交一次精确工具结果、history 与 checkpoint。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    activation_id: str
    expected_session_revision: int = Field(ge=0)
    tool_call_id: str
    expected_tool_version: int = Field(ge=0)
    result: ToolResult
    message_delta: list[Msg] = Field(default_factory=list)
    checkpoint: RunCheckpoint
    now: datetime

    @field_validator("run_id", "activation_id", "tool_call_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class SuspendRun(_Command):
    """原子提交当前模型步并将 active run 转为 waiting。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    activation_id: str
    expected_session_revision: int = Field(ge=0)
    message_delta: list[Msg] = Field(default_factory=list)
    prepared_tool_calls: list[RunToolCallRecord] = Field(default_factory=list)
    checkpoint: RunCheckpoint
    pending_interaction: HumanInteraction
    assistant_message: Msg | None = None
    usage: RunUsage
    now: datetime

    @field_validator("run_id", "activation_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class ResolveInteraction(_Command):
    """以 response/version/fingerprint CAS 解决 pending interaction。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    interaction_id: str
    expected_interaction_version: int = Field(ge=0)
    response: HumanInteractionResponse
    expected_fingerprint: str
    now: datetime

    @field_validator("run_id", "interaction_id", "expected_fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class RequestCancellation(_Command):
    """记录首次 cancellation request，可显式结算 waiting run。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    activation_id: str | None = None
    reason: str
    settle_waiting: bool = False
    now: datetime

    @field_validator("run_id", "reason")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("activation_id")
    @classmethod
    def _validate_optional_activation(cls, value: str | None) -> str | None:
        return None if value is None else _trim_required(value, field_name="activation_id")

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class FinishRun(_Command):
    """将 active 或 waiting run 原子结算为 terminal。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    activation_id: str | None = None
    stop_reason: RunStopReason
    assistant_message: Msg | None = None
    error: RunErrorInfo | None = None
    interaction_close_reason: str | None = None
    now: datetime

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        return _trim_required(value, field_name="run_id")

    @field_validator("activation_id", "interaction_close_reason")
    @classmethod
    def _validate_optional_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        return None if value is None else _trim_required(value, field_name=str(info.field_name))

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")


class RecoverActiveRun(_Command):
    """根据 durable effect facts 放弃并恢复或终止旧 activation。"""

    run_id: str
    expected_run_revision: int = Field(ge=0)
    expected_activation_id: str
    expected_checkpoint_sequence: int = Field(ge=0)
    recovery_disposition: RecoveryDisposition
    new_activation_id: str | None = None
    now: datetime

    @field_validator("run_id", "expected_activation_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("new_activation_id")
    @classmethod
    def _validate_optional_activation(cls, value: str | None) -> str | None:
        return None if value is None else _trim_required(value, field_name="new_activation_id")

    @field_validator("now")
    @classmethod
    def _validate_now(cls, value: datetime) -> datetime:
        return _required_aware_utc(value, field_name="now")

    @model_validator(mode="after")
    def _validate_disposition(self) -> RecoverActiveRun:
        if (self.recovery_disposition is RecoveryDisposition.RESUME) != (
            self.new_activation_id is not None
        ):
            raise ValueError("resume recovery 必须且只能包含 new_activation_id")
        return self


class RunCommit(_Command):
    """一次 mutation 原子提交后返回的完整事实集合。"""

    run: RunRecord
    session: SessionSnapshot | None = None
    checkpoint: RunCheckpoint | None = None
    interaction: HumanInteraction | None = None
    events: tuple[RunEvent, ...] = ()
    result: RunResult | None = None


class LifecycleStore(Protocol):
    """Logical run aggregate 的同步 durable boundary。"""

    def create_run(self, command: CreateRun) -> RunCommit: ...

    def resume_waiting_run(self, command: ResumeWaitingRun) -> RunCommit: ...

    def reserve_model_step(self, command: ReserveModelStep) -> RunCommit: ...

    def commit_model_step(self, command: CommitModelStep) -> RunCommit: ...

    def claim_tool_call(self, command: ClaimToolCall) -> RunCommit: ...

    def commit_tool_result(self, command: CommitToolResult) -> RunCommit: ...

    def suspend_run(self, command: SuspendRun) -> RunCommit: ...

    def resolve_interaction(self, command: ResolveInteraction) -> RunCommit: ...

    def request_cancellation(self, command: RequestCancellation) -> RunCommit: ...

    def finish_run(self, command: FinishRun) -> RunCommit: ...

    def recover_active_run(self, command: RecoverActiveRun) -> RunCommit: ...

    def load_run(self, run_id: str) -> RunRecord | None: ...

    def load_run_control(self, run_id: str) -> RunControlSnapshot | None: ...

    def load_session(self, session_id: str) -> SessionSnapshot: ...

    def load_session_lane(self, session_id: str) -> str | None: ...

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None: ...

    def load_checkpoint(self, run_id: str) -> RunCheckpoint | None: ...

    def load_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
    ) -> RunToolCallRecord | None: ...

    def list_tool_calls(self, run_id: str) -> list[RunToolCallRecord]: ...

    def load_result(self, run_id: str) -> RunResult | None: ...

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]: ...


__all__ = [
    "ResumeWaitingRun",
    "ClaimToolCall",
    "CommitModelStep",
    "CommitToolResult",
    "CreateRun",
    "FinishRun",
    "LifecycleStore",
    "RecoverActiveRun",
    "RequestCancellation",
    "ReserveModelStep",
    "ResolveInteraction",
    "RunCommit",
    "SuspendRun",
]
