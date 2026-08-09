"""Inner engine 所需的 required commit port 与事务 DTO。

本模块只描述 runtime 与 lifecycle owner 之间的同步提交边界。具体 store 映射由
后续 harness 实现，runtime 本身不直接写 session、checkpoint 或 interaction。
"""

# region imports
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticSerializationError

from ..exceptions import IrisRunConflictError, IrisRunStateError
from ..hitl import HumanInteraction, HumanInteractionRequest, make_call_fingerprint
from ..lifecycle import CheckpointResumability, SessionSnapshot, validate_json_safe
from ..message import Msg
from ..tools import PreparedToolCall, ToolEffectGuard, ToolResult
from .models import RuntimeActivationInput, RuntimeCursor

# endregion


class _FrozenCommitModel(BaseModel):
    """Required commit facts 的不可变模型基类。"""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    @model_validator(mode="after")
    def _validate_json_safe_fact(self) -> Self:
        """拒绝 required commit fact 中不可持久化的 live object。"""
        try:
            serialized = self.model_dump(mode="json")
        except (PydanticSerializationError, TypeError, ValueError) as exc:
            raise ValueError("required commit fact 必须是 JSON-safe") from exc
        validate_json_safe(serialized, field_name="required commit fact")
        return self


class ModelStepReservation(_FrozenCommitModel):
    """Provider effect 前的 durable model-step reservation。"""

    granted: bool
    step_index: int = Field(ge=0)
    cursor: RuntimeCursor
    remaining_deadline_seconds: float | None = Field(default=None, ge=0)


class RuntimeToolCall(_FrozenCommitModel):
    """Runtime 提交给 port 的精确 prepared tool fact。"""

    run_id: str
    activation_id: str
    step_index: int = Field(ge=0)
    ordinal: int = Field(ge=1)
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    interaction_id: str | None = None
    tool_version: int = Field(default=1, ge=1)

    @field_validator(
        "run_id",
        "activation_id",
        "tool_call_id",
        "tool_name",
        "interaction_id",
    )
    @classmethod
    def _validate_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized

    @field_validator("arguments")
    @classmethod
    def _validate_arguments(cls, value: dict[str, object]) -> dict[str, object]:
        return validate_json_safe(value, field_name="arguments")


class ToolCallClaim(_FrozenCommitModel):
    """工具 effect 前由 commit port 返回的 durable claim。"""

    run_id: str
    activation_id: str
    tool_call_id: str
    tool_name: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_version: int = Field(ge=1)

    @field_validator("run_id", "activation_id", "tool_call_id", "tool_name")
    @classmethod
    def _validate_text(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized


class RuntimeModelStepCommit(_FrozenCommitModel):
    """一次 provider response 的完整原子提交事实。"""

    cursor_before: RuntimeCursor
    message_delta: tuple[Msg, ...] = ()
    assistant_message: Msg
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    prepared_tool_calls: tuple[RuntimeToolCall, ...] = ()
    cursor_after: RuntimeCursor
    resumability: CheckpointResumability = CheckpointResumability.SAFE


class RuntimeToolResultCommit(_FrozenCommitModel):
    """一次 claimed 或 preflight-only 工具结果的原子提交事实。"""

    tool_call: RuntimeToolCall
    claim: ToolCallClaim | None = None
    result: ToolResult
    message_delta: tuple[Msg, ...]
    cursor_after: RuntimeCursor


class RuntimeSuspension(_FrozenCommitModel):
    """模型步、prepared batch 与 interaction 的原子等待输入。"""

    cursor_before: RuntimeCursor
    message_delta: tuple[Msg, ...] = ()
    assistant_message: Msg
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    prepared_tool_calls: tuple[RuntimeToolCall, ...]
    cursor: RuntimeCursor
    interaction_request: HumanInteractionRequest
    expires_at: datetime | None = None
    resumability: CheckpointResumability = CheckpointResumability.SAFE

    @field_validator("expires_at")
    @classmethod
    def _validate_expiry(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at 必须包含时区")
        return value.astimezone(UTC)


class RuntimeSuspensionResult(_FrozenCommitModel):
    """等待事务成功后返回给 engine 的 durable projection。"""

    cursor: RuntimeCursor
    interaction: HumanInteraction


class RuntimeCommitPort(Protocol):
    """Runner-owned required commit boundary。"""

    def load_session(self) -> SessionSnapshot:
        """读取 port 绑定 session 的 revisioned history。"""

    def reserve_model_step(self, cursor: RuntimeCursor) -> ModelStepReservation:
        """在 provider effect 前预留一个模型步。"""

    def commit_model_step(self, commit: RuntimeModelStepCommit) -> RuntimeCursor:
        """提交 provider response、history、tool intents 与 cursor。"""

    def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
        """在工具 effect 前 durable claim 精确调用。"""

    def commit_tool_result(self, commit: RuntimeToolResultCommit) -> RuntimeCursor:
        """提交工具结果、history 与更新后的 cursor。"""

    def suspend(self, suspension: RuntimeSuspension) -> RuntimeSuspensionResult:
        """原子提交模型事实并结束当前 activation 为 waiting。"""

    def cancellation_requested(self) -> bool:
        """读取 logical run 的 durable cancellation request。"""

    def remaining_deadline_seconds(self) -> float | None:
        """返回当前 run deadline 的剩余秒数。"""


def build_runtime_tool_call(
    *,
    activation: RuntimeActivationInput,
    cursor: RuntimeCursor,
    prepared: PreparedToolCall,
    workspace_root: Path,
    ordinal: int | None = None,
    interaction_id: str | None = None,
) -> RuntimeToolCall:
    """把一条 revalidated/preflight 调用转换为 durable tool fact。"""
    arguments: dict[str, object] = dict(
        prepared.validated_params or prepared.tool_use.input
    )
    fingerprint = make_call_fingerprint(
        session_id=activation.session_id,
        run_id=activation.run_id,
        tool_call_id=prepared.tool_use.id,
        tool_name=prepared.tool_use.name,
        arguments=arguments,
        workspace_root=str(workspace_root.resolve()),
    )
    if prepared.human_request is not None:
        fingerprint = prepared.human_request.tool_call.fingerprint
    return RuntimeToolCall(
        run_id=activation.run_id,
        activation_id=activation.activation_id,
        step_index=cursor.step_index,
        ordinal=ordinal or cursor.next_tool_index + 1,
        tool_call_id=prepared.tool_use.id,
        tool_name=prepared.tool_use.name,
        arguments=arguments,
        fingerprint=fingerprint,
        interaction_id=interaction_id,
    )


class CommitPortToolEffectGuard(ToolEffectGuard):
    """把 revalidated tool call 转换为 required durable claim。"""

    def __init__(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        commits: RuntimeCommitPort,
        workspace_root: Path,
        tool_index: int | None = None,
        interaction_id: str | None = None,
    ) -> None:
        if tool_index is not None and (
            cursor.position != "tool_batch"
            or tool_index < cursor.next_tool_index
            or tool_index >= len(cursor.tool_calls)
        ):
            raise IrisRunStateError("effect guard tool 索引不在未提交 batch 后缀中")
        self._activation = activation
        self._cursor = cursor
        self._commits = commits
        self._workspace_root = workspace_root.resolve()
        self._tool_index = cursor.next_tool_index if tool_index is None else tool_index
        self._interaction_id = interaction_id
        self._claims: dict[str, tuple[RuntimeToolCall, ToolCallClaim]] = {}

    def before_effect(self, prepared: PreparedToolCall) -> None:
        """校验 cursor subject 并在任何执行生命周期前提交 claim。"""
        index = self._tool_index
        if self._cursor.position != "tool_batch" or index >= len(self._cursor.tool_calls):
            raise IrisRunStateError("effect guard cursor 不指向可执行工具")
        expected = self._cursor.tool_calls[index]
        if prepared.tool_use != expected:
            raise IrisRunConflictError("revalidated tool call 与 cursor subject 不匹配")

        call = self._runtime_tool_call(prepared)
        existing = self._claims.get(call.tool_call_id)
        if existing is not None:
            if existing[0] != call:
                raise IrisRunConflictError("相同 tool call ID 的 claim subject 已变化")
            return
        claim = self._commits.claim_tool_call(call)
        if (
            claim.run_id != call.run_id
            or claim.activation_id != call.activation_id
            or claim.tool_call_id != call.tool_call_id
            or claim.tool_name != call.tool_name
            or claim.fingerprint != call.fingerprint
            or claim.tool_version != call.tool_version + 1
        ):
            raise IrisRunConflictError("commit port 返回的 tool claim identity 不匹配")
        self._claims[call.tool_call_id] = (call, claim)

    def claim_for(self, tool_call_id: str) -> ToolCallClaim | None:
        """返回本 guard 已确认的精确 durable claim。"""
        item = self._claims.get(tool_call_id)
        return None if item is None else item[1]

    def call_for(self, tool_call_id: str) -> RuntimeToolCall | None:
        """返回 durable claim 对应的 runtime tool fact。"""
        item = self._claims.get(tool_call_id)
        return None if item is None else item[0]

    def _runtime_tool_call(self, prepared: PreparedToolCall) -> RuntimeToolCall:
        return build_runtime_tool_call(
            activation=self._activation,
            cursor=self._cursor,
            prepared=prepared,
            workspace_root=self._workspace_root,
            ordinal=self._tool_index + 1,
            interaction_id=self._interaction_id,
        )


__all__ = [
    "CommitPortToolEffectGuard",
    "ModelStepReservation",
    "RuntimeCommitPort",
    "RuntimeModelStepCommit",
    "RuntimeSuspension",
    "RuntimeSuspensionResult",
    "RuntimeToolCall",
    "RuntimeToolResultCommit",
    "ToolCallClaim",
    "build_runtime_tool_call",
]
