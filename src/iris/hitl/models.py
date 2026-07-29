"""HITL 领域的 JSON-safe 数据契约。"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class InteractionKind(StrEnum):
    """人工交互的请求类别。"""

    PERMISSION = "permission"
    QUESTION = "question"


class InteractionStatus(StrEnum):
    """人工响应的生命周期状态。"""

    PENDING = "pending"
    RESOLVED = "resolved"
    CONSUMED = "consumed"
    CLOSED = "closed"


class InteractionResumePhase(StrEnum):
    """Runtime 消费人工响应后的恢复进度。"""

    WAITING = "waiting"
    CLAIMED = "claimed"
    RESULT_READY = "result_ready"
    RESULT_COMMITTED = "result_committed"


def _new_interaction_id() -> str:
    return f"int_{uuid.uuid4().hex}"


def _now() -> datetime:
    return datetime.now(UTC)


def _validate_json_safe(value: Any, *, field_name: str) -> Any:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是 JSON-safe 数据") from exc
    return value


def _trim_required(value: str, *, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} 不能为空")
    return value


def make_call_fingerprint(
    *,
    session_id: str,
    run_id: str,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    workspace_root: str,
) -> str:
    """为精确的一次工具调用生成稳定 SHA-256 指纹。"""
    payload = {
        "arguments": arguments,
        "run_id": run_id,
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "workspace_root": workspace_root,
    }
    canonical_json = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class ToolCallSnapshot(BaseModel):
    """触发人工 gate 的精确工具调用身份。"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    workspace_root: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @field_validator("tool_call_id", "tool_name", "workspace_root", "fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("arguments")
    @classmethod
    def _validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, field_name="arguments")


class PermissionPrompt(BaseModel):
    """向人展示的一次工具权限确认。"""

    kind: Literal[InteractionKind.PERMISSION] = InteractionKind.PERMISSION
    reason: str

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _trim_required(value, field_name="reason")


class QuestionPrompt(BaseModel):
    """向人展示的一次信息问题。"""

    kind: Literal[InteractionKind.QUESTION] = InteractionKind.QUESTION
    question: str
    options: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @field_validator("question")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: list[str]) -> list[str]:
        normalized = [_trim_required(option, field_name="option") for option in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("options 不能包含重复项")
        return normalized


class PermissionInteractionResponse(BaseModel):
    """人工对权限请求的单次决定。"""

    kind: Literal[InteractionKind.PERMISSION] = InteractionKind.PERMISSION
    decision: Literal["approve", "reject"]

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class QuestionInteractionResponse(BaseModel):
    """人工对问题请求的自由文本回答。"""

    kind: Literal[InteractionKind.QUESTION] = InteractionKind.QUESTION
    answer: str

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @field_validator("answer")
    @classmethod
    def _validate_answer(cls, value: str) -> str:
        return _trim_required(value, field_name="answer")


HumanInteractionPrompt = Annotated[
    PermissionPrompt | QuestionPrompt,
    Field(discriminator="kind"),
]
HumanInteractionResponse = Annotated[
    PermissionInteractionResponse | QuestionInteractionResponse,
    Field(discriminator="kind"),
]


class ApprovedToolCall(BaseModel):
    """人工批准后返回给 lifecycle owner 的纯工具调用投影。"""

    interaction_id: str
    tool_call_id: str
    tool_name: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("interaction_id", "tool_call_id", "tool_name", "fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))


class HumanInteractionRequest(BaseModel):
    """所有人工 gate 共用的工具调用与提示信封。"""

    tool_call: ToolCallSnapshot
    prompt: HumanInteractionPrompt

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class HumanInteraction(BaseModel):
    """持久化的一次人工 gate 与其恢复状态。"""

    interaction_id: str = Field(default_factory=_new_interaction_id)
    session_id: str
    run_id: str
    step_index: int = Field(ge=0)
    tool_call_id: str
    status: InteractionStatus = InteractionStatus.PENDING
    resume_phase: InteractionResumePhase = InteractionResumePhase.WAITING
    request: HumanInteractionRequest
    response: HumanInteractionResponse | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=_now)
    expires_at: datetime | None = None
    resolved_at: datetime | None = None
    consumed_at: datetime | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    @model_validator(mode="before")
    @classmethod
    def _derive_legacy_tool_call_id(cls, value: Any) -> Any:
        """让 Phase 5 删除前的旧构造路径自动补齐显式 subject identity。"""
        if not isinstance(value, Mapping) or value.get("tool_call_id") is not None:
            return value
        request = value.get("request")
        tool_call = getattr(request, "tool_call", None)
        tool_call_id = getattr(tool_call, "tool_call_id", None)
        if tool_call_id is None and isinstance(request, Mapping):
            tool_call = request.get("tool_call")
            tool_call_id = tool_call.get("tool_call_id") if isinstance(tool_call, Mapping) else None
        return dict(value) | ({"tool_call_id": tool_call_id} if tool_call_id is not None else {})

    @field_validator("interaction_id", "session_id", "run_id", "tool_call_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, field_name="checkpoint")

    @field_validator("close_reason")
    @classmethod
    def _validate_close_reason(cls, value: str | None) -> str | None:
        return None if value is None else _trim_required(value, field_name="close_reason")

    @field_validator("created_at", "expires_at", "resolved_at", "consumed_at", "closed_at")
    @classmethod
    def _validate_target_times(
        cls, value: datetime | None, info: ValidationInfo
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{info.field_name} 必须包含时区")
        return value

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> HumanInteraction:
        request = getattr(self, "request", None)
        tool_call_id = getattr(self, "tool_call_id", None)
        expected_tool_call_id = getattr(
            getattr(request, "tool_call", None),
            "tool_call_id",
            None,
        )
        if request is not None and tool_call_id != expected_tool_call_id:
            raise ValueError("interaction tool_call_id 必须匹配 request subject")
        if self.response is not None and self.response.kind != self.request.prompt.kind:
            raise ValueError("interaction prompt kind 必须匹配 response kind")
        if self.status is InteractionStatus.PENDING and self.response is not None:
            raise ValueError("pending interaction 不能包含 response")
        if (
            self.status in {InteractionStatus.RESOLVED, InteractionStatus.CONSUMED}
            and self.response is None
        ):
            raise ValueError("resolved 或 consumed interaction 必须包含 response")
        if (
            self.status in {InteractionStatus.PENDING, InteractionStatus.RESOLVED}
            and self.resume_phase is not InteractionResumePhase.WAITING
        ):
            raise ValueError("pending 或 resolved interaction 的 resume_phase 必须是 waiting")
        if (
            self.status is InteractionStatus.CONSUMED
            and self.resume_phase is InteractionResumePhase.WAITING
        ):
            raise ValueError("consumed interaction 的 resume_phase 不能是 waiting")
        if self.status is InteractionStatus.CLOSED:
            if self.closed_at is None or self.close_reason is None:
                raise ValueError("closed interaction 必须包含 closed_at 与 close_reason")
        elif self.closed_at is not None or self.close_reason is not None:
            raise ValueError("非 closed interaction 不能包含关闭事实")
        return self


__all__ = [
    "ApprovedToolCall",
    "HumanInteraction",
    "HumanInteractionPrompt",
    "HumanInteractionRequest",
    "HumanInteractionResponse",
    "InteractionKind",
    "InteractionResumePhase",
    "InteractionStatus",
    "PermissionPrompt",
    "PermissionInteractionResponse",
    "QuestionPrompt",
    "QuestionInteractionResponse",
    "ToolCallSnapshot",
    "make_call_fingerprint",
]
