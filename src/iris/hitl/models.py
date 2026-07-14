"""HITL 领域的 JSON-safe 数据契约。"""

from __future__ import annotations

import hashlib
import json
import uuid
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


class InteractionResumePhase(StrEnum):
    """Runtime 消费人工响应后的恢复进度。"""

    WAITING = "waiting"
    CLAIMED = "claimed"
    RESULT_READY = "result_ready"
    RESULT_COMMITTED = "result_committed"


def _new_interaction_id() -> str:
    return f"int_{uuid.uuid4().hex}"


def _now() -> datetime:
    return datetime.now()


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


class PermissionInteractionRequest(BaseModel):
    """一次等待人工确认的工具权限请求。"""

    kind: Literal[InteractionKind.PERMISSION] = InteractionKind.PERMISSION
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    workspace_root: str
    call_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @field_validator("tool_call_id", "tool_name", "reason", "workspace_root", "call_fingerprint")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("arguments")
    @classmethod
    def _validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, field_name="arguments")


class QuestionInteractionRequest(BaseModel):
    """一次等待人工回答的问题请求。"""

    kind: Literal[InteractionKind.QUESTION] = InteractionKind.QUESTION
    tool_call_id: str
    question: str
    options: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @field_validator("tool_call_id", "question")
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


HumanInteractionRequest = Annotated[
    PermissionInteractionRequest | QuestionInteractionRequest,
    Field(discriminator="kind"),
]
HumanInteractionResponse = Annotated[
    PermissionInteractionResponse | QuestionInteractionResponse,
    Field(discriminator="kind"),
]


class HumanInteraction(BaseModel):
    """持久化的一次人工 gate 与其恢复状态。"""

    interaction_id: str = Field(default_factory=_new_interaction_id, pattern=r"^int_[0-9a-f]{32}$")
    session_id: str
    run_id: str
    step_index: int = Field(ge=0)
    tool_call_id: str
    kind: InteractionKind
    status: InteractionStatus = InteractionStatus.PENDING
    resume_phase: InteractionResumePhase = InteractionResumePhase.WAITING
    request: HumanInteractionRequest
    response: HumanInteractionResponse | None = None
    checkpoint: dict[str, Any]
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=_now)
    resolved_at: datetime | None = None
    consumed_at: datetime | None = None

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @field_validator("interaction_id", "session_id", "run_id", "tool_call_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _trim_required(value, field_name=str(info.field_name))

    @field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, field_name="checkpoint")

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> HumanInteraction:
        if self.request.kind != self.kind:
            raise ValueError("interaction kind 必须匹配 request kind")
        if self.request.tool_call_id != self.tool_call_id:
            raise ValueError("interaction tool_call_id 必须匹配 request tool_call_id")
        if self.response is not None and self.response.kind != self.kind:
            raise ValueError("interaction kind 必须匹配 response kind")
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
        return self


__all__ = [
    "HumanInteraction",
    "HumanInteractionRequest",
    "HumanInteractionResponse",
    "InteractionKind",
    "InteractionResumePhase",
    "InteractionStatus",
    "PermissionInteractionRequest",
    "PermissionInteractionResponse",
    "QuestionInteractionRequest",
    "QuestionInteractionResponse",
    "make_call_fingerprint",
]
