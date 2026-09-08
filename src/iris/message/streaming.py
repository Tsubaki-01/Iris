"""Provider-neutral 模型流式事件。

本模块定义 provider raw stream 离开 adapter 前必须转换成的稳定事件边界。

Example:
    >>> scope = ModelStreamScope(
    ...     model_stream_id="stream-1",
    ...     provider="openai",
    ...     model="gpt-4o",
    ...     attempt=1,
    ... )
    >>> scope.attempt
    1
"""

# region imports

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from .llm import LLMResponse

# endregion

# ==========================================
#                 类型常量
# ==========================================
# region constants

_NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelBlockKind = Literal["text", "thinking", "tool_call"]
ModelDeltaChannel = Literal["text", "thinking", "tool_name", "tool_arguments"]

# endregion


class ModelStreamScope(BaseModel):
    """标识一次 provider 流式调用。"""

    model_stream_id: _NonEmptyString
    provider: _NonEmptyString
    model: _NonEmptyString
    attempt: int = Field(ge=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelBlockRef(BaseModel):
    """标识一次模型响应中的内容块。"""

    index: int = Field(ge=0)
    block_id: _NonEmptyString
    kind: ModelBlockKind
    tool_call_id: _NonEmptyString | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_tool_call_id(self) -> Self:
        """校验工具块与工具调用标识的一致性。"""
        if self.kind == "tool_call" and self.tool_call_id is None:
            raise ValueError("tool_call block 必须提供 tool_call_id")
        if self.kind != "tool_call" and self.tool_call_id is not None:
            raise ValueError("非 tool_call block 不得提供 tool_call_id")
        return self


class ModelUsageSnapshot(BaseModel):
    """描述流式响应当前已知的 token 用量。"""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    complete: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderStreamError(BaseModel):
    """可安全暴露的 provider 流式错误。"""

    code: Annotated[
        str,
        StringConstraints(strip_whitespace=True, pattern=r"^[A-Z][A-Z0-9_]*$"),
    ]
    message: _NonEmptyString
    retryable: bool

    model_config = ConfigDict(frozen=True, extra="forbid")


class _ModelStreamEventBase(BaseModel):
    """所有 provider-neutral 流式事件的公共字段。"""

    scope: ModelStreamScope
    sequence: int = Field(ge=1)
    occurred_at: AwareDatetime

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelResponseStarted(_ModelStreamEventBase):
    """表示 provider 已开始一次模型响应。"""

    kind: Literal["response.started"] = "response.started"
    response_id: _NonEmptyString


class ModelBlockStarted(_ModelStreamEventBase):
    """表示一个模型内容块已经开始。"""

    kind: Literal["block.started"] = "block.started"
    block: ModelBlockRef


class ModelBlockDelta(_ModelStreamEventBase):
    """表示一个模型内容块的相对增量。"""

    kind: Literal["block.delta"] = "block.delta"
    block: ModelBlockRef
    channel: ModelDeltaChannel
    delta: str = Field(min_length=1)
    snapshot: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_channel(self) -> Self:
        """校验 delta channel 与 block kind 相容。"""
        allowed_channels: dict[ModelBlockKind, set[ModelDeltaChannel]] = {
            "text": {"text"},
            "thinking": {"thinking"},
            "tool_call": {"tool_name", "tool_arguments"},
        }
        if self.channel not in allowed_channels[self.block.kind]:
            raise ValueError("delta channel 与 block kind 不相容")
        return self


class ModelBlockCompleted(_ModelStreamEventBase):
    """表示一个模型内容块已经完成。"""

    kind: Literal["block.completed"] = "block.completed"
    block: ModelBlockRef


class ModelUsageUpdated(_ModelStreamEventBase):
    """表示 provider 返回了新的 token 用量快照。"""

    kind: Literal["usage.updated"] = "usage.updated"
    usage: ModelUsageSnapshot


class ModelResponseCompleted(_ModelStreamEventBase):
    """表示 provider 已产生完整且可提交的模型响应。"""

    kind: Literal["response.completed"] = "response.completed"
    response: LLMResponse
    semantic_output_emitted: bool


class ModelResponseFailed(_ModelStreamEventBase):
    """表示 provider 流式响应失败。"""

    kind: Literal["response.failed"] = "response.failed"
    error: ProviderStreamError
    semantic_output_emitted: bool


class ModelResponseCancelled(_ModelStreamEventBase):
    """表示 provider 主动取消流式响应。"""

    kind: Literal["response.cancelled"] = "response.cancelled"
    error: ProviderStreamError | None = None
    semantic_output_emitted: bool


ModelStreamEvent = Annotated[
    ModelResponseStarted
    | ModelBlockStarted
    | ModelBlockDelta
    | ModelBlockCompleted
    | ModelUsageUpdated
    | ModelResponseCompleted
    | ModelResponseFailed
    | ModelResponseCancelled,
    Field(discriminator="kind"),
]


__all__ = [
    "ModelBlockCompleted",
    "ModelBlockDelta",
    "ModelBlockKind",
    "ModelBlockRef",
    "ModelBlockStarted",
    "ModelDeltaChannel",
    "ModelResponseCancelled",
    "ModelResponseCompleted",
    "ModelResponseFailed",
    "ModelResponseStarted",
    "ModelStreamEvent",
    "ModelStreamScope",
    "ModelUsageSnapshot",
    "ModelUsageUpdated",
    "ProviderStreamError",
]
