"""Live plane 的 wire-neutral public models。

本模块只定义 raw/public 边界；broker 内部排序状态与 transport framing 不在这里。

Example:
    cursor = LiveCursor(
        stream_epoch="epoch-1",
        scope="run",
        scope_id="run-1",
        after_live_sequence=0,
    )
"""

# region imports

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..harness.session_manager import ResumeReceipt, SubmissionMode, SubmitReceipt
from ..hitl import HumanInteractionResponse
from ..lifecycle import (
    AgentRunOptions,
    RunEvent,
    RunResult,
    RunSnapshot,
    RunToolCallRecord,
    validate_json_safe,
)

# endregion

type LiveScope = Literal["run", "session"]
type ReplayGapReason = Literal[
    "epoch_changed",
    "cursor_expired",
    "unknown_cursor",
    "slow_consumer",
]
type SubscriptionTerminalReason = Literal["slow_consumer", "broker_closed"]
type GatewayCommandKind = Literal["subscribe", "submit", "resume", "cancel", "sync", "snapshot"]

_NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_StableCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Z][A-Z0-9_]*$"),
]


class _FrozenWireModel(BaseModel):
    """Live wire models 共用的 immutable/strict-extra 配置。"""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class LiveCursor(_FrozenWireModel):
    """一个 epoch 内特定 scope 的 live replay cursor。"""

    stream_epoch: _NonEmptyString
    scope: LiveScope
    scope_id: _NonEmptyString
    after_live_sequence: int = Field(ge=0, strict=True)


class DurableRunCursor(_FrozenWireModel):
    """按 run durable sequence 读取的 cursor。"""

    run_id: _NonEmptyString
    after_sequence: int = Field(ge=0, strict=True)


class LiveEnvelope(_FrozenWireModel):
    """Broker 分配 live sequence 后的 remote-safe event。"""

    stream_epoch: _NonEmptyString
    scope: LiveScope
    scope_id: _NonEmptyString
    live_sequence: int = Field(ge=1, strict=True)
    kind: _NonEmptyString
    run_id: _NonEmptyString | None = None
    session_id: _NonEmptyString | None = None
    activation_id: _NonEmptyString | None = None
    durable_sequence: int | None = Field(default=None, ge=1, strict=True)
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _validate_payload(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """拒绝 NaN 等非严格 JSON 值。"""
        validate_json_safe(value, field_name="payload")
        return value


class ReplayGap(_FrozenWireModel):
    """说明 live replay 或 delivery 出现需要 durable sync 的缺口。"""

    kind: Literal["replay.gap"] = "replay.gap"
    reason: ReplayGapReason
    requested_cursor: LiveCursor | None = None
    current_epoch: _NonEmptyString


class SubscriptionTerminal(_FrozenWireModel):
    """结束单个 live subscription 的安全 control item。"""

    kind: Literal["subscription.terminal"] = "subscription.terminal"
    reason: SubscriptionTerminalReason
    message: _NonEmptyString


type LiveStreamItem = LiveEnvelope | ReplayGap | SubscriptionTerminal


class LiveSubscriptionRequest(_FrozenWireModel):
    """创建 run/session live subscription 的边界请求。"""

    scope: LiveScope
    scope_id: _NonEmptyString
    cursor: LiveCursor | None = None

    @model_validator(mode="after")
    def _validate_cursor_scope(self) -> LiveSubscriptionRequest:
        if self.cursor is not None and (
            self.cursor.scope != self.scope or self.cursor.scope_id != self.scope_id
        ):
            raise ValueError("cursor 必须与 subscription scope 和 scope_id 一致")
        return self


def _require_unique_durable_cursors(
    cursors: tuple[DurableRunCursor, ...],
) -> tuple[DurableRunCursor, ...]:
    """拒绝同一 command 中重复的 run cursor。"""
    run_ids = [cursor.run_id for cursor in cursors]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("durable cursors 包含 duplicate run_id")
    return cursors


class SubscribeCommand(_FrozenWireModel):
    """创建 live subscription 并可附带 durable sync cursors。"""

    kind: Literal["subscribe"] = "subscribe"
    request_id: _NonEmptyString
    scope: LiveScope
    scope_id: _NonEmptyString
    cursor: LiveCursor | None = None
    durable_cursors: tuple[DurableRunCursor, ...] = ()

    @field_validator("durable_cursors")
    @classmethod
    def _validate_durable_cursors(
        cls,
        value: tuple[DurableRunCursor, ...],
    ) -> tuple[DurableRunCursor, ...]:
        return _require_unique_durable_cursors(value)

    @model_validator(mode="after")
    def _validate_cursor_scope(self) -> SubscribeCommand:
        if self.cursor is not None and (
            self.cursor.scope != self.scope or self.cursor.scope_id != self.scope_id
        ):
            raise ValueError("cursor 必须与 subscribe scope 和 scope_id 一致")
        return self


class SubmitCommand(_FrozenWireModel):
    """向 bound session manager 提交普通输入。"""

    kind: Literal["submit"] = "submit"
    request_id: _NonEmptyString
    input: _NonEmptyString
    mode: SubmissionMode | None = None
    options: AgentRunOptions | None = None


class ResumeCommand(_FrozenWireModel):
    """提交 typed HITL response。"""

    kind: Literal["resume"] = "resume"
    request_id: _NonEmptyString
    interaction_id: _NonEmptyString
    response: HumanInteractionResponse


class CancelCommand(_FrozenWireModel):
    """请求中断 bound session 的当前 run。"""

    kind: Literal["cancel"] = "cancel"
    request_id: _NonEmptyString
    reason: _NonEmptyString | None = None


class SyncCommand(_FrozenWireModel):
    """按 caller 已知 run cursors 请求有限事件页。"""

    kind: Literal["sync"] = "sync"
    request_id: _NonEmptyString
    cursors: tuple[DurableRunCursor, ...] = ()

    @field_validator("cursors")
    @classmethod
    def _validate_cursors(
        cls,
        value: tuple[DurableRunCursor, ...],
    ) -> tuple[DurableRunCursor, ...]:
        return _require_unique_durable_cursors(value)


class SnapshotCommand(_FrozenWireModel):
    """显式读取 caller 已知 run 的完整状态。"""

    kind: Literal["snapshot"] = "snapshot"
    request_id: _NonEmptyString
    run_ids: tuple[_NonEmptyString, ...] = ()


GatewayCommand = Annotated[
    SubmitCommand | ResumeCommand | CancelCommand | SyncCommand | SnapshotCommand,
    Field(discriminator="kind"),
]


class DurableRunPage(_FrozenWireModel):
    """一个 caller 已知 run 的有限事件页，不附带状态快照。"""

    run_id: _NonEmptyString
    events: tuple[RunEvent, ...] = ()
    next_cursor: DurableRunCursor | None = None


class DurableSync(_FrozenWireModel):
    """Bound session 的 ordered durable run pages。"""

    session_id: _NonEmptyString
    runs: tuple[DurableRunPage, ...] = ()


class DurableSyncItem(_FrozenWireModel):
    """Subscription initial path 的 out-of-band durable 事件页。"""

    kind: Literal["sync.page"] = "sync.page"
    sync: DurableSync


type GatewayStreamItem = LiveStreamItem | DurableSyncItem


class DurableRunSnapshot(_FrozenWireModel):
    """一个 run 按需读取的状态、结果与完整工具记录。"""

    run: RunSnapshot
    result: RunResult | None = None
    tool_calls: tuple[RunToolCallRecord, ...] = ()


class DurableSnapshot(_FrozenWireModel):
    """Bound session 中 caller 指定 run 的有序状态快照。"""

    session_id: _NonEmptyString
    runs: tuple[DurableRunSnapshot, ...] = ()


class SubmitAccepted(_FrozenWireModel):
    """Submit command admission receipt。"""

    event: Literal["command.submit.accepted"] = "command.submit.accepted"
    request_id: _NonEmptyString
    receipt: SubmitReceipt


class ResumeAccepted(_FrozenWireModel):
    """Resume command admission receipt；运行结果由 durable observation 获取。"""

    event: Literal["command.resume.accepted"] = "command.resume.accepted"
    request_id: _NonEmptyString
    receipt: ResumeReceipt


class CancelAccepted(_FrozenWireModel):
    """Cancel command snapshot receipt。"""

    event: Literal["command.cancel.accepted"] = "command.cancel.accepted"
    request_id: _NonEmptyString
    run: RunSnapshot


class SyncAccepted(_FrozenWireModel):
    """Sync command receipt。"""

    event: Literal["command.sync.accepted"] = "command.sync.accepted"
    request_id: _NonEmptyString
    sync: DurableSync


class SnapshotAccepted(_FrozenWireModel):
    """Snapshot command 的按需状态快照回执。"""

    event: Literal["command.snapshot.accepted"] = "command.snapshot.accepted"
    request_id: _NonEmptyString
    snapshot: DurableSnapshot


class SubscribeAccepted(_FrozenWireModel):
    """Subscribe command receipt。"""

    event: Literal["command.subscribe.accepted"] = "command.subscribe.accepted"
    request_id: _NonEmptyString
    stream_epoch: _NonEmptyString
    scope: LiveScope
    scope_id: _NonEmptyString


class CommandRejected(_FrozenWireModel):
    """不暴露 traceback/context 的安全 command rejection。"""

    event: Literal["command.rejected"] = "command.rejected"
    request_id: _NonEmptyString | None = None
    command_kind: GatewayCommandKind | None = None
    code: _StableCode
    message: _NonEmptyString


CommandReceipt = Annotated[
    SubmitAccepted
    | ResumeAccepted
    | CancelAccepted
    | SyncAccepted
    | SnapshotAccepted
    | SubscribeAccepted
    | CommandRejected,
    Field(discriminator="event"),
]


__all__ = [
    "CancelAccepted",
    "CancelCommand",
    "CommandReceipt",
    "CommandRejected",
    "DurableRunCursor",
    "DurableRunPage",
    "DurableRunSnapshot",
    "DurableSnapshot",
    "DurableSync",
    "DurableSyncItem",
    "GatewayCommand",
    "GatewayCommandKind",
    "GatewayStreamItem",
    "LiveCursor",
    "LiveEnvelope",
    "LiveScope",
    "LiveStreamItem",
    "LiveSubscriptionRequest",
    "ReplayGap",
    "ReplayGapReason",
    "ResumeAccepted",
    "ResumeCommand",
    "SnapshotAccepted",
    "SnapshotCommand",
    "SubmitAccepted",
    "SubmitCommand",
    "SubscribeAccepted",
    "SubscribeCommand",
    "SubscriptionTerminal",
    "SubscriptionTerminalReason",
    "SyncAccepted",
    "SyncCommand",
]
