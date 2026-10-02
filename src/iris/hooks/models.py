"""Hooks 的轻量公共事件、注册边界与事件输出投影。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from ..lifecycle.models import RunResult, RunSnapshot
    from ..message import DataBlock
    from ..tools.base import ToolResult

type HookEventName = Literal["run.started", "run.finished", "tool.before", "tool.after"]
type _NonemptyText = Annotated[str, Field(pattern=r"\S")]


def _utc_now() -> datetime:
    """产生事件关联所需的 UTC 时间。"""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class _EventContext:
    """已由执行 owner 确定的身份，不包含任何可写执行上下文。"""

    agent_id: str
    session_id: str
    workspace: str
    run_id: str | None = None
    activation_id: str | None = None
    occurred_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStartedEvent(_EventContext):
    """新 logical run 开始时的输入与运行快照。"""

    run: RunSnapshot
    input: str | list[DataBlock]
    event: Literal["run.started"] = field(default="run.started", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFinishedEvent(_EventContext):
    """来自本次新终态提交的运行结果。"""

    result: RunResult
    event: Literal["run.finished"] = field(default="run.finished", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolBeforeEvent(_EventContext):
    """普通工具执行前已经过参数及权限检查的调用快照。"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    event: Literal["tool.before"] = field(default="tool.before", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolAfterEvent(_EventContext):
    """真实 body 已执行且结果已知时的只读结果快照。"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult
    body_status: Literal["success", "error"]
    event: Literal["tool.after"] = field(default="tool.after", init=False)


type HookEvent = RunStartedEvent | RunFinishedEvent | ToolBeforeEvent | ToolAfterEvent


class ToolBeforeResult(BaseModel):
    """拒绝当前调用的原因；None 表示继续。"""

    deny_reason: _NonemptyText
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolAfterResult(BaseModel):
    """在真实工具结果之后追加的一段模型反馈。"""

    feedback: _NonemptyText
    model_config = ConfigDict(frozen=True, extra="forbid")


type HookResult = ToolBeforeResult | ToolAfterResult | None
type HookHandler = Callable[[HookEvent], Awaitable[HookResult]]


class HookRegistration(BaseModel):
    """SDK 注册边界；handler 只接受事件，工具过滤使用实际调用名。"""

    event: HookEventName
    name: _NonemptyText
    handler: HookHandler
    tool_names: Annotated[tuple[_NonemptyText, ...], Field(min_length=1)] | None = None
    timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_tool_filter(self) -> Self:
        """工具过滤字段只适用于工具事件。"""
        if self.tool_names is not None and self.event not in {"tool.before", "tool.after"}:
            raise ValueError("tool_names 只能用于工具事件")
        return self


def event_to_dict(event: HookEvent) -> dict[str, Any]:
    """将可信事件投影为脚本可编码的字段，不运行时导入领域模型。"""
    payload: dict[str, Any] = {
        "event": event.event,
        "agent_id": event.agent_id,
        "session_id": event.session_id,
        "run_id": event.run_id,
        "activation_id": event.activation_id,
        "workspace": event.workspace,
        "occurred_at": event.occurred_at.isoformat(),
    }
    if isinstance(event, RunStartedEvent):
        payload.update(
            run=event.run.model_dump(mode="json"),
            input=event.input
            if isinstance(event.input, str)
            else [block.model_dump(mode="json") for block in event.input],
        )
    elif isinstance(event, RunFinishedEvent):
        payload["result"] = event.result.model_dump(mode="json")
    else:
        payload.update(call_id=event.call_id, tool_name=event.tool_name, arguments=event.arguments)
        if isinstance(event, ToolAfterEvent):
            payload.update(
                result=event.result.model_dump(mode="json"), body_status=event.body_status
            )
    return payload
