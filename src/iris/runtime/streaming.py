"""Runtime live streaming 事件与工具 effect guard 适配。

本模块只描述同进程 live facts；事件不会进入 durable commit、checkpoint 或 store。

Example:
    sink.emit(event)
"""

# region imports

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ..message import ModelStreamEvent
from ..tools import PreparedToolCall, ToolResult
from .commit import (
    CommitPortToolEffectGuard,
    RuntimeToolCall,
    ToolCallClaim,
)
from .models import RuntimeActivationInput

# endregion

_RuntimeStreamEventKind = Literal[
    "model.step.started",
    "model.event",
    "tool.preparing",
    "tool.started",
    "tool.completed",
]


@dataclass(frozen=True, slots=True)
class RuntimeStreamEvent:
    """Runtime 顺序产生的一条同进程 live fact。

    Attributes:
        kind (str): Runtime fact 类型。
        run_id (str): Logical run identity。
        session_id (str): Session identity。
        activation_id (str): 当前 activation identity。
        step_index (int): 当前 model step。
        model_event (ModelStreamEvent | None): Provider-neutral model event。
        tool_call_id (str | None): Tool call identity。
        tool_name (str | None): Tool 名称。
        tool_ordinal (int | None): 1-based model order。
        tool_result (ToolResult | None): Durable commit 成功后的工具结果。
    """

    kind: _RuntimeStreamEventKind
    run_id: str
    session_id: str
    activation_id: str
    step_index: int
    model_event: ModelStreamEvent | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_ordinal: int | None = None
    tool_result: ToolResult | None = None


class RuntimeEventSink(Protocol):
    """同步接收 runtime live event 的最小协议。"""

    def emit(self, event: RuntimeStreamEvent) -> None:
        """同步发布一条 live event。

        Args:
            event (RuntimeStreamEvent): 当前 runtime 已确认的 live fact。
        """


def _runtime_stream_event(
    kind: _RuntimeStreamEventKind,
    *,
    activation: RuntimeActivationInput,
    step_index: int,
    model_event: ModelStreamEvent | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_ordinal: int | None = None,
    tool_result: ToolResult | None = None,
) -> RuntimeStreamEvent:
    """从 trusted runtime objects 构造唯一的 live event 形状。"""
    return RuntimeStreamEvent(
        kind=kind,
        run_id=activation.run_id,
        session_id=activation.session_id,
        activation_id=activation.activation_id,
        step_index=step_index,
        model_event=model_event,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_ordinal=tool_ordinal,
        tool_result=tool_result,
    )


class _LiveToolEffectGuard:
    """在 durable claim 成功后发布 ``tool.started``。"""

    def __init__(
        self,
        *,
        guard: CommitPortToolEffectGuard,
        sink: RuntimeEventSink,
        activation: RuntimeActivationInput,
        step_index: int,
        tool_ordinal: int,
    ) -> None:
        self._guard = guard
        self._sink = sink
        self._activation = activation
        self._step_index = step_index
        self._tool_ordinal = tool_ordinal

    def before_effect(self, prepared: PreparedToolCall) -> None:
        """先执行既有 effect guard，再同步发布 started fact。"""
        self._guard.before_effect(prepared)
        self._sink.emit(
            _runtime_stream_event(
                "tool.started",
                activation=self._activation,
                step_index=self._step_index,
                tool_call_id=prepared.tool_use.id,
                tool_name=prepared.tool_use.name,
                tool_ordinal=self._tool_ordinal,
            )
        )

    def claim_for(self, tool_call_id: str) -> ToolCallClaim | None:
        """返回 underlying guard 已确认的 durable claim。"""
        return self._guard.claim_for(tool_call_id)

    def call_for(self, tool_call_id: str) -> RuntimeToolCall | None:
        """返回 underlying guard 对应的 runtime tool fact。"""
        return self._guard.call_for(tool_call_id)


__all__ = [
    "RuntimeEventSink",
    "RuntimeStreamEvent",
]
