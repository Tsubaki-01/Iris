"""Harness live plane 的 trusted fact 与 publisher 合同。

本模块只桥接同进程 typed facts，不分配 live cursor，也不执行 fan-out。

Example:
    sink = _RuntimeLiveSink(publisher)
"""

# region imports

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..lifecycle import RunEvent
from ..runtime import RuntimeEventSink, RuntimeStreamEvent
from .session_manager import SubmissionEvent

# endregion


@dataclass(frozen=True, slots=True)
class SessionSubmissionEvent:
    """为 process-local submission event 补充 trusted session identity。

    Attributes:
        session_id (str): Bound manager 已验证的 session identity。
        event (SubmissionEvent): 原 public submission event。
    """

    session_id: str
    event: SubmissionEvent


type LiveFact = RuntimeStreamEvent | RunEvent | SessionSubmissionEvent


class LivePublisher(Protocol):
    """同步接收 harness live fact 的最小协议。"""

    def publish(self, fact: LiveFact) -> None:
        """发布一条 trusted fact。

        Args:
            fact (LiveFact): Runtime、durable 或 submission fact。
        """


class _RuntimeLiveSink(RuntimeEventSink):
    """把 runtime sink 调用直接委托给 harness publisher。"""

    def __init__(self, publisher: LivePublisher) -> None:
        self._publisher = publisher

    def emit(self, event: RuntimeStreamEvent) -> None:
        """把 runtime live event 原样发布。"""
        self._publisher.publish(event)


__all__ = ["LiveFact", "LivePublisher", "SessionSubmissionEvent"]
