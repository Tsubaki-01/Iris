"""Harness live plane 的 trusted fact 与 publisher 合同。

本模块只桥接同进程 typed facts，不分配 live cursor，也不执行 fan-out。

Example:
    sink = _RuntimeLiveSink(publisher)
"""

# region imports

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..goal.models import GoalChanged
from ..lifecycle import RunErrorInfo, RunEvent
from ..lifecycle.history import RunLineage
from ..observability.facts import ConfigurationApplied, SourceAdopted
from ..runtime import RuntimeEventSink, RuntimeStreamEvent
from ..runtime.diagnostics import ContextPreparation
from .control import SessionControlSnapshot
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


@dataclass(frozen=True, slots=True)
class SessionControlChanged:
    """Manager 已原子替换的完整控制视图，必须作为 critical 事件传送。"""

    snapshot: SessionControlSnapshot


@dataclass(frozen=True, slots=True)
class CommandCleanupFailed:
    """本次异步执行环境清理失败，run 仍等待结算而非 durable terminal。"""

    run_id: str
    session_id: str
    error: RunErrorInfo


@dataclass(frozen=True, slots=True)
class SubagentLinked:
    """Admission 已确认的持久 child 关系。"""

    lineage: RunLineage


type UnscopedLiveFact = (
    RuntimeStreamEvent
    | RunEvent
    | SessionSubmissionEvent
    | CommandCleanupFailed
    | GoalChanged
    | SessionControlChanged
    | SubagentLinked
    | ContextPreparation
    | ConfigurationApplied
    | SourceAdopted
)


@dataclass(frozen=True, slots=True)
class LineagedLiveFact:
    """原事实及固定 lineage；不改写原 run/session 身份。"""

    fact: UnscopedLiveFact
    lineage: RunLineage


type LiveFact = UnscopedLiveFact | LineagedLiveFact


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

    def emit(self, event: RuntimeStreamEvent | ContextPreparation) -> None:
        """把 runtime live event 原样发布。"""
        self._publisher.publish(event)


class _LineagePublisher:
    """Child 借用的 publisher，在发布时补充已固定的关系。"""

    def __init__(self, publisher: LivePublisher, lineage: RunLineage) -> None:
        self._publisher = publisher
        self.lineage = lineage

    def publish(self, fact: LiveFact) -> None:
        """嵌套 child 已携带自身关系时原样转交。"""
        self._publisher.publish(
            fact if isinstance(fact, LineagedLiveFact) else LineagedLiveFact(fact, self.lineage)
        )


__all__ = [
    "RunLineage",
    "SubagentLinked",
    "LineagedLiveFact",
    "CommandCleanupFailed",
    "LiveFact",
    "LivePublisher",
    "SessionSubmissionEvent",
    "SessionControlChanged",
]
