"""将已提交会话原文投影成不可变记忆材料，保留来源而不重复学习读回记忆。"""

from __future__ import annotations

from ..lifecycle import RunRecord
from ..lifecycle.history import RunMessageSlice
from ..memory.generation_models import MemoryCaptureSource
from ..memory.models import MemoryEpisode, MemoryRecord, MemorySourceType
from ._capture_records import capture_records


def capture_source(run: RunRecord, *, source_id: str, namespace: str) -> MemoryCaptureSource:
    """以 run admission 时的原文截点构造首次登记，不纳入继承历史。"""
    return MemoryCaptureSource(
        lifecycle_source_id=source_id,
        run_id=run.run_id,
        session_id=run.session_id,
        namespace=namespace,
        initial_message_count=run.initial_session_message_count,
        captured_until=run.initial_session_message_count,
    )


def capture_episode(
    source: MemoryCaptureSource, messages: RunMessageSlice, *, through_count: int | None = None
) -> tuple[MemoryCaptureSource, MemoryEpisode | None]:
    """只投影选中已提交后缀，终态封口不越过调用方提示的范围。"""
    end = messages.end_message_count
    if through_count is not None:
        end = max(messages.start_message_count, min(end, through_count))
    records = tuple(
        MemoryRecord(
            id=record.ref,
            role=record.role,
            text=record.text,
            source_type=MemorySourceType.TOOL_EVENT
            if record.tool_event
            else MemorySourceType.MESSAGE,
            source_id=record.ref,
            occurred_at=record.occurred_at,
            metadata=record.metadata,
        )
        for record in capture_records(messages, end=end)
    )
    terminal = messages.terminal_message_count if end == messages.terminal_message_count else None
    updated = source.model_copy(
        update={
            "captured_until": end,
            "terminal_message_count": terminal,
            "outcome": messages.outcome.value
            if terminal is not None and messages.outcome
            else None,
        }
    )
    episode = None
    if records:
        episode = MemoryEpisode(
            namespace=source.namespace,
            source_type=MemorySourceType.TASK,
            source_id=source.run_id,
            records=tuple(records),
            metadata={
                "lifecycle_source_id": source.lifecycle_source_id,
                "session_id": source.session_id,
                "start_message_count": messages.start_message_count,
                "end_message_count": end,
                "outcome": updated.outcome,
            },
        )
    return updated, episode
