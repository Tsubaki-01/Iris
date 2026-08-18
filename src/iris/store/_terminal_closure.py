"""构造 terminal tool-call history closer。"""

from __future__ import annotations

from datetime import datetime

from ..exceptions import IrisRunStateError
from ..lifecycle.models import RunToolCallRecord, ToolCallPhase
from ..message.message import Msg
from ..tools.base import ToolErrorInfo, ToolResult

_OUTCOME_UNKNOWN_MESSAGE = (
    "工具调用已被 claim，但 run 终止时无法证明执行结果。不要盲目重试；"
    "仅当工具只读或幂等时重试，否则先核实外部状态或询问用户。"
)
_NOT_STARTED_MESSAGE = "工具调用在 run 终止前未开始，未产生工具副作用；如仍需要可以重试。"


def build_terminal_tool_closure(
    record: RunToolCallRecord,
    *,
    now: datetime,
) -> tuple[RunToolCallRecord, Msg]:
    """按 durable claim phase 构造 terminal tool fact 与模型可见 closer。"""
    if record.phase is ToolCallPhase.CLAIMED:
        updated = RunToolCallRecord.model_validate(
            record.model_dump()
            | {
                "phase": ToolCallPhase.OUTCOME_UNKNOWN,
                "version": record.version + 1,
                "updated_at": now,
            }
        )
        error = ToolErrorInfo(
            code="TOOL_OUTCOME_UNKNOWN",
            message=_OUTCOME_UNKNOWN_MESSAGE,
            retryable=False,
        )
    elif record.phase is ToolCallPhase.PREPARED:
        updated = record.model_copy(deep=True)
        error = ToolErrorInfo(
            code="TOOL_NOT_STARTED",
            message=_NOT_STARTED_MESSAGE,
            retryable=True,
        )
    else:
        raise IrisRunStateError(
            "terminal tool closure 只接受 prepared 或 claimed call",
            tool_call_id=record.tool_call_id,
            phase=record.phase.value,
        )

    result = ToolResult(
        tool_use_id=record.tool_call_id,
        tool_name=record.tool_name,
        is_error=True,
        error=error,
    )
    return updated, Msg.tool_result(
        tool_use_id=result.tool_use_id,
        content=result.model_content,
        is_error=True,
        name=result.tool_name,
        metadata=result.to_block_metadata(),
    )
