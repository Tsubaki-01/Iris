"""两种 store 共用的历史来源规则与可信结果投影。

Example:
    validate_fork_source(run, is_child=False)
    point = project_fork_point(run)
"""

from datetime import datetime
from typing import cast

from ..exceptions import IrisRunStateError
from ..lifecycle.history import ForkPoint, ForkPointCursor, ForkPointPage
from ..lifecycle.models import RunPhase, RunRecord, RunStopReason, SessionContextWindow


def validate_context_window_initialization(
    current: SessionContextWindow | None, initial: SessionContextWindow | None
) -> None:
    """首输入必须初始化窗口，后续输入只能复用已初始化窗口。"""
    if current is None:
        if initial is None:
            raise IrisRunStateError("首次输入必须提供 context window")
    elif initial is not None:
        raise IrisRunStateError("已有 context window 不能由后续输入替换")


def validate_fork_source(run: RunRecord, *, is_child: bool) -> None:
    """在 store 操作边界检查分支来源，不重验已类型化的消息截点。"""
    if run.phase is not RunPhase.TERMINAL:
        raise IrisRunStateError("fork 来源必须是 terminal run", run_id=run.run_id)
    if is_child:
        raise IrisRunStateError("fork 来源必须是顶层 run", run_id=run.run_id)


def run_message_slice_bounds(
    run: RunRecord, *, session_message_count: int, after_count: int, limit: int
) -> tuple[int, int]:
    """在来源读取边界校验分页参数，并限定本 run 的半开消息区间。"""
    if limit <= 0:
        raise IrisRunStateError("limit 必须大于 0", limit=limit)
    end = (
        session_message_count
        if run.terminal_session_message_count is None
        else run.terminal_session_message_count
    )
    if not 0 <= after_count <= end:
        raise IrisRunStateError(
            "after_count 必须位于已提交消息范围内", run_id=run.run_id, after_count=after_count
        )
    start = max(after_count, run.initial_session_message_count)
    return start, min(start + limit, end)


def project_fork_point(run: RunRecord) -> ForkPoint:
    """将已通过来源筛选的 terminal run 投影为分支点。"""
    return ForkPoint(
        run_id=run.run_id,
        session_id=run.session_id,
        agent_id=run.agent_id,
        input=run.request.input,
        stop_reason=cast(RunStopReason, run.stop_reason),
        created_at=run.created_at,
        finished_at=cast(datetime, run.finished_at),
        message_count=cast(int, run.terminal_session_message_count),
    )


def build_fork_point_page(points: list[ForkPoint], *, limit: int) -> ForkPointPage:
    """将最多 limit+1 个已排序结果组成一页，游标指向最后返回项。"""
    items = tuple(points[:limit])
    cursor = None
    if len(points) > limit:
        last = items[-1]
        cursor = ForkPointCursor(created_at=last.created_at, run_id=last.run_id)
    return ForkPointPage(items=items, next_cursor=cursor)
