"""两种 store 共用的历史来源规则与可信结果投影。

Example:
    validate_fork_source(run, is_child=False)
    point = project_fork_point(run)
"""

from datetime import datetime
from typing import cast

from ..exceptions import IrisRunStateError
from ..lifecycle.history import ForkPoint, ForkPointCursor, ForkPointPage
from ..lifecycle.models import RunPhase, RunRecord, RunStopReason


def validate_fork_source(run: RunRecord, *, is_child: bool) -> None:
    """在 store 操作边界检查分支来源，不重验已类型化的消息截点。"""
    if run.phase is not RunPhase.TERMINAL:
        raise IrisRunStateError("fork 来源必须是 terminal run", run_id=run.run_id)
    if is_child:
        raise IrisRunStateError("fork 来源必须是顶层 run", run_id=run.run_id)


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
