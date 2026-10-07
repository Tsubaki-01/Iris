"""两种后端共用的有限导航页投影，不负责查询与过滤。"""

from ..lifecycle.history import (
    ChildRunPage,
    ChildRunSummary,
    RunCursor,
    RunPage,
    SessionCursor,
    SessionPage,
    SessionSummary,
)
from ..lifecycle.models import RunSnapshot


def session_page(items: list[SessionSummary], limit: int) -> SessionPage:
    """将至多 limit+1 个候选投影为会话页。"""
    selected = tuple(items[:limit])
    cursor = (
        SessionCursor(selected[-1].latest_run_at, selected[-1].session_id)
        if len(items) > limit
        else None
    )
    return SessionPage(selected, cursor)


def run_page(items: list[RunSnapshot], limit: int) -> RunPage:
    """将至多 limit+1 个运行快照投影为运行页。"""
    selected = tuple(items[:limit])
    cursor = RunCursor(selected[-1].created_at, selected[-1].run_id) if len(items) > limit else None
    return RunPage(selected, cursor)


def child_run_page(items: list[ChildRunSummary], limit: int) -> ChildRunPage:
    """保留 child 身份与父工具关系，并生成下一页位置。"""
    selected = tuple(items[:limit])
    cursor = (
        RunCursor(selected[-1].run.created_at, selected[-1].run.run_id)
        if len(items) > limit
        else None
    )
    return ChildRunPage(selected, cursor)
