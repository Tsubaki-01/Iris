"""会话历史分支的进程内查询结果。

Example:
    page = store.list_fork_points("main", limit=20)
    preview = store.load_session_at_run(page.items[0].run_id)
"""

from dataclasses import dataclass
from datetime import datetime

from ..message.message import Msg
from .models import RunStopReason


@dataclass(frozen=True, slots=True)
class ForkPointCursor:
    """按创建时间与 run ID 翻页的不可变位置。"""

    created_at: datetime
    run_id: str


@dataclass(frozen=True, slots=True)
class ForkPoint:
    """一个 terminal 顶层 run 的请求信息与历史消息截点。"""

    run_id: str
    session_id: str
    agent_id: str
    input: str
    stop_reason: RunStopReason
    created_at: datetime
    finished_at: datetime
    message_count: int


@dataclass(frozen=True, slots=True)
class ForkPointPage:
    """已过滤分支点的一页结果，以及下一页位置。"""

    items: tuple[ForkPoint, ...]
    next_cursor: ForkPointCursor | None


@dataclass(frozen=True, slots=True)
class RunHistorySnapshot:
    """指定 run 末尾的历史前缀，不提供当前 session 的 CAS revision。"""

    point: ForkPoint
    messages: tuple[Msg, ...]


@dataclass(frozen=True, slots=True)
class RunMessageSlice:
    """同一读取快照内的 run 新增消息区间，计数均为 session 累计消息数。"""

    source_id: str
    run_id: str
    session_id: str
    initial_message_count: int
    start_message_count: int
    end_message_count: int
    terminal_message_count: int | None
    outcome: RunStopReason | None
    messages: tuple[Msg, ...]


__all__ = ["ForkPoint", "ForkPointCursor", "ForkPointPage", "RunHistorySnapshot", "RunMessageSlice"]
