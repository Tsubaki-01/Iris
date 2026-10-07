"""会话历史分支的进程内查询结果。

Example:
    page = store.list_fork_points("main", limit=20)
    preview = store.load_session_at_run(page.items[0].run_id)
"""

from dataclasses import dataclass
from datetime import datetime

from ..message.message import Msg, Role
from .models import (
    RunSnapshot,
    RunStopReason,
    SessionCompaction,
    SessionContextWindow,
    SessionToolDiscovery,
)


def is_ordinary_user(message: Msg) -> bool:
    """区分普通输入与 context、工具结果占用的 user role。"""
    return message.role is Role.USER and message.sender != "context" and not message.tool_results


@dataclass(frozen=True, slots=True)
class SessionHeader:
    """输入准备所需的窄会话信息，不携带历史与摘要正文。"""

    session_id: str
    revision: int
    message_count: int
    context_window: SessionContextWindow | None


@dataclass(frozen=True, slots=True)
class SessionContextSnapshot:
    """同一版本的模型有效历史及保留绝对位置的当前任务锚点。"""

    header: SessionHeader
    compaction: SessionCompaction | None
    raw_tail: tuple[Msg, ...]
    protected_indices: tuple[int, ...]
    protected_prefix_messages: tuple[tuple[int, Msg], ...]
    tool_discovery: SessionToolDiscovery | None


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


@dataclass(frozen=True, slots=True)
class SessionMessagePage:
    """一次读取快照中的有限原始消息页，位置从零开始。"""

    items: tuple[tuple[int, Msg], ...]
    next_index: int | None
    total_count: int


@dataclass(frozen=True, slots=True)
class SessionCursor:
    """按最新 Run 时间和 session ID 降序分页的位置。"""

    latest_run_at: datetime | None
    session_id: str


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """不包含正文的根会话导航信息。"""

    session_id: str
    revision: int
    message_count: int
    current_run_id: str | None
    latest_run_id: str | None
    latest_run_at: datetime | None
    forked_from_run_id: str | None


@dataclass(frozen=True, slots=True)
class SessionPage:
    """一次读取快照中的根会话页，未运行的持久 fork 排在末尾。"""

    items: tuple[SessionSummary, ...]
    next_cursor: SessionCursor | None


@dataclass(frozen=True, slots=True)
class RunCursor:
    """按创建时间和 Run ID 升序分页的位置。"""

    created_at: datetime
    run_id: str


@dataclass(frozen=True, slots=True)
class RunPage:
    """包含所有 phase 的运行页，不复用 fork-only 过滤。"""

    items: tuple[RunSnapshot, ...]
    next_cursor: RunCursor | None


@dataclass(frozen=True, slots=True)
class ChildRunSummary:
    """持久父工具关系、真实 selector 与 child 自身的运行快照。"""

    run: RunSnapshot
    parent_run_id: str
    parent_tool_call_id: str
    agent_selector: str


@dataclass(frozen=True, slots=True)
class ChildRunPage:
    """父 Run 的直接 child 页；后代按各自父 Run 继续查询。"""

    items: tuple[ChildRunSummary, ...]
    next_cursor: RunCursor | None


__all__ = [
    "SessionCursor",
    "SessionSummary",
    "SessionPage",
    "RunCursor",
    "RunPage",
    "ChildRunSummary",
    "ChildRunPage",
    "SessionHeader",
    "SessionContextSnapshot",
    "is_ordinary_user",
    "ForkPoint",
    "ForkPointCursor",
    "ForkPointPage",
    "RunHistorySnapshot",
    "RunMessageSlice",
    "SessionMessagePage",
]
