"""SessionManager 在原状态变化点产生的不可变控制投影。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..hitl import InteractionStatus
from ..lifecycle import RunSnapshot
from ..message import DataBlock


@dataclass(frozen=True, slots=True)
class PendingSubmission:
    """尚未完成投递的进程内输入；块序列使用 tuple 保持只读。"""

    submission_id: str
    run_id: str
    mode: Literal["steer", "follow_up"] | None
    input: str | tuple[DataBlock, ...]
    stage: Literal["queued", "committing", "admitting"]
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class SessionControlSnapshot:
    """只读控制提示，revision 不参与 durable CAS。"""

    manager_id: str
    revision: int
    session_id: str
    current_run_id: str | None = None
    run: RunSnapshot | None = None
    driver_state: Literal["idle", "admitting", "running", "settling", "detached", "closed"] = "idle"
    pending: tuple[PendingSubmission, ...] = ()
    allowed_commands: tuple[str, ...] = ()
    pending_scope: Literal["process_local"] = "process_local"
    interaction_status: InteractionStatus | None = None


@dataclass(frozen=True, slots=True)
class RestoreReceipt:
    """显式接管达到的状态；不表示整个 Run 已完成。"""

    run_id: str
    disposition: Literal["already_managed", "attached_waiting", "recovery_started", "settled"]
    control: SessionControlSnapshot


__all__ = ["PendingSubmission", "RestoreReceipt", "SessionControlSnapshot"]
