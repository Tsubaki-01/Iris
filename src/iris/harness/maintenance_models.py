"""协调器现有调度事实的不可变读取投影。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

type MaintenanceState = Literal[
    "idle", "waiting_for_idle", "waiting_for_foreground", "waiting_for_lock", "running", "closing"
]


@dataclass(frozen=True, slots=True)
class ResourceMaintenanceView:
    """一个 Memory 或 Evolution 资源的当前调度事实。"""

    resource_ref: str
    state: MaintenanceState
    pending_request_id: str | None = None
    cycle_id: str | None = None
    next_eligible_at: datetime | None = None
    last_result_ref: str | None = None


@dataclass(frozen=True, slots=True)
class MaintenanceSnapshot:
    """同一 coordinator 的完整资源投影；不负责持久状态或恢复。"""

    coordinator_id: str
    revision: int
    foreground_count: int
    resources: tuple[ResourceMaintenanceView, ...] = ()


@dataclass(frozen=True, slots=True)
class MaintenanceChanged:
    """资源作用域通知，只携带这一个资源的控制视图。"""

    coordinator_id: str
    revision: int
    foreground_count: int
    resource: ResourceMaintenanceView
