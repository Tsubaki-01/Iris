"""派发器与执行 owner、命令适配器之间的非持久化交接类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from .models import HookEvent, HookEventName, HookResult

if TYPE_CHECKING:
    from ..command.models import CommandStopSlot
    from ..exceptions import IrisToolOutcomeUnknownError
    from ..tools.base import CancellationSignal

type HookControlOrigin = Literal[
    "task_cancelled", "run_cancelled", "unknown", "cleanup", "environment_interrupted"
]


@dataclass(frozen=True, slots=True)
class HookRejection:
    """仅当前工具调用的拒绝或普通处理器失败。"""

    code: Literal["HOOK_REJECTED", "HOOK_ERROR"]
    reason: str


@dataclass(frozen=True, slots=True)
class HookControl:
    """保留原控制与命令事实槽；不把停止事实放入公开结果或 JSON。"""

    origin: HookControlOrigin
    error: BaseException
    stop_slot: CommandStopSlot | None = None
    call_id: str | None = None
    unknown_error: IrisToolOutcomeUnknownError | None = None


@dataclass(frozen=True, slots=True)
class HookInvocationOutcome:
    """单个命令 adapter 已归一化的结果或尚待 owner 消费的控制。"""

    result: HookResult = None
    control: HookControl | None = None


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    """一次事件派发的拒绝、已收集反馈与停止原因。"""

    rejection: HookRejection | None = None
    feedback: tuple[str, ...] = ()
    control: HookControl | None = None


class CommandHookHandler(Protocol):
    """私有命令调用协议；后端拥有期限，回调直接返回类型化控制事实。"""

    async def __call__(
        self, event: HookEvent, *, cancellation: CancellationSignal | None
    ) -> HookInvocationOutcome:
        """使用自身 binding 与期限执行并交接尚未收口的命令事实。"""
        ...


@dataclass(frozen=True, slots=True)
class CommandHookRegistration:
    """由 assembly 构造的命令注册；通过独立类型区分执行方式。"""

    event: HookEventName
    name: str
    handler: CommandHookHandler
    tool_names: tuple[str, ...] | None = None
