"""Goal 的会话 SDK 与宿主控制协议，不持有 runner 或异步调度对象。"""

from __future__ import annotations

from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from ..exceptions import IrisGoalStateError
from ..lifecycle.models import AgentRunOptions
from .models import (
    GoalControlResult,
    GoalCreateInput,
    GoalEditInput,
    GoalReason,
    GoalText,
    GoalView,
)
from .service import GoalService

_OPTIONAL_ID = TypeAdapter(GoalText | None)


def _user_reason(reason: str) -> GoalReason:
    """在 SDK 原始输入边界解析用户原因一次。"""
    try:
        return GoalReason(code="user", text=reason)
    except ValidationError as exc:
        raise IrisGoalStateError("目标控制原因不能为空", error=str(exc)) from exc


class GoalControlPort(Protocol):
    """绑定单个 session 的宿主入口；所有操作经同一个 admission owner。"""

    async def create(self, input_data: GoalCreateInput) -> GoalControlResult:
        """保存并允许调度已解析的目标。"""
        ...

    async def get(self) -> GoalView:
        """只读当前目标和会话执行状态。"""
        ...

    async def edit(self, input_data: GoalEditInput) -> GoalControlResult:
        """应用已解析编辑并暂停后续推进。"""
        ...

    async def pause(self, reason: GoalReason) -> GoalControlResult:
        """暂停后续推进，保留当前 Run。"""
        ...

    async def complete(self, reason: GoalReason) -> GoalControlResult:
        """宿主完成目标，保留当前 Run。"""
        ...

    async def resume(self, *, expected_activation_id: str | None = None) -> GoalControlResult:
        """先处理原执行的附着或恢复，再决定是否推进。"""
        ...

    async def clear(self) -> GoalControlResult:
        """撤销当前目标选择并停止后续推进。"""
        ...


class GoalSession:
    """会话 Goal SDK：解析原始字段后将类型化操作交给宿主。"""

    def __init__(self, service: GoalService, port: GoalControlPort) -> None:
        """共享领域解析和只接收 typed input 的宿主接口。"""
        self._service = service
        self._port = port

    async def create(
        self,
        objective: str,
        *,
        max_rounds: int | None = None,
        run_options: AgentRunOptions | None = None,
    ) -> GoalControlResult:
        """保存并显式允许目标推进，不等待模型完成。"""
        return await self._port.create(
            self._service.parse_create(
                objective,
                max_rounds=max_rounds,
                run_options=run_options,
            )
        )

    async def get(self) -> GoalView:
        """只读投影，不启动、恢复或补结算。"""
        return await self._port.get()

    async def edit(
        self,
        *,
        objective: str | None = None,
        max_rounds: int | None = None,
        run_options: AgentRunOptions | None = None,
    ) -> GoalControlResult:
        """编辑至少一个目标字段，暂停后续推进。"""
        return await self._port.edit(
            self._service.parse_edit(
                objective=objective,
                max_rounds=max_rounds,
                run_options=run_options,
            )
        )

    async def pause(self, *, reason: str) -> GoalControlResult:
        """暂停目标；立即取消当前 Run 使用宿主 interrupt。"""
        return await self._port.pause(_user_reason(reason))

    async def complete(self, *, reason: str) -> GoalControlResult:
        """按用户原因完成目标，不取消已准入 Run。"""
        return await self._port.complete(_user_reason(reason))

    async def resume(self, *, expected_activation_id: str | None = None) -> GoalControlResult:
        """恢复当前目标，存量 ACTIVE 的接管须明确 activation fence。"""
        try:
            expected_activation_id = _OPTIONAL_ID.validate_python(expected_activation_id)
        except ValidationError as exc:
            raise IrisGoalStateError("恢复 activation ID 不能为空", error=str(exc)) from exc
        return await self._port.resume(expected_activation_id=expected_activation_id)

    async def clear(self) -> GoalControlResult:
        """取消当前选择，保留目标、绑定和历史记录。"""
        return await self._port.clear()


__all__ = ["GoalSession", "GoalControlPort"]
