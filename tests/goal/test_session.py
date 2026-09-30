"""GoalSession 原始输入只解析一次，并经类型化 port 控制执行。"""

import pytest

from iris.exceptions import IrisGoalStateError
from iris.goal.models import (
    GoalControlResult,
    GoalCreateInput,
    GoalEditInput,
    GoalReason,
    GoalView,
)
from iris.goal.service import GoalService
from iris.goal.session import GoalSession
from iris.lifecycle import AgentRunOptions
from iris.store import InMemoryLifecycleStore


class _Port:
    """模拟持锁后的宿主控制端，仅消费已解析输入。"""

    def __init__(self, service: GoalService) -> None:
        self.service = service
        self.calls: list[tuple[str, object]] = []

    async def create(self, input_data: GoalCreateInput) -> GoalControlResult:
        self.calls.append(("create", input_data))
        self.service.create_validated("s", input_data)
        return GoalControlResult(view=await self.get(), disposition="scheduled")

    async def edit(self, input_data: GoalEditInput) -> GoalControlResult:
        self.calls.append(("edit", input_data))
        self.service.edit_validated(self.service.get_current("s").ref, input_data)
        return GoalControlResult(view=await self.get(), disposition="stopped")

    async def pause(self, reason: GoalReason) -> GoalControlResult:
        self.calls.append(("pause", reason))
        self.service.pause(self.service.get_current("s").ref, reason=reason)
        return GoalControlResult(view=await self.get(), disposition="stopped")

    async def complete(self, reason: GoalReason) -> GoalControlResult:
        self.calls.append(("complete", reason))
        self.service.complete(self.service.get_current("s").ref, reason=reason)
        return GoalControlResult(view=await self.get(), disposition="stopped")

    async def resume(self, *, expected_activation_id: str | None = None) -> GoalControlResult:
        self.calls.append(("resume", expected_activation_id))
        self.service.resume(self.service.get_current("s").ref)
        return GoalControlResult(view=await self.get(), disposition="scheduled")

    async def clear(self) -> GoalControlResult:
        self.calls.append(("clear", None))
        current = self.service.get_current("s")
        self.service.clear("s", expected=None if current is None else current.ref)
        return GoalControlResult(view=await self.get(), disposition="stopped")

    async def get(self) -> GoalView:
        return self.service.get_view("s")


@pytest.mark.asyncio
async def test_create_edit_parse_once_before_port_and_preserve_typed_options() -> None:
    checked: list[AgentRunOptions] = []
    service = GoalService(InMemoryLifecycleStore(), run_options_validator=checked.append)
    port = _Port(service)
    session = GoalSession(service, port)
    options = AgentRunOptions()
    created = await session.create("完整目标", max_rounds=4, run_options=options)
    assert created.disposition == "scheduled"
    assert created.view.goal.objective == "完整目标"
    assert created.view.goal.max_rounds == 4
    assert checked == [options]
    assert port.calls[0][1].run_options is options
    edited = await session.edit(objective="修改后目标", run_options=options)
    assert edited.view.goal.status == "paused"
    assert checked == [options, options]
    assert port.calls[1][1].run_options is options


@pytest.mark.asyncio
async def test_invalid_public_fields_fail_before_port_mutation() -> None:
    service = GoalService(InMemoryLifecycleStore())
    port = _Port(service)
    session = GoalSession(service, port)
    for kwargs in ({"objective": " "}, {"objective": "目标", "max_rounds": 0}):
        with pytest.raises(IrisGoalStateError):
            await session.create(**kwargs)
    with pytest.raises(IrisGoalStateError):
        await session.edit()
    with pytest.raises(IrisGoalStateError):
        await session.pause(reason=" ")
    with pytest.raises(IrisGoalStateError):
        await session.complete(reason="")
    with pytest.raises(IrisGoalStateError):
        await session.resume(expected_activation_id=" ")
    assert port.calls == []
    assert await session.get() == service.get_view("s")


@pytest.mark.asyncio
async def test_controls_forward_reason_and_recovery_fence_without_starting_work() -> None:
    service = GoalService(InMemoryLifecycleStore())
    port = _Port(service)
    session = GoalSession(service, port)
    await session.create("目标")
    paused = await session.pause(reason="等我确认")
    assert paused.view.goal.reason == GoalReason(code="user", text="等我确认")
    resumed = await session.resume(expected_activation_id="exact-activation")
    assert resumed.disposition == "scheduled"
    assert port.calls[-1] == ("resume", "exact-activation")
    completed = await session.complete(reason="已经验收")
    assert completed.view.goal.status == "completed"
    assert (await session.clear()).view.goal is None
    assert service.store.load_session_lane("s") is None
