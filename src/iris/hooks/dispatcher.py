"""有序 Hooks 派发与单次处理器取消收口；不拥有 durable 状态。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from ..command.models import CommandStopSlot
from ..exceptions import (
    IrisCancellationRequestedError,
    IrisCommandCleanupError,
    IrisError,
    IrisHookError,
    IrisHookProtocolError,
    IrisToolOutcomeUnknownError,
)
from ._dispatch_types import (
    CommandHookRegistration,
    DispatchOutcome,
    HookControl,
    HookInvocationOutcome,
    HookRejection,
)
from .models import (
    HookEvent,
    HookRegistration,
    HookResult,
    RunFinishedEvent,
    ToolAfterEvent,
    ToolAfterResult,
    ToolBeforeEvent,
    ToolBeforeResult,
)

if TYPE_CHECKING:
    from ..tools.base import CancellationSignal

logger = logging.getLogger(__name__)

type _Registration = HookRegistration | CommandHookRegistration
_CONTROL_ERRORS = (
    asyncio.CancelledError,
    IrisCancellationRequestedError,
    IrisToolOutcomeUnknownError,
    IrisCommandCleanupError,
)


class HookDispatcher:
    """按注册顺序派发一个事件；每次派发拥有独立反馈和控制状态。"""

    def __init__(self, registrations: Sequence[_Registration] = ()) -> None:
        """保留已校验注册项的固定顺序，不再次解析 SDK 输入。"""
        self.registrations = tuple(registrations)

    async def dispatch(
        self, event: HookEvent, *, cancellation: CancellationSignal | None = None
    ) -> DispatchOutcome:
        """派发匹配项；普通失败按事件处理，控制连同已有反馈交还 owner。"""
        feedback: list[str] = []
        for registration in self.registrations:
            if not _matches(registration, event):
                continue
            if (
                isinstance(registration, CommandHookRegistration)
                and isinstance(event, RunFinishedEvent)
                and event.result.run.stop_reason != "completed"
            ):
                logger.info("跳过非 completed Run 的命令 Hook：%s", registration.name)
                continue
            try:
                # 每个处理器获得独立原输入，不能修改 engine 或后续处理器的视图。
                outcome = await _invoke(registration, deepcopy(event), cancellation)
            except Exception as error:
                code = error.runtime_code if isinstance(error, IrisError) else "HOOK_ERROR"
                reason = f"Hook {registration.name} 执行失败：{error}"
                logger.warning("%s [%s] event=%s", reason, code, event.event)
                if isinstance(event, ToolBeforeEvent):
                    return DispatchOutcome(rejection=HookRejection("HOOK_ERROR", reason))
                continue
            if outcome.control is not None:
                return DispatchOutcome(feedback=tuple(feedback), control=outcome.control)
            if outcome.result is None:
                continue
            if isinstance(event, ToolBeforeEvent):
                result = cast(ToolBeforeResult, outcome.result)
                return DispatchOutcome(rejection=HookRejection("HOOK_REJECTED", result.deny_reason))
            if isinstance(event, ToolAfterEvent):
                feedback.append(cast(ToolAfterResult, outcome.result).feedback)
        return DispatchOutcome(feedback=tuple(feedback))


def _matches(registration: _Registration, event: HookEvent) -> bool:
    """只匹配固定事件与实际工具调用名。"""
    return registration.event == event.event and (
        registration.tool_names is None
        or isinstance(event, (ToolBeforeEvent, ToolAfterEvent))
        and event.tool_name in registration.tool_names
    )


def _python_result(event: HookEvent, result: object) -> HookResult:
    """在 Python 回调输出首次进入框架时验证当前事件的返回类型。"""
    if result is None:
        return None
    if isinstance(event, ToolBeforeEvent) and isinstance(result, ToolBeforeResult):
        return result
    if isinstance(event, ToolAfterEvent) and isinstance(result, ToolAfterResult):
        return result
    raise IrisHookProtocolError(f"{event.event} 不允许返回 {type(result).__name__}")


async def _invoke(
    registration: _Registration,
    event: HookEvent,
    cancellation: CancellationSignal | None,
) -> HookInvocationOutcome:
    """唯一 invocation owner，区分 Python 自身期限与外部取消并等待必要清理。"""
    budget: asyncio.Timeout | None = None

    async def execute() -> HookInvocationOutcome:
        """Python 期限放在处理器 task 内；命令 adapter 使用后端自己的期限。"""
        nonlocal budget
        if isinstance(registration, CommandHookRegistration):
            command = cast(CommandHookRegistration, registration)
            return await command.handler(event, cancellation=cancellation)
        budget = asyncio.timeout(registration.timeout_seconds)
        try:
            async with budget:
                result = await registration.handler(event)
        except TimeoutError as error:
            if budget.expired():
                raise IrisHookError(f"Hook {registration.name} 超时") from error
            raise
        if budget.expired():
            raise IrisHookError(f"Hook {registration.name} 超时，丢弃迟到结果")
        return HookInvocationOutcome(result=_python_result(event, result))

    task = asyncio.create_task(execute())
    control: HookControl | None = None
    cancellation_sent = False

    def interrupt() -> None:
        """外部中断禁用尚未到期的预算，避免预算在清理期间再次取消处理器。"""
        nonlocal cancellation_sent
        expired = budget is not None and budget.expired()
        if budget is not None and not expired:
            budget.reschedule(None)
        if not task.done() and not cancellation_sent and not expired:
            task.cancel()
        cancellation_sent = True

    try:
        while not task.done():
            if cancellation is not None and cancellation.requested and control is None:
                control = _control_from_error(IrisCancellationRequestedError("Hook 响应 Run 取消"))
                interrupt()
            await asyncio.wait(
                {task}, timeout=0.01 if cancellation is not None and not cancellation_sent else None
            )
    except asyncio.CancelledError as error:
        control = control or _control_from_error(error)
        if not task.done() and not cancellation_sent:
            interrupt()
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                continue
    if cancellation is not None and cancellation.requested and control is None:
        control = _control_from_error(IrisCancellationRequestedError("Hook 响应 Run 取消"))
    try:
        outcome = task.result()
    except _CONTROL_ERRORS as error:
        incoming = _control_from_error(error)
        return HookInvocationOutcome(control=_merge_control(control, incoming))
    except Exception:
        if control is not None:
            return HookInvocationOutcome(control=control)
        raise
    if control is not None:
        return HookInvocationOutcome(control=_merge_control(control, outcome.control))
    return outcome


def _control_from_error(error: BaseException) -> HookControl:
    """将已知控制异常和其中的命令停止事实投影为内部类型。"""
    if isinstance(error, IrisToolOutcomeUnknownError):
        slot = CommandStopSlot(receipt=error.stop_receipt)
        return HookControl("unknown", error, stop_slot=slot, unknown_error=error)
    if isinstance(error, IrisCommandCleanupError):
        slot = CommandStopSlot(cleanup_error=error)
        if error.command_outcome is not None:
            slot.record(
                status=error.command_outcome.status, receipt=error.command_outcome.stop_receipt
            )
        return HookControl("cleanup", error, stop_slot=slot)
    if isinstance(error, IrisCancellationRequestedError):
        return HookControl("run_cancelled", error)
    return HookControl("task_cancelled", error)


def _merge_control(first: HookControl | None, incoming: HookControl | None) -> HookControl | None:
    """保持先观察到的取消来源，同时交接收口期间产生的 unknown/cleanup。"""
    if first is None:
        return incoming
    if incoming is None:
        return first
    return replace(
        first,
        stop_slot=first.stop_slot or incoming.stop_slot,
        call_id=first.call_id or incoming.call_id,
        unknown_error=first.unknown_error or incoming.unknown_error,
    )
