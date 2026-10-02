"""工具包装链唯一组合 owner：单次下游、失败回退与控制锁存。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..exceptions import (
    IrisCancellationRequestedError,
    IrisCommandCleanupError,
    IrisToolExecutionError,
    IrisToolOutcomeUnknownError,
)
from ._execution_control import ToolExecutionControlSlot
from .base import ToolResult
from .middleware import ToolCall, ToolMiddleware, ToolNext

logger = logging.getLogger(__name__)

CONTROL_ERRORS = (
    asyncio.CancelledError,
    IrisCancellationRequestedError,
    IrisToolOutcomeUnknownError,
    IrisCommandCleanupError,
)


@dataclass(slots=True)
class _Frame:
    """一层包装器拥有的 continuation 生命周期。"""

    active: bool = True
    started: bool = False
    operation: asyncio.Task[ToolResult] | None = None
    waiter: asyncio.Task[Any] | None = None
    contract_error: IrisToolExecutionError | None = None


async def drain_operation(
    task: asyncio.Task[Any], *, control: ToolExecutionControlSlot | None = None
) -> None:
    """等待操作收口；尚未转发的首次取消取消下游，重复取消不打断清理。"""
    cancellation_sent = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError as error:
            if control is not None:
                control.record(error)
                if not cancellation_sent and not task.done():
                    task.cancel()
                    cancellation_sent = True


async def run_middleware_chain(
    middleware: Sequence[ToolMiddleware],
    call: ToolCall,
    leaf: ToolNext,
    *,
    control: ToolExecutionControlSlot,
    remember_result: Callable[[ToolResult], None],
    middleware_error: Callable[[Exception], ToolResult],
) -> ToolResult:
    """组合 onion 链，并收回包装器已经启动而未等待的下游。"""

    async def invoke(index: int) -> ToolResult:
        """执行一层，控制异常始终优先于包装器给出的恢复结果。"""
        if index == len(middleware):
            return await leaf()
        frame = _Frame()

        async def call_next() -> ToolResult:
            """只允许有效 frame 启动一次下游。"""
            if not frame.active or frame.started:
                error = IrisToolExecutionError("call_next 只能在当前包装器内调用一次")
                if frame.active:
                    frame.contract_error = error
                raise error
            frame.started = True
            frame.waiter = asyncio.current_task()
            frame.operation = asyncio.create_task(invoke(index + 1))
            try:
                return await asyncio.shield(frame.operation)
            except asyncio.CancelledError as error:
                control.record(error)
                if not frame.operation.done():
                    frame.operation.cancel()
                await drain_operation(frame.operation)
                raise
            except CONTROL_ERRORS as error:
                control.record(error)
                raise

        result: ToolResult | None = None
        failure: BaseException | None = None
        try:
            result = await middleware[index].wrap_tool_call(call, call_next)
        except CONTROL_ERRORS as error:
            control.record(error)
            failure = error
        except Exception as error:
            failure = error
        finally:
            frame.active = False

        early = frame.operation is not None and not frame.operation.done()
        if early:
            if control.control is not None:
                frame.operation.cancel()
                await drain_operation(frame.operation)
            else:
                await drain_operation(frame.operation, control=control)
        # 单独 create_task(call_next()) 的 waiter 也必须结束，并取回其异常。
        if frame.waiter is not None and frame.waiter is not asyncio.current_task():
            await drain_operation(frame.waiter, control=control)
            if not frame.waiter.cancelled():
                frame.waiter.exception()

        downstream: ToolResult | None = None
        downstream_error: BaseException | None = None
        if frame.operation is not None:
            try:
                downstream = frame.operation.result()
            except CONTROL_ERRORS as error:
                control.record(error)
                downstream_error = error
            except Exception as error:
                downstream_error = error

        if control.control is not None:
            raise control.control.error
        failure = frame.contract_error or failure
        if early or failure is not None:
            if downstream is not None:
                logger.warning("工具 Middleware 后置失败，保留下游结果：%s", failure or "提前返回")
                return downstream
            if downstream_error is not None:
                raise downstream_error
            # 没有下游的普通失败属于 Middleware，而非工具 body。
            assert isinstance(failure, Exception)
            return middleware_error(failure)
        assert result is not None
        remember_result(result)
        return result

    return await invoke(0)
