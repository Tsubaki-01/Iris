"""普通工具 body 的取消 ownership 与结果边界。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from iris.exceptions import IrisCancellationRequestedError
from iris.message import ToolUseBlock
from iris.tools import (
    BaseTool,
    CallableExecutionMode,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)


class Cancellation:
    """由测试显式改变的共享取消信号。"""

    requested = False

    def raise_if_requested(self) -> None:
        """保持与实际 activation 相同的控制异常。"""
        if self.requested:
            raise IrisCancellationRequestedError("测试取消")


def _start(
    tmp_path: Path,
    signal: Cancellation,
    body: Callable[[], str | Awaitable[str]],
    *,
    middleware: Sequence[ToolMiddleware] = (),
) -> asyncio.Task[ToolResult]:
    """通过普通 executor 入口运行受控工具。"""
    registry = ToolRegistry()
    registry.register_function(body, name="body", description="受控工具")
    executor = ToolExecutor(registry, middleware=middleware)
    return asyncio.create_task(
        executor.execute_one(
            ToolUseBlock(id="call", name="body", input={}),
            ToolExecutionContext(workspace_root=tmp_path, cancellation=signal),
        )
    )


@pytest.mark.asyncio
async def test_completed_body_result_wins_over_signal(tmp_path: Path) -> None:
    """观察到结果与取消同时存在时，保留已经完成的结果。"""
    signal = Cancellation()

    async def body() -> str:
        """返回结果前同步设置取消，不留时间竞争。"""
        signal.requested = True
        return "known"

    result = await _start(tmp_path, signal, body)
    assert result.model_content == "known"


@pytest.mark.asyncio
async def test_cancellation_during_before_call_skips_body(tmp_path: Path) -> None:
    """before_call 已让出控制权，因此 body 启动前需读取最新取消状态。"""
    signal = Cancellation()
    entered = False

    class CancelBefore(ToolMiddleware):
        """在 before_call 挂起后通知取消。"""

        async def before_call(
            self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
        ) -> None:
            """模拟 hook 执行期间到达的取消。"""
            await asyncio.sleep(0)
            signal.requested = True

    def body() -> str:
        """记录是否错误地启动了 body。"""
        nonlocal entered
        entered = True
        return "unexpected"

    with pytest.raises(IrisCancellationRequestedError):
        await _start(tmp_path, signal, body, middleware=[CancelBefore()])
    assert not entered


@pytest.mark.asyncio
async def test_cancellation_during_after_call_preserves_result(tmp_path: Path) -> None:
    """body 已完成时继续已有 after_call 处理，不丢弃结果。"""
    signal = Cancellation()

    class CancelAfter(ToolMiddleware):
        """在处理确定结果时通知取消。"""

        async def after_call(
            self, tool: BaseTool, result: ToolResult, context: ToolExecutionContext
        ) -> ToolResult:
            """取消后依旧交付 hook 的处理结果。"""
            await asyncio.sleep(0)
            signal.requested = True
            return result

    def body() -> str:
        """提供确定结果。"""
        return "known"

    result = await _start(tmp_path, signal, body, middleware=[CancelAfter()])
    assert result.model_content == "known"


@pytest.mark.asyncio
@pytest.mark.parametrize("first_cancel", ["signal", "task"])
async def test_outer_cancellation_drains_body_without_recancelling_cleanup(
    tmp_path: Path, first_cancel: str
) -> None:
    """signal、外层及重复取消共享一次 body 清理，外层取消仍传播。"""
    signal = Cancellation()
    entered = asyncio.Event()
    release_body = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_interrupted = False

    async def body() -> str:
        """暴露取消清理的等待点。"""
        nonlocal cleanup_interrupted
        entered.set()
        try:
            await release_body.wait()
            return "released"
        finally:
            cleanup_entered.set()
            try:
                await release_cleanup.wait()
            except asyncio.CancelledError:
                cleanup_interrupted = True
                raise
            cleanup_finished.set()

    execution = _start(tmp_path, signal, body)
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        if first_cancel == "signal":
            signal.requested = True
        else:
            execution.cancel()
        await asyncio.wait_for(cleanup_entered.wait(), timeout=0.25)
        execution.cancel()
        await asyncio.sleep(0)
        execution.cancel()
        await asyncio.sleep(0)
        assert not execution.done()
        assert not cleanup_interrupted
    finally:
        release_body.set()
        release_cleanup.set()
        [outcome] = await asyncio.gather(execution, return_exceptions=True)

    assert isinstance(outcome, asyncio.CancelledError)
    assert cleanup_finished.is_set()
    assert not cleanup_interrupted


@pytest.mark.asyncio
async def test_body_self_cancellation_is_not_an_iris_request(tmp_path: Path) -> None:
    """没有 signal 请求的 body 自取消不归一化为普通错误或领域取消。"""

    async def body() -> str:
        """模拟工具自身的程序中断。"""
        raise asyncio.CancelledError("body stopped")

    with pytest.raises(asyncio.CancelledError, match="body stopped"):
        await _start(tmp_path, Cancellation(), body)


@pytest.mark.asyncio
async def test_direct_thread_arun_uses_caller_task_cancellation(tmp_path: Path) -> None:
    """低层 arun 只负责 THREAD placement，调用者拥有 waiter 取消。"""
    signal = Cancellation()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def body() -> str:
        """worker 在 waiter 退出后仍可继续，最终由测试放行。"""
        entered.set()
        try:
            release.wait(timeout=2)
            return "late"
        finally:
            finished.set()

    registry = ToolRegistry()
    tool = registry.register_function(body, execution_mode=CallableExecutionMode.THREAD)
    execution = asyncio.create_task(
        tool.arun(
            tool.validate_input({}),
            ToolExecutionContext(workspace_root=tmp_path, cancellation=signal),
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        signal.requested = True
        done, _ = await asyncio.wait({execution}, timeout=0.05)
        assert not done
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.gather(execution, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 1)
