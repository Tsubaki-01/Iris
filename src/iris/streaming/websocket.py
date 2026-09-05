"""Framework-neutral WebSocket command parser 与 single-writer state machine。

Host 负责 socket upgrade 与 authentication；本模块只消费两个 callback。

Example:
    await WebSocketAdapter(gateway=gateway).serve(receive, send)
"""

# region imports

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError

from ..exceptions import IrisError
from .gateway import GatewaySubscription, StreamingGateway
from .models import (
    CancelCommand,
    CommandReceipt,
    CommandRejected,
    GatewayStreamItem,
    ResumeCommand,
    SubmitCommand,
    SubscribeAccepted,
    SubscribeCommand,
    SyncCommand,
)

# endregion

logger = logging.getLogger(__name__)

type _IncomingCommand = Annotated[
    SubscribeCommand | SubmitCommand | ResumeCommand | CancelCommand | SyncCommand,
    Field(discriminator="kind"),
]
type _ConnectionPhase = Literal["initial", "awaiting_subscription", "subscribed"]

_COMMAND_ADAPTER = TypeAdapter(_IncomingCommand)


@dataclass(slots=True)
class _ConnectionState:
    """单次 serve 调用的 task-shared 状态。"""

    phase: _ConnectionPhase = "initial"
    subscription: GatewaySubscription | None = None
    subscription_ready: asyncio.Event = field(default_factory=asyncio.Event)
    receiver_done: asyncio.Event = field(default_factory=asyncio.Event)
    receipt_capacity: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))


def _parse_frame(frame: str | bytes) -> _IncomingCommand:
    """在 raw socket boundary 解码 UTF-8 JSON 并校验 typed command。"""
    value = frame.decode("utf-8") if isinstance(frame, bytes) else frame
    return _COMMAND_ADAPTER.validate_json(value)


def _invalid_command() -> CommandRejected:
    """返回不含 raw frame 或 validation details 的固定拒绝。"""
    return CommandRejected(
        code="INVALID_COMMAND",
        message="命令 JSON 或 schema 无效",
    )


class WebSocketAdapter:
    """为一个 exact-session gateway 提供单 writer socket orchestration。"""

    def __init__(self, *, gateway: StreamingGateway) -> None:
        """绑定 host 已完成授权的 gateway。"""
        self._gateway = gateway

    async def serve(
        self,
        receive: Callable[[], Awaitable[str | bytes | None]],
        send: Callable[[str], Awaitable[None]],
    ) -> None:
        """运行一个连接，disconnect 时只关闭 observation。

        Args:
            receive (Callable): 返回下一条 text/bytes frame；None 表示断线。
            send (Callable): 唯一 sender task 使用的 text-frame callback。
        """
        receipts: asyncio.Queue[CommandReceipt] = asyncio.Queue(maxsize=1)
        state = _ConnectionState()
        receiver = asyncio.create_task(self._receive_loop(receive, receipts, state))
        sender = asyncio.create_task(self._send_loop(send, receipts, state))
        tasks = (receiver, sender)
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.warning(
                        "websocket adapter child 失败",
                        extra={
                            "session_id": self._gateway.session_id,
                            "adapter": type(self).__qualname__,
                            "exception_type": type(exc).__qualname__,
                        },
                    )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if state.subscription is not None:
                await state.subscription.aclose()

    async def _receive_loop(
        self,
        receive: Callable[[], Awaitable[str | bytes | None]],
        receipts: asyncio.Queue[CommandReceipt],
        state: _ConnectionState,
    ) -> None:
        """顺序解析 command，只向 receipt queue 写入 typed 结果。"""
        try:
            while True:
                frame = await receive()
                if frame is None:
                    return
                # 在命令改变 manager 状态前预留确认容量，直到唯一 writer 完成发送。
                await state.receipt_capacity.acquire()
                try:
                    command = _parse_frame(frame)
                except (UnicodeDecodeError, ValidationError):
                    receipt = _invalid_command()
                else:
                    receipt = await self._route_command(command, state)
                receipts.put_nowait(receipt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "websocket receive callback 失败",
                extra={
                    "session_id": self._gateway.session_id,
                    "adapter": type(self).__qualname__,
                    "exception_type": type(exc).__qualname__,
                },
            )
        finally:
            state.receiver_done.set()

    async def _route_command(
        self,
        command: _IncomingCommand,
        state: _ConnectionState,
    ) -> CommandReceipt:
        """按 connection phase 路由一个已校验 command。"""
        if isinstance(command, SubscribeCommand):
            if state.phase == "subscribed":
                return CommandRejected(
                    request_id=command.request_id,
                    command_kind="subscribe",
                    code="ALREADY_SUBSCRIBED",
                    message="连接已建立 live subscription",
                )
            return self._accept_subscription(command, state)

        if state.phase == "initial":
            if isinstance(command, SyncCommand):
                state.phase = "awaiting_subscription"
                return await self._gateway.handle(command)
            return CommandRejected(
                request_id=command.request_id,
                command_kind=command.kind,
                code="FIRST_COMMAND_REQUIRED",
                message="首个有效命令必须是 subscribe 或 sync",
            )
        if state.phase == "awaiting_subscription":
            if isinstance(command, SyncCommand):
                return await self._gateway.handle(command)
            return CommandRejected(
                request_id=command.request_id,
                command_kind=command.kind,
                code="SUBSCRIPTION_REQUIRED",
                message="执行该命令前必须先 subscribe",
            )
        return await self._gateway.handle(command)

    def _accept_subscription(
        self,
        command: SubscribeCommand,
        state: _ConnectionState,
    ) -> CommandReceipt:
        """创建唯一 gateway subscription 并返回安全 receipt。"""
        try:
            subscription = self._gateway.subscribe(command)
        except IrisError as exc:
            return CommandRejected(
                request_id=command.request_id,
                command_kind="subscribe",
                code=exc.runtime_code,
                message=exc.message,
            )
        except Exception as exc:
            logger.warning(
                "websocket subscribe 失败",
                extra={
                    "request_id": command.request_id,
                    "session_id": self._gateway.session_id,
                    "adapter": type(self).__qualname__,
                    "exception_type": type(exc).__qualname__,
                },
            )
            return CommandRejected(
                request_id=command.request_id,
                command_kind="subscribe",
                code="INTERNAL_ERROR",
                message="订阅处理失败",
            )
        state.subscription = subscription
        state.phase = "subscribed"
        state.subscription_ready.set()
        return SubscribeAccepted(
            request_id=command.request_id,
            stream_epoch=subscription.stream_epoch,
            scope=subscription.scope,
            scope_id=subscription.scope_id,
        )

    async def _send_loop(
        self,
        send: Callable[[str], Awaitable[None]],
        receipts: asyncio.Queue[CommandReceipt],
        state: _ConnectionState,
    ) -> None:
        """作为连接唯一 writer，合并 receipt 与 live item。"""
        receipt_task: asyncio.Task[CommandReceipt] | None = asyncio.create_task(
            receipts.get()
        )
        ready_task: asyncio.Task[bool] | None = asyncio.create_task(
            state.subscription_ready.wait()
        )
        done_task: asyncio.Task[bool] | None = asyncio.create_task(
            state.receiver_done.wait()
        )
        stream_task: asyncio.Task[GatewayStreamItem] | None = None
        try:
            while True:
                pending = {
                    task
                    for task in (receipt_task, ready_task, done_task, stream_task)
                    if task is not None
                }
                completed, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receipt_task is not None and receipt_task in completed:
                    receipt = receipt_task.result()
                    await send(receipt.model_dump_json())
                    receipts.task_done()
                    state.receipt_capacity.release()
                    receipt_task = asyncio.create_task(receipts.get())
                    continue
                if ready_task is not None and ready_task in completed:
                    state.subscription_ready.clear()
                    ready_task = None
                    if state.subscription is not None:
                        stream_task = asyncio.create_task(anext(state.subscription))
                    continue
                if stream_task is not None and stream_task in completed:
                    try:
                        item = stream_task.result()
                    except StopAsyncIteration:
                        stream_task = None
                    else:
                        await send(item.model_dump_json())
                        stream_task = (
                            asyncio.create_task(anext(state.subscription))
                            if state.subscription is not None
                            else None
                        )
                    continue
                if done_task is not None and done_task in completed:
                    return
        finally:
            for task in (receipt_task, ready_task, done_task, stream_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(
                    task
                    for task in (receipt_task, ready_task, done_task, stream_task)
                    if task is not None
                ),
                return_exceptions=True,
            )


__all__ = ["WebSocketAdapter"]
