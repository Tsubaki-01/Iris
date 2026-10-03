"""原生语音 adapter 共享的连接、传输错误和收发任务生命周期。"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from ...exceptions import IrisSpeechError

_CONNECT_TIMEOUT = 10.0


@dataclass(frozen=True, slots=True)
class WebSocketTransport:
    """一个已连接的 socket 与其错误归因，不解释厂商消息。"""

    socket: ClientConnection
    adapter: str
    request_id: str

    def error(self, message: str, **context: object) -> IrisSpeechError:
        """为当前连接构造语音领域错误。"""
        return IrisSpeechError(message, adapter=self.adapter, request_id=self.request_id, **context)

    async def send(self, frame: bytes | str) -> None:
        """发送原生帧，仅归一化此操作的网络异常。"""
        try:
            await self.socket.send(frame)
        except (OSError, WebSocketException) as exc:
            raise self.error("语音数据发送失败") from exc

    async def receive(self) -> bytes | str:
        """接收原生帧，连接关闭不能替代服务协议终态。"""
        try:
            return await self.socket.recv()
        except (OSError, WebSocketException) as exc:
            raise self.error("语音在正常终态前接收失败") from exc


@asynccontextmanager
async def open_transport(
    endpoint: str, headers: dict[str, str], *, adapter: str, request_id: str
) -> AsyncGenerator[WebSocketTransport, None]:
    """拥有连接作用域，仅在建立连接的边界归一化网络异常。"""
    async with AsyncExitStack() as stack:
        try:
            socket = await stack.enter_async_context(
                connect(endpoint, additional_headers=headers, open_timeout=_CONNECT_TIMEOUT)
            )
        except (OSError, WebSocketException) as exc:
            raise IrisSpeechError("语音连接失败", adapter=adapter, request_id=request_id) from exc
        yield WebSocketTransport(socket=socket, adapter=adapter, request_id=request_id)


async def received_messages(
    transport: WebSocketTransport, sender: asyncio.Task[float], *, final_timeout: float
) -> AsyncGenerator[bytes | str, None]:
    """监听发送失败并逐帧接收，关闭时取消排空两个任务。

    Args:
        transport: 当前连接。
        sender: 厂商发送协程；成功返回结束输入动作完成时的 loop.time()。
        final_timeout: 结束输入动作完成后等待服务终态的绝对期限。

    调用方以 aclosing 包围本生成器；本 helper 接管 sender 的取消和 await。
    """
    receiver = asyncio.create_task(transport.receive())
    deadline: float | None = None
    try:
        while True:
            pending: set[asyncio.Task[object]] = {receiver}
            if deadline is None:
                pending.add(sender)
            try:
                async with asyncio.timeout_at(deadline):
                    done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            except TimeoutError as exc:
                raise transport.error("语音等待最终结果超时") from exc
            if sender in done:
                deadline = sender.result() + final_timeout
            if receiver in done:
                frame = receiver.result()
                receiver = asyncio.create_task(transport.receive())
                yield frame
    finally:
        sender.cancel()
        receiver.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)
