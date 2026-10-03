"""流式语音 adapter 共用的内存 WebSocket。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import TracebackType

Frame = bytes | str


class FakeWebSocket:
    """记录收发并允许测试控制外部响应，不启动真实端口。"""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Frame | Exception] = asyncio.Queue()
        self.sent: list[Frame] = []
        self.on_send: Callable[[Frame], Awaitable[None]] | None = None
        self.receive_waiting = asyncio.Event()
        self.active_receives = 0
        self.max_receives = 0
        self.closed = False
        self.open_error: Exception | None = None

    async def __aenter__(self) -> FakeWebSocket:
        """模拟连接建立。"""
        if self.open_error is not None:
            raise self.open_error
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """记录连接作用域已退出。"""
        self.closed = True

    async def send(self, frame: Frame) -> None:
        """记录发送并执行测试场景的服务响应。"""
        self.sent.append(frame)
        if self.on_send is not None:
            await self.on_send(frame)

    async def recv(self) -> Frame:
        """等待一条服务帧或指定异常。"""
        self.active_receives += 1
        self.max_receives = max(self.max_receives, self.active_receives)
        self.receive_waiting.set()
        try:
            frame = await self.incoming.get()
            if isinstance(frame, Exception):
                raise frame
            return frame
        finally:
            self.active_receives -= 1


class FakeConnect:
    """按调用顺序提供连接并记录握手参数。"""

    def __init__(self, *sockets: FakeWebSocket) -> None:
        self.sockets = iter(sockets)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(
        self, endpoint: str, *, additional_headers: dict[str, str], open_timeout: float
    ) -> FakeWebSocket:
        """返回下一次识别使用的连接。"""
        self.calls.append((endpoint, additional_headers, open_timeout))
        return next(self.sockets)
