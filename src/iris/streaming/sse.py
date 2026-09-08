"""SSE framing、cursor codec 与 heartbeat iterator。

本模块不创建 HTTP route；host 负责认证、header/body 读取与 response lifecycle。

Example:
    async for frame in SSEAdapter(heartbeat_interval_s=15).stream(subscription):
        await send(frame)
"""

# region imports

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import suppress

from .gateway import GatewaySubscription
from .models import GatewayStreamItem, LiveCursor, LiveEnvelope

# endregion

_HEARTBEAT = b": heartbeat\n\n"


def encode_live_cursor(cursor: LiveCursor) -> str:
    """把 typed live cursor 编码为 compact single-line JSON。"""
    return cursor.model_dump_json()


def decode_live_cursor(value: str) -> LiveCursor:
    """在 raw header boundary 解析并验证一个 live cursor。"""
    if "\r" in value or "\n" in value:
        raise ValueError("live cursor 必须是 single-line JSON")
    return LiveCursor.model_validate_json(value)


def _encode_item(item: GatewayStreamItem) -> bytes:
    """把 typed gateway item 编码为单个 SSE frame。"""
    lines: list[str] = []
    if isinstance(item, LiveEnvelope):
        lines.append(
            "id: "
            + encode_live_cursor(
                LiveCursor.model_construct(
                    stream_epoch=item.stream_epoch,
                    scope=item.scope,
                    scope_id=item.scope_id,
                    after_live_sequence=item.live_sequence,
                )
            )
        )
    lines.extend((f"event: {item.kind}", f"data: {item.model_dump_json()}"))
    return ("\n".join(lines) + "\n\n").encode()


class SSEAdapter:
    """把一个 gateway subscription 适配为 SSE bytes iterator。"""

    def __init__(self, *, heartbeat_interval_s: float) -> None:
        """绑定显式有限正 heartbeat interval。

        Raises:
            ValueError: Interval 不是有限正数。
        """
        if (
            isinstance(heartbeat_interval_s, bool)
            or not isinstance(heartbeat_interval_s, (int, float))
            or not math.isfinite(heartbeat_interval_s)
            or heartbeat_interval_s <= 0
        ):
            raise ValueError("heartbeat_interval_s 必须是有限正数")
        self._heartbeat_interval_s = float(heartbeat_interval_s)

    async def stream(
        self,
        subscription: GatewaySubscription,
    ) -> AsyncIterator[bytes]:
        """交替产生 typed item frame 与不占 sequence 的 heartbeat。"""
        pending = asyncio.create_task(anext(subscription))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {pending},
                    timeout=self._heartbeat_interval_s,
                )
                if not done:
                    yield _HEARTBEAT
                    continue
                try:
                    item = pending.result()
                except StopAsyncIteration:
                    return
                pending = asyncio.create_task(anext(subscription))
                yield _encode_item(item)
        finally:
            if not pending.done():
                pending.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
            await subscription.aclose()


__all__ = ["SSEAdapter", "decode_live_cursor", "encode_live_cursor"]
