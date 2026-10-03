"""豆包原生 WebSocket ASR 的流式转录 adapter。"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from ...exceptions import IrisSpeechError
from ..models import TranscriptionEvent
from ._doubao_protocol import decode_response, encode_audio, encode_request

_CONNECT_TIMEOUT = 10.0
_FINAL_TIMEOUT = 10.0


async def _send(socket: ClientConnection, frame: bytes, request_id: str) -> None:
    try:
        await socket.send(frame)
    except (OSError, WebSocketException) as exc:
        raise IrisSpeechError(
            "豆包语音发送失败", adapter="doubao_asr", request_id=request_id
        ) from exc


async def _receive(socket: ClientConnection, request_id: str) -> bytes | str:
    try:
        return await socket.recv()
    except (OSError, WebSocketException) as exc:
        raise IrisSpeechError(
            "豆包语音在正常终态前接收失败", adapter="doubao_asr", request_id=request_id
        ) from exc


async def _send_audio(
    socket: ClientConnection,
    audio: AsyncIterable[bytes],
    request_id: str,
    ending: asyncio.Event,
) -> float:
    chunks = aiter(audio)
    last = await anext(chunks)
    sequence = 2
    async for chunk in chunks:
        await _send(socket, encode_audio(last, sequence), request_id)
        last = chunk
        sequence += 1
    ending.set()
    await _send(socket, encode_audio(last, sequence, final=True), request_id)
    return asyncio.get_running_loop().time()


@dataclass(frozen=True, slots=True, kw_only=True)
class DoubaoASRAdapter:
    """保存豆包连接参数；每次 stream 独立拥有连接和识别状态。

    Args:
        endpoint: 豆包原生 ASR WebSocket 地址。
        model: 已开通的语音资源 ID，用于 X-Api-Resource-Id。
        api_key: 豆包语音服务 API Key。
    """

    endpoint: str
    model: str
    api_key: str = field(repr=False)

    async def stream(self, audio: AsyncIterable[bytes]) -> AsyncGenerator[TranscriptionEvent, None]:
        """并行收发已检查的 PCM，服务正常确认末包后产出 final。"""
        request_id = str(uuid4())
        async with AsyncExitStack() as stack:
            try:
                socket = await stack.enter_async_context(
                    connect(
                        self.endpoint,
                        additional_headers={
                            "X-Api-Key": self.api_key,
                            "X-Api-Resource-Id": self.model,
                            "X-Api-Request-Id": request_id,
                        },
                        open_timeout=_CONNECT_TIMEOUT,
                    )
                )
            except (OSError, WebSocketException) as exc:
                raise IrisSpeechError(
                    "豆包语音连接失败", adapter="doubao_asr", request_id=request_id
                ) from exc

            await _send(socket, encode_request(), request_id)
            ending = asyncio.Event()
            sender = asyncio.create_task(_send_audio(socket, audio, request_id, ending))
            receiver = asyncio.create_task(_receive(socket, request_id))
            deadline: float | None = None
            text = ""
            try:
                while True:
                    pending: set[asyncio.Task[object]] = {receiver}
                    if deadline is None:
                        pending.add(sender)
                    try:
                        async with asyncio.timeout_at(deadline):
                            done, _ = await asyncio.wait(
                                pending, return_when=asyncio.FIRST_COMPLETED
                            )
                    except TimeoutError as exc:
                        raise IrisSpeechError(
                            "豆包语音等待最终结果超时", adapter="doubao_asr", request_id=request_id
                        ) from exc
                    if sender in done:
                        deadline = sender.result() + _FINAL_TIMEOUT
                    if receiver not in done:
                        continue

                    response = decode_response(receiver.result(), request_id=request_id)
                    if response.text is not None:
                        text = response.text
                    if response.is_final:
                        if not ending.is_set():
                            raise IrisSpeechError(
                                "豆包语音在音频输入结束前返回终态",
                                adapter="doubao_asr",
                                request_id=request_id,
                            )
                        await sender
                        yield TranscriptionEvent(text=text, is_final=True)
                        return
                    receiver = asyncio.create_task(_receive(socket, request_id))
                    if response.text is not None:
                        yield TranscriptionEvent(text=text, is_final=False)
            finally:
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)


__all__ = ["DoubaoASRAdapter"]
