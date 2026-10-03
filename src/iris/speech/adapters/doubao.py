"""豆包原生 WebSocket ASR 的流式转录 adapter。"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import aclosing
from dataclasses import dataclass, field
from uuid import uuid4

from ..models import TranscriptionEvent
from ._doubao_protocol import decode_response, encode_audio, encode_request
from ._transport import WebSocketTransport, open_transport, received_messages

_FINAL_TIMEOUT = 10.0


async def _send_audio(
    transport: WebSocketTransport,
    audio: AsyncIterable[bytes],
    ending: asyncio.Event,
) -> float:
    chunks = aiter(audio)
    last = await anext(chunks)
    sequence = 2
    async for chunk in chunks:
        await transport.send(encode_audio(last, sequence))
        last = chunk
        sequence += 1
    ending.set()
    await transport.send(encode_audio(last, sequence, final=True))
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
        async with open_transport(
            self.endpoint,
            {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.model,
                "X-Api-Request-Id": request_id,
            },
            adapter="doubao_asr",
            request_id=request_id,
        ) as transport:
            await transport.send(encode_request())
            ending = asyncio.Event()
            sender = asyncio.create_task(_send_audio(transport, audio, ending))
            text = ""
            async with aclosing(
                received_messages(transport, sender, final_timeout=_FINAL_TIMEOUT)
            ) as messages:
                async for frame in messages:
                    response = decode_response(frame, request_id=request_id)
                    if response.text is not None:
                        text = response.text
                    if response.is_final:
                        if not ending.is_set():
                            raise transport.error("豆包语音在音频输入结束前返回终态")
                        await sender
                        yield TranscriptionEvent(text=text, is_final=True)
                        return
                    if response.text is not None:
                        yield TranscriptionEvent(text=text, is_final=False)


__all__ = ["DoubaoASRAdapter"]
