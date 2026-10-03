"""公共 PCM 输入检查与转录流委托。"""

from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import aclosing

from ..exceptions import IrisSpeechError
from .models import TranscriptionEvent
from .protocols import SpeechAdapter


def _check_pcm_chunk(chunk: bytes) -> None:
    if not chunk:
        raise IrisSpeechError("PCM 音频块不能为空")
    if len(chunk) % 2:
        raise IrisSpeechError("PCM 音频块必须按 16-bit 采样点对齐")


async def _checked_audio(
    first: bytes, remaining: AsyncIterator[bytes]
) -> AsyncGenerator[bytes, None]:
    yield first
    async for chunk in remaining:
        _check_pcm_chunk(chunk)
        yield chunk


class SpeechClient:
    """复用 adapter 的公共转录入口，每次调用独立管理流的关闭。"""

    def __init__(self, adapter: SpeechAdapter) -> None:
        """保存本次客户端使用的 adapter，构造期间不读取音频。"""
        self._adapter = adapter

    async def stream(self, audio: AsyncIterable[bytes]) -> AsyncGenerator[TranscriptionEvent, None]:
        """惰性消费 raw PCM，并直接转发 adapter 的完整文本事件。

        Args:
            audio: 16 kHz、16-bit little-endian、单声道 PCM 分块流。

        Yields:
            当前全文快照；成功识别最后产生一次 final。

        Raises:
            IrisSpeechError: 音频为空、块无效，或 adapter 识别失败。

        宿主负责音频源和设备；提前退出时应以 aclosing 包围返回的流。
        """
        remaining = aiter(audio)
        try:
            first = await anext(remaining)
        except StopAsyncIteration:
            raise IrisSpeechError("PCM 音频流不能为空") from None
        _check_pcm_chunk(first)

        async with aclosing(_checked_audio(first, remaining)) as checked:
            async with aclosing(self._adapter.stream(checked)) as events:
                async for event in events:
                    yield event


__all__ = ["SpeechClient"]
