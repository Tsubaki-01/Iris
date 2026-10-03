"""语音转录 adapter 的静态调用契约。"""

from collections.abc import AsyncGenerator, AsyncIterable
from typing import Protocol

from .models import TranscriptionEvent


class SpeechAdapter(Protocol):
    """将已检查的 PCM 流映射为完整文本快照。"""

    def stream(self, audio: AsyncIterable[bytes]) -> AsyncGenerator[TranscriptionEvent, None]:
        """返回拥有本次连接与任务的可关闭异步生成器。"""
        ...


__all__ = ["SpeechAdapter"]
