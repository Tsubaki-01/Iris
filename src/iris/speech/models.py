"""服务商无关的语音转录结果。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptionEvent:
    """当前整段转录全文，以及整次识别是否正常完成。"""

    text: str
    is_final: bool


__all__ = ["TranscriptionEvent"]
