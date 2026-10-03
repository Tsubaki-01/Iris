"""服务商无关的流式语音转录接口。"""

from .client import SpeechClient
from .models import TranscriptionEvent
from .protocols import SpeechAdapter

__all__ = ["SpeechAdapter", "SpeechClient", "TranscriptionEvent"]
