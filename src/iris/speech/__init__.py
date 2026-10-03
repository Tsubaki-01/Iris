"""服务商无关的流式语音转录接口。"""

from .client import SpeechClient
from .config import SpeechConfig
from .factory import create_speech_client
from .models import TranscriptionEvent
from .protocols import SpeechAdapter

__all__ = [
    "SpeechAdapter",
    "SpeechClient",
    "SpeechConfig",
    "TranscriptionEvent",
    "create_speech_client",
]
