"""语音输入与转录服务异常。"""

from .provider import IrisProviderError


class IrisSpeechError(IrisProviderError):
    """音频输入无效或语音转录未能完成。"""

    runtime_error_code = "SPEECH_ERROR"


__all__ = ["IrisSpeechError"]
