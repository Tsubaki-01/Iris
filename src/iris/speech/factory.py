"""在唯一启用开关和凭据边界装配语音客户端。"""

from typing import cast

from ..config import get_config
from ..exceptions import IrisConfigError
from .adapters.dashscope_funasr import DashScopeFunASRAdapter
from .adapters.doubao import DoubaoASRAdapter
from .client import SpeechClient
from .config import SpeechAdapterName, SpeechConfig
from .protocols import SpeechAdapter


def create_speech_client(
    config: SpeechConfig, *, api_key: str | None = None
) -> SpeechClient | None:
    """按已验证配置选择 adapter，构造过程不连接服务或读取音频。

    Args:
        config: 语音输入声明。
        api_key: 可选显式语音凭据；提供时覆盖全局专属凭据。

    Returns:
        关闭时为 None，开启时为可复用的 SpeechClient。

    Raises:
        IrisConfigError: 显式 key 为空、全局配置未初始化或缺少专属凭据。
    """
    if not config.enabled:
        return None

    adapter_name = cast(SpeechAdapterName, config.adapter)
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            raise IrisConfigError("显式语音 API key 不能为空", adapter=adapter_name)
    else:
        api_key = get_config().provider_api_keys.get(adapter_name)
        if api_key is None:
            raise IrisConfigError("缺少语音服务专属 API key", adapter=adapter_name)

    # 启用字段的完整性由 SpeechConfig 保证，此处只做静态类型收窄。
    endpoint, model = cast(str, config.endpoint), cast(str, config.model)
    adapter: SpeechAdapter
    if adapter_name == "doubao_asr":
        adapter = DoubaoASRAdapter(endpoint=endpoint, model=model, api_key=api_key)
    else:
        adapter = DashScopeFunASRAdapter(endpoint=endpoint, model=model, api_key=api_key)
    return SpeechClient(adapter)


__all__ = ["create_speech_client"]
