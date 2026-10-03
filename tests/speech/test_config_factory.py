"""语音配置、开关与凭据装配的唯一边界。"""

from collections.abc import AsyncGenerator, AsyncIterable
from typing import Literal
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from iris import config as global_config
from iris.config import Config
from iris.exceptions import IrisConfigError
from iris.speech import (
    SpeechClient,
    SpeechConfig,
    TranscriptionEvent,
    create_speech_client,
    factory,
)


def enabled_config(
    adapter: Literal["doubao_asr", "dashscope_funasr"] = "doubao_asr",
) -> SpeechConfig:
    """提供已通过边界校验的配置。"""
    return SpeechConfig(enabled=True, adapter=adapter, endpoint="wss://custom.test", model="asr")


def test_speech_defaults_and_frozen_schema() -> None:
    config = SpeechConfig()
    assert not config.enabled
    assert config.adapter is config.endpoint is config.model is None
    with pytest.raises(ValidationError):
        config.enabled = True


@pytest.mark.parametrize(
    "raw",
    [
        {"enabled": True},
        {"enabled": True, "adapter": "doubao_asr", "endpoint": "wss://test"},
        {"enabled": True, "adapter": "doubao_asr", "model": "asr"},
        {"enabled": True, "endpoint": "wss://test", "model": "asr"},
        {"adapter": "unknown"},
        {"endpoint": "   "},
        {"model": "\t"},
        {"enabled": "true"},
        {"api_key": "not-in-yaml"},
    ],
)
def test_invalid_schema_is_validation_error(raw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SpeechConfig.model_validate(raw)


def test_connection_strings_are_normalized_at_config_boundary() -> None:
    config = SpeechConfig(
        enabled=True, adapter="doubao_asr", endpoint=" wss://custom.test ", model=" resource \t"
    )
    assert config.endpoint == "wss://custom.test"
    assert config.model == "resource"


def test_disabled_factory_never_reads_keys_or_constructs_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = Mock(side_effect=AssertionError("disabled speech must not construct or read keys"))
    for name in ("get_config", "SpeechClient", "DoubaoASRAdapter", "DashScopeFunASRAdapter"):
        monkeypatch.setattr(factory, name, forbidden)
    assert create_speech_client(SpeechConfig(), api_key=" ") is None
    forbidden.assert_not_called()


@pytest.mark.parametrize("adapter", ["doubao_asr", "dashscope_funasr"])
def test_explicit_key_needs_no_global_config_and_uses_exact_adapter(
    monkeypatch: pytest.MonkeyPatch, adapter: Literal["doubao_asr", "dashscope_funasr"]
) -> None:
    read_config = Mock(side_effect=AssertionError("explicit key must not read global config"))
    constructor = Mock()
    other_constructor = Mock(side_effect=AssertionError("adapter must not be inferred from URL"))
    monkeypatch.setattr(factory, "get_config", read_config)
    selected = "DoubaoASRAdapter" if adapter == "doubao_asr" else "DashScopeFunASRAdapter"
    other = "DashScopeFunASRAdapter" if adapter == "doubao_asr" else "DoubaoASRAdapter"
    monkeypatch.setattr(factory, selected, constructor)
    monkeypatch.setattr(factory, other, other_constructor)
    speech = create_speech_client(enabled_config(adapter), api_key=" explicit-key \n")
    assert isinstance(speech, SpeechClient)
    constructor.assert_called_once_with(
        endpoint="wss://custom.test", model="asr", api_key="explicit-key"
    )
    read_config.assert_not_called()
    other_constructor.assert_not_called()


@pytest.mark.parametrize("adapter", ["doubao_asr", "dashscope_funasr"])
def test_global_key_uses_normalized_provider_namespace(
    monkeypatch: pytest.MonkeyPatch, adapter: Literal["doubao_asr", "dashscope_funasr"]
) -> None:
    config = Config(api_key="chat-key", provider_api_keys={adapter.upper(): " speech-key "})
    monkeypatch.setattr(factory, "get_config", lambda: config)
    constructor = Mock()
    name = "DoubaoASRAdapter" if adapter == "doubao_asr" else "DashScopeFunASRAdapter"
    monkeypatch.setattr(factory, name, constructor)
    assert create_speech_client(enabled_config(adapter)) is not None
    constructor.assert_called_once_with(
        endpoint="wss://custom.test", model="asr", api_key="speech-key"
    )


def test_uninitialized_global_config_is_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "_config", None)
    with pytest.raises(IrisConfigError, match="尚未初始化"):
        create_speech_client(enabled_config())


def test_chat_key_does_not_replace_missing_speech_key(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config(api_key="chat-key", provider_api_keys={"other": "other-key"})
    monkeypatch.setattr(factory, "get_config", lambda: config)
    with pytest.raises(IrisConfigError, match="doubao_asr"):
        create_speech_client(enabled_config())


@pytest.mark.parametrize("key", ["", " \t\n"])
def test_explicit_empty_key_never_falls_back(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    read_config = Mock(side_effect=AssertionError("explicit empty key must fail before fallback"))
    monkeypatch.setattr(factory, "get_config", read_config)
    with pytest.raises(IrisConfigError):
        create_speech_client(enabled_config(), api_key=key)
    read_config.assert_not_called()


@pytest.mark.asyncio
async def test_same_consumer_works_with_either_constructed_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []

    class Adapter:
        """只实现统一流接口的构造替身。"""

        def __init__(self, *, endpoint: str, model: str, api_key: str) -> None:
            calls.append((endpoint, model, api_key))

        async def stream(
            self, audio: AsyncIterable[bytes]
        ) -> AsyncGenerator[TranscriptionEvent, None]:
            """不暴露厂商字段。"""
            async for _chunk in audio:
                pass
            yield TranscriptionEvent("转录文本", True)

    async def audio() -> AsyncGenerator[bytes, None]:
        yield b"ab"

    monkeypatch.setattr(factory, "DoubaoASRAdapter", Adapter)
    monkeypatch.setattr(factory, "DashScopeFunASRAdapter", Adapter)
    for adapter in ("doubao_asr", "dashscope_funasr"):
        speech = create_speech_client(enabled_config(adapter), api_key=adapter)
        assert speech is not None
        assert [event async for event in speech.stream(audio())] == [
            TranscriptionEvent("转录文本", True)
        ]
    assert calls == [
        ("wss://custom.test", "asr", "doubao_asr"),
        ("wss://custom.test", "asr", "dashscope_funasr"),
    ]
