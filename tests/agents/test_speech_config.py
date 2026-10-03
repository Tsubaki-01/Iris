"""Agent YAML 只加载语音声明，不在加载时装配输入设备或服务。"""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from iris.agents import AgentConfig, load_agent_config
from iris.exceptions import IrisConfigError
from iris.speech import SpeechConfig, factory


def test_agent_without_speech_stays_disabled(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("name: agent\nmodel: openai/test\nsystem: system\n", encoding="utf-8")
    config = load_agent_config(path)
    assert config.speech == SpeechConfig()


@pytest.mark.parametrize("adapter", ["doubao_asr", "dashscope_funasr"])
def test_agent_loading_is_declarative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, adapter: str
) -> None:
    forbidden = Mock(side_effect=AssertionError("loading YAML must not construct speech"))
    for name in ("get_config", "SpeechClient", "DoubaoASRAdapter", "DashScopeFunASRAdapter"):
        monkeypatch.setattr(factory, name, forbidden)
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: system\n"
        f"speech:\n  enabled: true\n  adapter: {adapter}\n"
        "  endpoint: wss://custom.test\n  model: asr-model\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.speech.enabled
    assert config.speech.adapter == adapter
    assert config.speech.endpoint == "wss://custom.test"
    assert config.speech.model == "asr-model"
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "speech", [{"enabled": True}, {"adapter": "unknown"}, {"enabled": False, "unknown": "field"}]
)
def test_direct_model_and_yaml_preserve_their_error_boundaries(
    tmp_path: Path, speech: dict[str, object]
) -> None:
    raw = {"name": "agent", "model": "openai/test", "system": "system", "speech": speech}
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(raw)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(IrisConfigError) as caught:
        load_agent_config(path)
    assert isinstance(caught.value.__cause__, ValidationError)
