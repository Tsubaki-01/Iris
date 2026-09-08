"""配置字段拥有唯一生效入口。"""

import pytest
from pydantic import BaseModel, ValidationError

from iris.agents import ModelConfig
from iris.config import Config, ProviderConfig
from iris.providers import create_provider_client


def test_model_options_do_not_choose_host_transport() -> None:
    """ModelConfig 只描述模型参数，stream transport 由 host 注入决定。"""
    model = ModelConfig(provider="openai", name="gpt-4o", temperature=0.4, timeout=17)
    assert "stream" not in ModelConfig.model_fields
    assert model.to_llm_request_options() == {
        "temperature": 0.4,
        "timeout": 17,
        "provider_options": {},
        "metadata": {},
    }


def test_global_config_does_not_expose_inert_provider_options() -> None:
    """全局配置只提供密钥和 provider registry，client 参数有实际 consumer。"""
    assert not {"base_url", "timeout", "debug"} & Config.model_fields.keys()
    client = create_provider_client(
        "openai/gpt-4o", api_key="test", base_url="http://localhost:8000/v1", timeout=17
    )
    assert client.base_url == "http://localhost:8000/v1"
    assert client.timeout == 17


@pytest.mark.parametrize(
    ("config_type", "values"),
    [
        (ModelConfig, {"provider": "openai", "name": "gpt-4o", "api_style": "chat"}),
        (ProviderConfig, {"api_style": "chat"}),
    ],
)
def test_config_rejects_api_style_without_a_behavior_choice(
    config_type: type[BaseModel], values: dict[str, str]
) -> None:
    """唯一 Chat 调用契约不再接受无行为选择的配置字段。"""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        config_type.model_validate(values)
