"""通过真实 LiteLLM 与 HTTP MockTransport 验证原生 /responses。"""

import json
from typing import Any

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from iris.config import Config, ProviderConfig
from iris.exceptions import IrisProviderError
from iris.message import LLMRequest, Msg, ToolUseBlock
from iris.providers import create_provider_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "overrides", "explicit_base", "expected_url"),
    [
        ("openai/gpt-4o", {}, None, "https://api.openai.com/v1/responses"),
        ("deepseek/deepseek-flash", {}, None, "https://api.deepseek.com/responses"),
        (
            "deepseek/deepseek-flash",
            {"deepseek": ProviderConfig(headers={"x-config": "yes"})},
            None,
            "https://api.deepseek.com/responses",
        ),
        (
            "deepseek/deepseek-flash",
            {"deepseek": ProviderConfig(base_url="https://gateway.test/v1")},
            "https://explicit.test/v1/",
            "https://explicit.test/v1/responses",
        ),
    ],
)
async def test_responses_native_http_route(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    overrides: dict[str, ProviderConfig],
    explicit_base: str | None,
    expected_url: str,
) -> None:
    import litellm
    import litellm.llms.custom_httpx.llm_http_handler as http_handler

    import iris.providers.factory as factory

    logical, model = route.split("/", 1)
    config = Config(provider_api_keys={logical: "logical-test-key"}, providers=overrides)
    monkeypatch.setattr(factory, "is_config_initialized", lambda: True)
    monkeypatch.setattr(factory, "get_config", lambda: config)

    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp-native",
                "created_at": 1,
                "object": "response",
                "model": model,
                "status": "completed",
                "store": False,
                "output": [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        )

    async def no_chat(**kwargs: Any) -> None:
        pytest.fail("native Responses must never invoke acompletion")

    monkeypatch.setattr(litellm, "acompletion", no_chat)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler = object.__new__(AsyncHTTPHandler)
        handler.client = http_client
        handler.timeout = 5
        monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda **kwargs: handler)
        client = create_provider_client(
            route, base_url=explicit_base, headers={"x-explicit": "yes"}
        )
        result = await client.complete(
            LLMRequest(
                model=model,
                messages=[
                    Msg.user("lookup"),
                    Msg.assistant([ToolUseBlock(id="call-1", name="lookup", input={})]),
                    Msg.tool_result(tool_use_id="call-1", content="result"),
                ],
                tools=[
                    {
                        "name": "lookup",
                        "description": "lookup",
                        "input_schema": {"type": "object", "properties": {}},
                        "strict": False,
                    }
                ],
                tool_choice={"name": "lookup"},
                provider_options={"num_retries": 0},
            )
        )
    assert result.provider == logical and result.to_msg().text == "done"
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == expected_url
    assert request.headers["authorization"] == "Bearer logical-test-key"
    assert request.headers["x-explicit"] == "yes"
    if "deepseek" in overrides and overrides["deepseek"].headers:
        assert request.headers["x-config"] == "yes"
    body = json.loads(request.content)
    assert body["model"] == model and body["store"] is False
    assert "messages" not in body and "previous_response_id" not in body
    assert body["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": [{"type": "input_text", "text": "result"}],
    }
    assert body["tools"][0]["name"] == "lookup" and "function" not in body["tools"][0]
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
async def test_responses_explicit_chat_transport_is_rejected_without_bridge(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    import litellm

    import iris.providers.factory as factory

    config = Config(providers={provider: ProviderConfig(litellm_provider=provider)})
    monkeypatch.setattr(factory, "is_config_initialized", lambda: True)
    monkeypatch.setattr(factory, "get_config", lambda: config)

    async def no_request(**kwargs: Any) -> None:
        pytest.fail("unsupported transport must not reach LiteLLM")

    monkeypatch.setattr(litellm, "aresponses", no_request)
    monkeypatch.setattr(litellm, "acompletion", no_request)
    client = create_provider_client(f"{provider}/model", api_key="test")
    with pytest.raises(IrisProviderError, match="原生 Responses"):
        await client.complete(LLMRequest(model="model", messages=[Msg.user("hello")]))
