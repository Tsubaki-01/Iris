import httpx
import pytest
from httpx import Headers

from iris.exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from iris.message import LLMRequest, Msg
from iris.providers import AnthropicMessageAdapter, OpenAIMessageAdapter, ProviderClient


def test_provider_client_does_not_expose_unimplemented_retry_field() -> None:
    assert "max_retries" not in ProviderClient.model_fields


@pytest.mark.asyncio
async def test_provider_client_posts_openai_chat_payload_to_default_endpoint() -> None:
    seen_url = ""
    seen_headers = Headers()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_headers, seen_url
        seen_url = str(request.url)
        seen_headers = request.headers
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "gpt-4o",
                "choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ProviderClient(
        adapter=OpenAIMessageAdapter(),
        api_key="test-key",
        http_client=http_client,
    )

    response = await client.complete(
        LLMRequest(model="gpt-4o", messages=[Msg.user("你好")])
    )

    assert seen_url == "https://api.openai.com/v1/chat/completions"
    assert seen_headers["authorization"] == "Bearer test-key"
    assert response.to_msg().text == "你好"
    await client.close()


@pytest.mark.asyncio
async def test_provider_client_uses_openai_responses_endpoint_when_requested() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "好"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    client = ProviderClient(
        adapter=OpenAIMessageAdapter(),
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    await client.complete(
        LLMRequest(
            model="gpt-4o",
            messages=[Msg.user("你好")],
            provider_options={"api_style": "responses"},
        )
    )

    assert seen["url"] == "https://api.openai.com/v1/responses"
    await client.close()


@pytest.mark.asyncio
async def test_provider_client_posts_anthropic_payload_to_messages_endpoint() -> None:
    seen_url = ""
    seen_headers = Headers()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_headers, seen_url
        seen_url = str(request.url)
        seen_headers = request.headers
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "你好"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    client = ProviderClient(
        adapter=AnthropicMessageAdapter(),
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    response = await client.complete(
        LLMRequest(model="claude-sonnet-4-5", messages=[Msg.user("你好")])
    )

    assert seen_url == "https://api.anthropic.com/v1/messages"
    assert seen_headers["x-api-key"] == "test-key"
    assert seen_headers["anthropic-version"] == "2023-06-01"
    assert response.to_msg().text == "你好"
    await client.close()


@pytest.mark.asyncio
async def test_provider_client_keeps_custom_anthropic_version_header() -> None:
    seen_headers = Headers()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_headers
        seen_headers = request.headers
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "你好"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    client = ProviderClient(
        adapter=AnthropicMessageAdapter(),
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        headers={"anthropic-version": "2024-01-01"},
    )

    await client.complete(
        LLMRequest(model="claude-sonnet-4-5", messages=[Msg.user("你好")])
    )

    assert seen_headers["anthropic-version"] == "2024-01-01"
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, IrisAuthenticationError),
        (403, IrisAuthenticationError),
        (429, IrisRateLimitExceededError),
        (500, IrisProviderError),
    ],
)
async def test_provider_client_maps_http_errors(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": {"message": "失败"}})

    client = ProviderClient(
        adapter=OpenAIMessageAdapter(),
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(expected_error):
        await client.complete(LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]))

    await client.close()


@pytest.mark.asyncio
async def test_provider_client_maps_connection_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("无法连接", request=request)

    client = ProviderClient(
        adapter=OpenAIMessageAdapter(),
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(IrisAPIConnectionError):
        await client.complete(LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]))

    await client.close()


@pytest.mark.asyncio
async def test_provider_client_rejects_streaming_in_complete() -> None:
    client = ProviderClient(adapter=OpenAIMessageAdapter(), api_key="test-key")

    with pytest.raises(IrisProviderError, match="stream"):
        await client.complete(LLMRequest(model="gpt-4o", stream=True))

    await client.close()
