"""Responses 图片请求与同源视觉计量的协议边界。"""

from __future__ import annotations

import base64
import importlib
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from iris.exceptions import IrisImageError, IrisProviderError
from iris.message import (
    ImageBlock,
    ImageFileRef,
    LLMRequest,
    ModelResponseCompleted,
    ModelResponseFailed,
    Msg,
    Role,
    TextBlock,
    ToolUseBlock,
)
from iris.providers import ProviderClient
from iris.providers.responses import ResponsesAdapter, ResponsesMapper


@pytest.fixture
def offline_text_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """只替换文字 tokenizer，保留真实图片解码、尺寸和数量计量。"""
    counter = importlib.import_module("litellm.litellm_core_utils.token_counter")
    monkeypatch.setattr(counter, "_get_count_function", lambda model, custom_tokenizer: len)


def _image(
    directory: Path,
    name: str,
    *,
    size: tuple[int, int] = (64, 32),
    image_format: str = "PNG",
    padding: bytes = b"",
) -> ImageBlock:
    """构造真实模型副本；原图路径故意不可读以证明请求只读取模型版。"""
    output = BytesIO()
    with Image.new("RGB", size, "blue") as image:
        image.save(output, format=image_format)
    path = directory / f"{name}.{image_format.lower()}"
    path.write_bytes(output.getvalue() + padding)
    return ImageBlock(
        original=ImageFileRef(
            path=directory / f"{name}.absent-original.png",
            mime_type="image/png",
            width=3000,
            height=2000,
        ),
        model=ImageFileRef(
            path=path, mime_type=f"image/{image_format.lower()}", width=size[0], height=size[1]
        ),
        name=name,
    )


def _image_part(block: ImageBlock) -> dict[str, str]:
    return {
        "type": "input_image",
        "image_url": (
            f"data:{block.model.mime_type};base64,"
            + base64.b64encode(block.model.path.read_bytes()).decode("ascii")
        ),
        "detail": "high",
    }


def _client() -> ProviderClient:
    return ProviderClient(provider="openai", api_key="test")


def test_user_mixed_images_preserve_order_model_encoding_and_history(tmp_path: Path) -> None:
    first = _image(tmp_path, "first", image_format="WEBP")
    second = _image(tmp_path, "second")
    message = Msg(
        role=Role.USER,
        content=[TextBlock(text="先看"), first, TextBlock(text="再看"), second],
    )
    before = message.model_dump_json()
    assert ResponsesMapper().format_messages([message]) == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "先看"},
                _image_part(first),
                {"type": "input_text", "text": "再看"},
                _image_part(second),
            ],
        }
    ]
    assert message.model_dump_json() == before


def test_image_only_user_keeps_one_message(tmp_path: Path) -> None:
    image = _image(tmp_path, "only")
    assert ResponsesMapper().format_messages([Msg(role=Role.USER, content=[image])]) == [
        {"type": "message", "role": "user", "content": [_image_part(image)]}
    ]


def test_parallel_tool_images_remain_in_their_own_ordered_outputs(tmp_path: Path) -> None:
    first = _image(tmp_path, "first")
    second = _image(tmp_path, "second", image_format="WEBP")
    calls = Msg.assistant(
        [
            ToolUseBlock(id="call-a", name="a", input={}),
            ToolUseBlock(id="call-b", name="b", input={}),
        ]
    )
    results = [
        Msg.tool_result(
            tool_use_id="call-a",
            name="a",
            content=[TextBlock(text="before"), first, TextBlock(text="after")],
        ),
        Msg.tool_result(tool_use_id="call-b", name="b", content=[second, first]),
    ]
    before = [message.model_dump_json() for message in [calls, *results]]
    items = ResponsesMapper().format_messages([calls, *results])
    assert [item["type"] for item in items] == [
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert items[2:] == [
        {
            "type": "function_call_output",
            "call_id": "call-a",
            "output": [
                {"type": "input_text", "text": "before"},
                _image_part(first),
                {"type": "input_text", "text": "after"},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call-b",
            "output": [_image_part(second), _image_part(first)],
        },
    ]
    assert [message.model_dump_json() for message in [calls, *results]] == before


def test_count_projection_preserves_images_and_plain_text_shape(tmp_path: Path) -> None:
    image = _image(tmp_path, "model", image_format="WEBP")
    request = LLMRequest(
        model="gpt-4o",
        messages=[
            Msg.user("plain"),
            Msg(role=Role.USER, content=[TextBlock(text="look"), image]),
            Msg.tool_result(tool_use_id="call-a", content=[image, TextBlock(text="result")]),
        ],
    )
    projection = ResponsesAdapter().token_count_projection(request, transport="openai")
    counted_image = {
        "type": "image_url",
        "image_url": {"url": _image_part(image)["image_url"], "detail": "high"},
    }
    assert projection["messages"] == [
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [{"type": "text", "text": "look"}, counted_image]},
        {
            "role": "tool",
            "tool_call_id": "call-a",
            "content": [counted_image, {"type": "text", "text": "result"}],
        },
    ]


@pytest.mark.parametrize("tool_result", [False, True])
@pytest.mark.usefixtures("offline_text_counter")
def test_image_cost_depends_on_dimensions_and_count_once(tmp_path: Path, tool_result: bool) -> None:
    small = _image(tmp_path, "small", size=(64, 64))
    large = _image(tmp_path, "large", size=(1024, 1024))
    client = _client()

    def count(images: list[ImageBlock]) -> int:
        content = [TextBlock(text="look"), *images]
        message = (
            Msg.tool_result(tool_use_id="call-a", content=content)
            if tool_result
            else Msg(role=Role.USER, content=content)
        )
        return client.estimate_input_tokens(LLMRequest(model="gpt-4o", messages=[message]))

    base = count([])
    single = count([small]) - base
    assert single > 0
    assert count([large]) > count([small])
    assert count([small, small]) - base == 2 * single


@pytest.mark.usefixtures("offline_text_counter")
def test_base64_padding_and_encrypted_reasoning_do_not_add_text_cost(tmp_path: Path) -> None:
    small = _image(tmp_path, "small")
    padded = _image(tmp_path, "padded", padding=b"\x00" * 16000)
    client = _client()
    request = LLMRequest(model="gpt-4o", messages=[Msg(role=Role.USER, content=[small])])
    padded_request = LLMRequest(model="gpt-4o", messages=[Msg(role=Role.USER, content=[padded])])
    assert client.estimate_input_tokens(request) == client.estimate_input_tokens(padded_request)
    reasoning = (
        ResponsesMapper()
        .parse_response(
            {
                "status": "completed",
                "output": [{"type": "reasoning", "summary": [], "encrypted_content": "x" * 16000}],
            },
            provider="openai",
        )
        .to_msg()
    )
    with_reasoning = request.model_copy(update={"messages": [*request.messages, reasoning]})
    assert client.estimate_input_tokens(with_reasoning) == client.estimate_input_tokens(request)


@pytest.mark.asyncio
async def test_complete_and_stream_send_same_model_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _image(tmp_path, "model", image_format="WEBP")
    request = LLMRequest(
        model="gpt-4o",
        messages=[
            Msg(role=Role.USER, content=[image]),
            Msg.assistant([ToolUseBlock(id="call-a", name="inspect", input={})]),
            Msg.tool_result(tool_use_id="call-a", content=[image]),
        ],
    )
    final = {
        "id": "response-image",
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "message-image",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "blue", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 500, "output_tokens": 1, "total_tokens": 501},
    }
    seen: list[dict[str, Any]] = []

    class RawStream:
        """提供真实 adapter 关闭接口和一条 completed 事件。"""

        response: RawStream

        def __init__(self) -> None:
            self.response = self
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.completed", "response": final}

        async def aclose(self) -> None:
            self.closed = True

    raw = RawStream()

    async def invoke(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return raw if kwargs.get("stream") else final

    monkeypatch.setattr("iris.providers.responses.litellm.aresponses", invoke)
    response = await _client().complete(request)
    events = [
        event async for event in _client().stream(request.model_copy(update={"stream": True}))
    ]
    expected = ResponsesMapper().format_messages(request.messages)
    assert seen[0]["input"] == seen[1]["input"] == expected
    assert seen[0]["input"][0]["content"] == [_image_part(image)]
    assert seen[0]["input"][2]["output"] == [_image_part(image)]
    assert response.to_msg().text == "blue"
    assert isinstance(events[-1], ModelResponseCompleted)
    assert events[-1].response.input_tokens == 500
    assert raw.closed


def test_missing_model_copy_fails_instead_of_dropping_image(tmp_path: Path) -> None:
    missing = ImageFileRef(
        path=tmp_path / "missing.png", mime_type="image/png", width=32, height=16
    )
    message = Msg(role=Role.USER, content=[ImageBlock(original=missing, model=missing)])
    with pytest.raises(IrisImageError):
        ResponsesMapper().format_messages([message])
    with pytest.raises(IrisProviderError):
        _client().estimate_input_tokens(LLMRequest(model="gpt-4o", messages=[message]))


@pytest.mark.asyncio
async def test_missing_model_copy_fails_complete_and_stream_before_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = ImageFileRef(
        path=tmp_path / "missing.png", mime_type="image/png", width=32, height=16
    )
    request = LLMRequest(
        model="gpt-4o",
        messages=[Msg(role=Role.USER, content=[ImageBlock(original=missing, model=missing)])],
    )

    async def unexpected_sdk_call(**kwargs: Any) -> Any:
        pytest.fail("不可读模型副本不得进入 SDK 调用")

    monkeypatch.setattr("iris.providers.responses.litellm.aresponses", unexpected_sdk_call)
    with pytest.raises(IrisProviderError):
        await _client().complete(request)
    events = [
        event async for event in _client().stream(request.model_copy(update={"stream": True}))
    ]
    assert len(events) == 1
    assert isinstance(events[0], ModelResponseFailed)
