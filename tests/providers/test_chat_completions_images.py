"""Chat 图片请求投影、调用和视觉预算。"""

import base64
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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
    ToolResultBlock,
    ToolUseBlock,
)
from iris.providers import ProviderClient
from iris.providers.chat_completions import ChatCompletionsAdapter, ChatCompletionsMapper


def _image(
    tmp_path: Path,
    name: str,
    *,
    size: tuple[int, int] = (40, 20),
    format: str = "PNG",
    padding: str = "",
) -> ImageBlock:
    path = tmp_path / name
    info = PngInfo()
    info.add_text("snapshot", name)
    info.add_text("description", padding)
    with Image.new("RGB", size, "red") as image:
        image.save(path, format=format, pnginfo=info)
    model = ImageFileRef(path=path, mime_type=Image.MIME[format], width=size[0], height=size[1])
    original = ImageFileRef(
        path=tmp_path / "unavailable-original.png", mime_type="image/png", width=3000, height=2000
    )
    return ImageBlock(original=original, model=model, name=name)


def _url(image: ImageBlock) -> str:
    encoded = base64.b64encode(image.model.path.read_bytes()).decode()
    return f"data:{image.model.mime_type};base64,{encoded}"


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_user_images_keep_mixed_order_and_actual_model_encoding(
    tmp_path: Path, format: str
) -> None:
    first = _image(tmp_path, "first.data", format=format)
    second = _image(tmp_path, "second.png")
    message = Msg(
        role=Role.USER,
        sender="user",
        content=[TextBlock(text="之前"), first, TextBlock(text="之间"), second],
    )
    before = message.model_dump_json()

    wire = ChatCompletionsMapper().format_messages([message])

    assert wire == [
        {
            "role": "user",
            "name": "user",
            "content": [
                {"type": "text", "text": "之前"},
                {"type": "image_url", "image_url": {"url": _url(first), "detail": "high"}},
                {"type": "text", "text": "之间"},
                {"type": "image_url", "image_url": {"url": _url(second), "detail": "high"}},
            ],
        }
    ]
    assert message.model_dump_json() == before


def test_pure_image_user_and_text_only_messages(tmp_path: Path) -> None:
    image = _image(tmp_path, "image.png")
    wire = ChatCompletionsMapper().format_messages(
        [
            Msg.system("规则"),
            Msg(role=Role.USER, content=[image]),
            Msg.user("文字"),
            Msg.assistant("回答"),
        ]
    )
    assert wire == [
        {"role": "system", "content": "规则"},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _url(image), "detail": "high"}}],
        },
        {"role": "user", "name": "user", "content": "文字"},
        {"role": "assistant", "content": "回答"},
    ]


def test_tool_images_wait_for_whole_group_before_steer_and_preserve_result_order(
    tmp_path: Path,
) -> None:
    first = _image(tmp_path, "first.png")
    second = _image(tmp_path, "second.png")
    third = _image(tmp_path, "third.png")
    messages = [
        Msg.assistant([ToolUseBlock(id="a", name="plot_a"), ToolUseBlock(id="b", name="plot_b")]),
        Msg.tool_result(tool_use_id="b", content=[TextBlock(text="B结果"), first]),
        Msg.tool_result(
            tool_use_id="a", content=[second, TextBlock(text="A结果"), third], name="plot_a"
        ),
        Msg.user("补充要求", sender="steer"),
    ]
    before = [message.model_dump_json() for message in messages]

    wire = ChatCompletionsMapper().format_messages(messages)

    assert [item["role"] for item in wire] == ["assistant", "tool", "tool", "user", "user"]
    assert [wire[1]["tool_call_id"], wire[2]["tool_call_id"]] == ["b", "a"]
    assert "B结果" in wire[1]["content"] and "plot_b" in wire[1]["content"]
    assert "A结果" in wire[2]["content"] and "plot_a" in wire[2]["content"]
    projected = wire[3]["content"]
    assert [part["type"] for part in projected] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "image_url",
    ]
    assert "call_id=b" in projected[0]["text"] and "plot_b" in projected[0]["text"]
    assert "call_id=a" in projected[2]["text"] and "plot_a" in projected[2]["text"]
    assert [part["image_url"]["url"] for part in projected if part["type"] == "image_url"] == [
        _url(first),
        _url(second),
        _url(third),
    ]
    assert wire[4] == {"role": "user", "name": "steer", "content": "补充要求"}
    assert [message.model_dump_json() for message in messages] == before


def test_two_tool_groups_and_pure_image_receipt(tmp_path: Path) -> None:
    image = _image(tmp_path, "image.png")
    messages = [
        Msg.assistant([ToolUseBlock(id="a", name="plot"), ToolUseBlock(id="b", name="read")]),
        Msg(
            role=Role.USER,
            content=[
                ToolResultBlock(tool_use_id="a", content=[image]),
                ToolResultBlock(tool_use_id="b", content=[TextBlock(text="done")]),
            ],
        ),
        Msg.assistant([ToolUseBlock(id="c", name="plot_again")]),
        Msg.tool_result(tool_use_id="c", content=[image]),
        Msg.assistant([ToolUseBlock(id="d", name="read")]),
        Msg.tool_result(tool_use_id="d", content="text only"),
    ]

    wire = ChatCompletionsMapper().format_messages(messages)

    assert [item["role"] for item in wire] == [
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
    ]
    assert "plot" in wire[1]["content"] and "call_id=a" in wire[1]["content"]
    assert wire[2]["content"] == "done"
    assert "call_id=a" in wire[3]["content"][0]["text"]
    assert "call_id=c" in wire[6]["content"][0]["text"]
    assert "plot_again" in wire[6]["content"][0]["text"]
    assert wire[-1]["content"] == "text only"


@pytest.mark.asyncio
async def test_complete_stream_and_count_share_image_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    image = _image(tmp_path, "image.png")
    request = LLMRequest(
        model="gpt-4o",
        messages=[
            Msg(role=Role.USER, content=[image]),
            Msg.assistant([ToolUseBlock(id="a", name="plot")]),
            Msg.tool_result(tool_use_id="a", content=[image]),
        ],
    )
    captured: list[dict[str, Any]] = []

    async def chunks() -> AsyncIterator[dict[str, Any]]:
        yield {
            "id": "chat-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "看到了"}, "finish_reason": "stop"}],
        }

    async def invoke(**kwargs: Any) -> Any:
        captured.append(kwargs)
        if kwargs.get("stream"):
            return chunks()
        return {
            "id": "chat-1",
            "model": "gpt-4o",
            "choices": [{"message": {"content": "看到了"}, "finish_reason": "stop"}],
        }

    monkeypatch.setattr(litellm, "acompletion", invoke)
    client = ProviderClient(provider="openai", api_key="test", api_style="chat_completions")
    response = await client.complete(request)
    events = [event async for event in client.stream(request.model_copy(update={"stream": True}))]

    assert response.to_msg().text == "看到了"
    assert isinstance(events[-1], ModelResponseCompleted)
    assert events[-1].response.to_msg().text == "看到了"
    counted = ChatCompletionsAdapter().token_count_projection(request, transport="openai")
    assert captured[0]["messages"] == captured[1]["messages"] == counted["messages"]
    assert captured[0]["messages"][0]["content"][0]["image_url"]["url"] == _url(image)
    assert captured[0]["messages"][-1]["content"][1]["image_url"]["url"] == _url(image)


@pytest.mark.asyncio
async def test_missing_model_image_fails_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    ref = ImageFileRef(
        path=tmp_path / "missing-model.png", mime_type="image/png", width=40, height=20
    )
    request = LLMRequest(
        model="gpt-4o",
        messages=[Msg(role=Role.USER, content=[ImageBlock(original=ref, model=ref)])],
    )

    async def unexpected(**kwargs: Any) -> None:
        pytest.fail("图片副本缺失时不能发起 provider 调用")

    monkeypatch.setattr(litellm, "acompletion", unexpected)
    with pytest.raises(IrisImageError):
        ChatCompletionsMapper().format_messages(request.messages)
    client = ProviderClient(provider="openai", api_key="test", api_style="chat_completions")
    with pytest.raises(IrisProviderError):
        await client.complete(request)
    with pytest.raises(IrisProviderError):
        client.estimate_input_tokens(request)
    events = [event async for event in client.stream(request.model_copy(update={"stream": True}))]
    assert len(events) == 1
    assert isinstance(events[0], ModelResponseFailed)


def test_visual_budget_uses_dimensions_and_images_not_base64_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    # 只替换文字 tokenizer，实际 LiteLLM 图片解析、缩放和 tile 计量仍执行，避免下载词表。
    counter = importlib.import_module("litellm.litellm_core_utils.token_counter")
    monkeypatch.setattr(counter, "_get_count_function", lambda model, custom_tokenizer: len)
    small = _image(tmp_path, "small.png", size=(40, 20))
    padded = _image(tmp_path, "padded.png", size=(40, 20), padding="x" * 80_000)
    large = _image(tmp_path, "large.png", size=(1600, 1600))
    client = ProviderClient(provider="openai", api_key="test", api_style="chat_completions")

    def estimate(images: list[ImageBlock]) -> int:
        return client.estimate_input_tokens(
            LLMRequest(model="gpt-4o", messages=[Msg(role=Role.USER, content=images)])
        )

    assert padded.model.path.stat().st_size > small.model.path.stat().st_size + 79_000
    assert estimate([small]) == estimate([padded])
    assert estimate([large]) - estimate([small]) == 510
    assert estimate([small, small]) - estimate([small]) == 255

    request = LLMRequest(
        model="gpt-4o",
        messages=[
            Msg.assistant([ToolUseBlock(id="a", name="plot")]),
            Msg.tool_result(tool_use_id="a", content=[small]),
        ],
    )
    wire = ChatCompletionsMapper().format_messages(request.messages)
    assert (
        sum(
            part["type"] == "image_url"
            for item in wire
            if isinstance(item["content"], list)
            for part in item["content"]
        )
        == 1
    )
    text_only_projection = [
        {**item, "content": [part for part in item["content"] if part["type"] == "text"]}
        if isinstance(item["content"], list)
        else item
        for item in wire
    ]
    assert (
        client.estimate_input_tokens(request)
        - counter.token_counter(model="gpt-4o", messages=text_only_projection)
        == 255
    )
