"""固定 GenAI 内容格式、模型可见结果和有界序列化。"""

import json
from pathlib import Path
from typing import Any

import pytest

from iris.message import (
    ImageBlock,
    ImageFileRef,
    LLMRequest,
    LLMResponse,
    Msg,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
)
from iris.observability.content import request_attributes, response_attributes, tool_attributes
from iris.tools import ToolErrorInfo, ToolResult


def _image() -> ImageBlock:
    original = ImageFileRef(
        path=Path.cwd() / "original.png", mime_type="image/png", width=100, height=200
    )
    model = ImageFileRef(
        path=Path.cwd() / "model.webp", mime_type="image/webp", width=50, height=100
    )
    return ImageBlock(original=original, model=model, name="示例")


def _image_part(image: ImageBlock) -> dict[str, str]:
    return {
        "type": "uri",
        "modality": "image",
        "uri": image.model.path.as_uri(),
        "mime_type": "image/webp",
    }


def test_request_preserves_message_order_and_function_schema() -> None:
    """system 属于请求消息历史；不另建或重复独立 instructions。"""
    image = _image()
    request = LLMRequest(
        model="model",
        messages=[
            Msg.system("规则一"),
            Msg.user([TextBlock(text="看图"), image]),
            Msg.system("规则二"),
            Msg.assistant(
                [
                    TextBlock(text="读取内容"),
                    ToolUseBlock(id="call-1", name="read_file", input={"path": "说明.md"}),
                ]
            ),
            Msg.tool_result(
                tool_use_id="call-1", content=[TextBlock(text="正文"), image], name="read_file"
            ),
        ],
        tools=[
            ToolSpec(
                name="read_file",
                description="读取文件",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                strict=True,
            )
        ],
        metadata={"do_not_traverse": object()},
        provider_options={"do_not_traverse": object()},
    )

    attributes = request_attributes(request, 65536)

    assert set(attributes) == {"gen_ai.input.messages", "gen_ai.tool.definitions"}
    assert json.loads(str(attributes["gen_ai.input.messages"])) == [
        {"role": "system", "parts": [{"type": "text", "content": "规则一"}]},
        {"role": "user", "parts": [{"type": "text", "content": "看图"}, _image_part(image)]},
        {"role": "system", "parts": [{"type": "text", "content": "规则二"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "读取内容"},
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": {"path": "说明.md"},
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": {
                        "parts": [{"type": "text", "content": "正文"}, _image_part(image)]
                    },
                }
            ],
        },
    ]
    assert json.loads(str(attributes["gen_ai.tool.definitions"])) == [
        {
            "type": "function",
            "name": "read_file",
            "description": "读取文件",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]


@pytest.mark.parametrize("provider", ["chat", "responses"])
def test_response_uses_typed_content_and_reasoning(provider: str) -> None:
    response = LLMResponse(
        provider=provider,
        content=[
            TextBlock(text="第一段"),
            ToolUseBlock(id="call-1", name="lookup", input={"query": "Iris"}),
            TextBlock(text="第二段"),
        ],
        reasoning="先查资料",
        finish_reason="tool_calls",
        metadata={"responses": object(), "headers": object()},
    )

    attributes = response_attributes(response, 65536)

    assert set(attributes) == {"gen_ai.output.messages"}
    assert json.loads(str(attributes["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "先查资料"},
                {"type": "text", "content": "第一段"},
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": {"query": "Iris"},
                },
                {"type": "text", "content": "第二段"},
            ],
        }
    ]


@pytest.mark.parametrize("structured_error", [False, True])
def test_tool_result_uses_final_model_blocks_without_reading_image(
    structured_error: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _image()
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="read_file",
        content=[TextBlock(text="已有正文和 artifact.txt 引用"), image],
        is_error=structured_error,
        error=ToolErrorInfo(code="FAILED", message="最终错误") if structured_error else None,
        hook_feedback=("检查图片", "继续处理"),
        metadata={"do_not_traverse": object()},
        data={"do_not_traverse": object()},
    )

    def refuse_io(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("内容投影不能读取图片")

    monkeypatch.setattr(Path, "open", refuse_io)
    monkeypatch.setattr(Path, "read_bytes", refuse_io)
    attributes = tool_attributes({"path": "输入.txt"}, result, 65536)

    assert json.loads(str(attributes["gen_ai.tool.call.arguments"])) == {"path": "输入.txt"}
    assert json.loads(str(attributes["gen_ai.tool.call.result"])) == {
        "parts": [
            {
                "type": "text",
                "content": "Error[FAILED]: 最终错误"
                if structured_error
                else "已有正文和 artifact.txt 引用",
            },
            _image_part(image),
            {"type": "text", "content": "[Hook feedback]\n检查图片"},
            {"type": "text", "content": "[Hook feedback]\n继续处理"},
        ]
    }
    assert result.content[0] == TextBlock(text="已有正文和 artifact.txt 引用")


def test_truncation_omits_standard_fields_and_names_full_attribute_keys() -> None:
    result = ToolResult(tool_use_id="call-1", tool_name="echo", content=[TextBlock(text="内容")])
    full = tool_attributes({"value": "长" * 20}, result, 65536)

    limited = tool_attributes({"value": "长" * 20}, result, 8)

    keys = ["gen_ai.tool.call.arguments", "gen_ai.tool.call.result"]
    assert limited == {
        **{f"iris.content.{key}.preview": str(full[key])[:8] for key in keys},
        "iris.content.truncated_fields": keys,
    }


def test_limit_applies_to_each_complete_json_and_accepts_exact_length() -> None:
    request = LLMRequest(model="model", messages=[Msg.user("中文")])
    full = request_attributes(request, 65536)
    encoded = str(full["gen_ai.input.messages"])

    assert "中文" in encoded
    assert request_attributes(request, len(encoded)) == full
    assert request_attributes(request, len(encoded) - 1) == {
        "iris.content.gen_ai.input.messages.preview": encoded[:-1],
        "iris.content.truncated_fields": ["gen_ai.input.messages"],
    }


def test_empty_request_and_response_are_valid_message_arrays() -> None:
    assert request_attributes(LLMRequest(model="model"), 65536) == {"gen_ai.input.messages": "[]"}
    assert json.loads(
        str(response_attributes(LLMResponse(provider="test"), 65536)["gen_ai.output.messages"])
    ) == [{"role": "assistant", "parts": []}]
