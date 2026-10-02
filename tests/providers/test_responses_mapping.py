"""Responses 正文权威、有序重放和本地计量投影。"""

import json
from copy import deepcopy

import pytest

from iris.exceptions import IrisProviderError
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.providers.responses import ResponsesMapper


def _completed() -> dict:
    return {
        "id": "resp-test",
        "model": "deepseek-flash",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs-1", "summary": [], "encrypted_content": "opaque"},
            {
                "type": "message",
                "id": "msg-1",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "检查", "annotations": []}],
            },
            {
                "type": "function_call",
                "id": "fc-item",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"q":"old"}',
            },
            {
                "type": "reasoning",
                "id": "rs-2",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "可见推理"}],
            },
            {
                "type": "message",
                "id": "msg-2",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "等待结果", "annotations": []}],
            },
        ],
        "usage": {
            "input_tokens": 13,
            "output_tokens": 7,
            "total_tokens": 20,
            "input_tokens_details": {"cached_tokens": 8},
            "output_tokens_details": {"reasoning_tokens": 4},
        },
    }


def test_responses_roundtrip_preserves_order_but_uses_current_typed_content() -> None:
    mapper = ResponsesMapper()
    response = mapper.parse_response(_completed(), provider="deepseek")
    assert response.finish_reason == "tool_calls"
    assert (response.input_tokens, response.output_tokens, response.total_tokens) == (13, 7, 20)
    assert response.reasoning == "可见推理"
    message = Msg.model_validate_json(response.to_msg().model_dump_json())
    assert message.tool_calls[0].id == "call-1"
    assert message.metadata["responses"]["status"] == "completed"
    metadata_text = json.dumps(message.metadata["responses"], ensure_ascii=False)
    assert '"arguments"' not in metadata_text and '"检查"' not in metadata_text
    message.content[0] = TextBlock(text="更新后的正文")
    message.content[1] = ToolUseBlock(id="call-1", name="new_lookup", input={"q": "new"})
    items = mapper.format_messages([message, Msg.tool_result(tool_use_id="call-1", content="答案")])
    assert [item["type"] for item in items] == [
        "reasoning",
        "message",
        "function_call",
        "reasoning",
        "message",
        "function_call_output",
    ]
    assert items[0] == _completed()["output"][0]
    assert items[1]["phase"] == "commentary"
    assert items[1]["content"] == [
        {"type": "output_text", "text": "更新后的正文", "annotations": []}
    ]
    assert items[2] == {
        "type": "function_call",
        "id": "fc-item",
        "call_id": "call-1",
        "name": "new_lookup",
        "arguments": '{"q":"new"}',
    }
    assert items[-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": [{"type": "input_text", "text": "答案"}],
    }
    assert mapper.format_messages([Msg.user("新窗口")]) == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "新窗口"}]}
    ]


def test_responses_plain_assistant_uses_typed_order_without_native_metadata() -> None:
    message = Msg.assistant(
        [
            TextBlock(text="前文"),
            ToolUseBlock(id="c", name="lookup", input={}),
            TextBlock(text="后文"),
        ]
    )
    items = ResponsesMapper().format_messages([message])
    assert [item["type"] for item in items] == ["message", "function_call", "message"]
    assert items[1]["call_id"] == "c" and "id" not in items[1]


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress", None])
def test_responses_failed_status_preserves_known_usage_without_partial_tools(status: str) -> None:
    data = _completed()
    data.update(status=status, incomplete_details={"reason": "max_output_tokens"})
    with pytest.raises(IrisProviderError) as captured:
        ResponsesMapper().parse_response(data, provider="deepseek")
    assert captured.value.context["usage"] == {
        "input_tokens": 13,
        "output_tokens": 7,
        "total_tokens": 20,
    }
    assert captured.value.context["reason"] == {"reason": "max_output_tokens"}


@pytest.mark.parametrize("arguments", ['{"unfinished":', "[]"])
def test_responses_malformed_completed_tool_is_failure_with_usage(arguments: str) -> None:
    data = _completed()
    data["output"][2]["arguments"] = arguments
    with pytest.raises(IrisProviderError) as captured:
        ResponsesMapper().parse_response(data, provider="openai")
    assert captured.value.context["usage"]["total_tokens"] == 20


def test_responses_text_only_completed_maps_to_summary_stop() -> None:
    from iris.runtime._compaction_summary import consume_summary_response

    data = _completed()
    data["output"] = [data["output"][-1]]
    response = ResponsesMapper().parse_response(data, provider="openai")
    assert response.finish_reason == "stop"
    assert consume_summary_response(response) == "等待结果"


def test_responses_count_projection_uses_actual_input_without_encrypted_content() -> None:
    mapper = ResponsesMapper()
    response = mapper.parse_response(_completed(), provider="openai")
    items = mapper.format_messages(
        [response.to_msg(), Msg.tool_result(tool_use_id="call-1", content="结果")]
    )
    original = deepcopy(items)
    messages = mapper.token_count_messages(items)
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "opaque" not in serialized and "encrypted_content" not in serialized
    assert "可见推理" in serialized and "结果" in serialized
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"q":"old"}'
    assert items == original


def test_plain_assistant_matches_sdk_easy_input_schema() -> None:
    """普通历史不冒充需要 id/status 的原生 output message。"""
    from openai.types.responses.easy_input_message_param import EasyInputMessageParam
    from pydantic import TypeAdapter

    item = ResponsesMapper().format_messages([Msg.assistant("hello")])[0]
    validated = TypeAdapter(EasyInputMessageParam).validate_python(item)
    assert validated["role"] == "assistant"
    assert "id" not in item


def test_native_assistant_replay_matches_sdk_output_schema() -> None:
    """原生回放保留 SDK 必需的 status 与 annotations。"""
    from openai.types.responses.response_output_message import ResponseOutputMessage

    data = _completed()
    for item in data["output"]:
        if item["type"] == "message":
            item["status"] = "completed"
    mapper = ResponsesMapper()
    restored = Msg.model_validate_json(
        mapper.parse_response(data, provider="openai").to_msg().model_dump_json()
    )
    items = mapper.format_messages([restored])
    for item in items:
        if item["type"] == "message":
            validated = ResponseOutputMessage.model_validate(item)
            assert validated.status == "completed"
            assert validated.content[0].annotations == []
