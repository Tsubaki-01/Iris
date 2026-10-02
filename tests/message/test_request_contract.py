"""模型请求以统一逻辑契约解析外部输入。"""

from typing import Any

import pytest
from pydantic import ValidationError

from iris.lifecycle import RuntimeExecutionOptions
from iris.message import LLMRequest, ToolSpec


def test_request_parses_logical_tools_and_output_options() -> None:
    """输入只描述工具与输出意图，不携带协议包装。"""
    request = LLMRequest(
        model="test",
        tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        tool_choice={"name": "lookup"},
        response_format={"name": "answer", "schema": {"type": "object"}, "strict": True},
    )
    assert isinstance(request.tools[0], ToolSpec)
    assert request.tools[0].name == "lookup"
    assert request.tools[0].strict is False
    assert request.tool_choice == {"name": "lookup"}
    assert request.response_format["strict"] is True


@pytest.mark.parametrize(
    "options",
    [
        {"tool_choice": {"type": "function", "name": "lookup"}},
        {"tool_choice": {"function": {"name": "lookup"}}},
        {"response_format": {"type": "json_object"}},
        {"response_format": {"type": "json_schema", "json_schema": {"name": "answer"}}},
        {"provider_options": {"api_style": "responses"}},
    ],
)
def test_request_and_runtime_overrides_reject_protocol_wrappers(options: dict[str, Any]) -> None:
    """公开请求和持久化选项的首次解析均使用当前逻辑契约。"""
    with pytest.raises(ValidationError):
        LLMRequest(model="test", **options)
    with pytest.raises(ValidationError):
        RuntimeExecutionOptions(request_options=options)


def test_runtime_overrides_keep_logical_choices_and_roundtrip() -> None:
    """覆盖项直接交给 runtime，恢复后仍维持相同逻辑形状。"""
    values = {
        "tool_choice": {"name": "lookup"},
        "response_format": "json_object",
        "provider_options": {"reasoning_effort": "low"},
    }
    options = RuntimeExecutionOptions(request_options=values)
    assert options.request_options == values
    assert RuntimeExecutionOptions.model_validate_json(options.model_dump_json()) == options
