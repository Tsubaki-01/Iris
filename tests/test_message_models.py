from iris.message import (
    Conversation,
    LLMRequest,
    LLMResponse,
    Msg,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_msg_text_blocks_and_tool_blocks_are_accessible() -> None:
    tool_call = ToolUseBlock(id="call_1", name="search", input={"query": "iris"})
    message = Msg.assistant([TextBlock(text="需要查询"), tool_call])

    assert message.role == Role.ASSISTANT
    assert message.text == "需要查询"
    assert message.blocks == [TextBlock(text="需要查询"), tool_call]
    assert message.tool_calls == [tool_call]
    assert message.has_tool_calls is True


def test_tool_result_keeps_existing_user_role_compatibility() -> None:
    message = Msg.tool_result(tool_use_id="call_1", content="完成", is_error=False)

    assert message.role == Role.USER
    assert message.tool_results == [
        ToolResultBlock(tool_use_id="call_1", content="完成", is_error=False)
    ]


def test_conversation_builds_llm_request_with_message_order() -> None:
    conversation = Conversation()
    conversation.add(Msg.system("你是助手"))
    conversation.add(Msg.user("你好"))

    request = conversation.to_llm_request("gpt-4o", temperature=0.2)

    assert isinstance(request, LLMRequest)
    assert request.model == "gpt-4o"
    assert request.temperature == 0.2
    assert request.messages == conversation.messages
    assert request.system_prompt() == "你是助手"
    assert request.non_system_messages() == [conversation.messages[1]]


def test_llm_response_converts_to_assistant_msg_with_usage_metadata() -> None:
    response = LLMResponse(
        provider="openai",
        id="resp_1",
        model="gpt-4o",
        content=[TextBlock(text="你好")],
        finish_reason="stop",
        input_tokens=3,
        output_tokens=5,
        total_tokens=8,
        reasoning="简短推理",
        metadata={"request_id": "req_1"},
    )

    message = response.to_msg()

    assert message.role == Role.ASSISTANT
    assert message.text == "你好"
    assert message.metadata["provider"] == "openai"
    assert message.metadata["id"] == "resp_1"
    assert message.metadata["model"] == "gpt-4o"
    assert message.metadata["finish_reason"] == "stop"
    assert message.metadata["usage"] == {
        "input_tokens": 3,
        "output_tokens": 5,
        "total_tokens": 8,
    }
    assert message.metadata["reasoning"] == "简短推理"
    assert message.metadata["request_id"] == "req_1"
