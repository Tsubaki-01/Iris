from iris.message import Conversation, LLMRequest, Msg, ToolUseBlock
from iris.providers import AnthropicMessageAdapter, OpenAIMessageAdapter


def test_openai_adapter_defaults_to_chat_completions_payload() -> None:
    request = LLMRequest(
        model="gpt-4o",
        messages=[Msg.system("你是助手"), Msg.user("你好")],
        temperature=0.1,
        max_tokens=64,
    )

    payload = OpenAIMessageAdapter().to_provider_request(request)

    assert payload == {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好", "name": "user"},
        ],
        "temperature": 0.1,
        "max_tokens": 64,
    }


def test_openai_adapter_supports_explicit_responses_payload() -> None:
    request = LLMRequest(
        model="gpt-4o",
        messages=[Msg.user("你好")],
        provider_options={"api_style": "responses"},
    )

    payload = OpenAIMessageAdapter().to_provider_request(request)

    assert payload == {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "你好", "name": "user"}],
    }


def test_openai_responses_adapter_formats_tool_result_as_function_call_output() -> None:
    request = LLMRequest(
        model="gpt-4o",
        messages=[Msg.tool_result(tool_use_id="call_1", content="完成")],
        provider_options={"api_style": "responses"},
    )

    assert OpenAIMessageAdapter().to_provider_request(request)["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "完成"}
    ]


def test_openai_adapter_parses_chat_completion_response() -> None:
    response = OpenAIMessageAdapter().from_provider_response(
        {
            "id": "chatcmpl_1",
            "model": "gpt-4o",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "你好",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query":"iris"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
            },
        }
    )

    assert response.provider == "openai"
    assert response.id == "chatcmpl_1"
    assert response.model == "gpt-4o"
    assert response.finish_reason == "stop"
    assert response.input_tokens == 3
    assert response.output_tokens == 5
    assert response.total_tokens == 8
    assert response.to_msg().text == "你好"
    assert response.to_msg().tool_calls == [
        ToolUseBlock(id="call_1", name="search", input={"query": "iris"})
    ]


def test_openai_adapter_parses_responses_response() -> None:
    response = OpenAIMessageAdapter().from_provider_response(
        {
            "id": "resp_1",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "你好"}],
                }
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
            },
        }
    )

    assert response.id == "resp_1"
    assert response.to_msg().text == "你好"
    assert response.input_tokens == 3
    assert response.output_tokens == 5


def test_anthropic_adapter_moves_system_prompt_to_top_level() -> None:
    conversation = Conversation(messages=[Msg.system("你是助手"), Msg.user("你好")])
    request = conversation.to_llm_request("claude-sonnet-4-5", max_tokens=128)

    payload = AnthropicMessageAdapter().to_provider_request(request)

    assert payload == {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "你好"}]},
        ],
        "system": "你是助手",
        "max_tokens": 128,
    }


def test_anthropic_adapter_parses_text_and_tool_use_response() -> None:
    response = AnthropicMessageAdapter().from_provider_response(
        {
            "id": "msg_1",
            "model": "claude-sonnet-4-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "需要查询"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"query": "iris"},
                },
            ],
            "usage": {"input_tokens": 7, "output_tokens": 11},
        }
    )

    assert response.provider == "anthropic"
    assert response.finish_reason == "tool_use"
    assert response.total_tokens == 18
    assert response.to_msg().text == "需要查询"
    assert response.to_msg().tool_calls == [
        ToolUseBlock(id="toolu_1", name="search", input={"query": "iris"})
    ]


def test_adapter_formats_tool_results_for_each_provider() -> None:
    request = LLMRequest(
        model="gpt-4o",
        messages=[Msg.tool_result(tool_use_id="call_1", content="完成")],
    )

    assert OpenAIMessageAdapter().to_provider_request(request)["messages"] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "完成"}
    ]
    assert AnthropicMessageAdapter().to_provider_request(request)["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "完成",
                    "is_error": False,
                }
            ],
        }
    ]
