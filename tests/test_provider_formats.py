from __future__ import annotations

from iris.message.message import (
    Conversation,
    Msg,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from iris.providers import (
    AnthropicMessageAdapter,
    MessageAdapter,
    OpenAIMessageAdapter,
)


def test_provider_adapters_extend_base_contract() -> None:
    assert issubclass(OpenAIMessageAdapter, MessageAdapter)
    assert issubclass(AnthropicMessageAdapter, MessageAdapter)


def test_openai_provider_serializes_messages_and_parses_responses() -> None:
    provider = OpenAIMessageAdapter(api_style="responses")
    message = Msg.assistant(
        [
            TextBlock(text="I will call a tool."),
            ToolUseBlock(id="call_123", name="lookup", input={"query": "iris"}),
        ]
    )

    assert provider.to_provider(message) == [
        {"role": "assistant", "content": "I will call a tool."},
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "lookup",
            "arguments": '{"query":"iris"}',
        },
    ]

    parsed = provider.from_provider(
        {
            "model": "gpt-test",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done"}],
                },
                {
                    "type": "function_call",
                    "id": "call_456",
                    "name": "save",
                    "arguments": '{"ok":true}',
                },
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
            },
        }
    )

    assert parsed.text == "Done"
    assert parsed.tool_calls == [
        ToolUseBlock(id="call_456", name="save", input={"ok": True})
    ]
    assert parsed.metadata["model"] == "gpt-test"


def test_anthropic_provider_serializes_messages_and_parses_responses() -> None:
    provider = AnthropicMessageAdapter()
    message = Msg(
        role=Role.USER,
        content=[
            TextBlock(text="Here is the result."),
            ToolResultBlock(tool_use_id="tool_123", content="OK", is_error=False),
        ],
    )

    assert provider.to_provider(message) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here is the result."},
            {
                "type": "tool_result",
                "tool_use_id": "tool_123",
                "content": "OK",
                "is_error": False,
            },
        ],
    }

    parsed = provider.from_provider(
        {
            "model": "claude-test",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Calling tool"},
                {
                    "type": "tool_use",
                    "id": "tool_456",
                    "name": "lookup",
                    "input": {"query": "iris"},
                },
            ],
            "usage": {"input_tokens": 2, "output_tokens": 4},
        }
    )

    assert parsed.text == "Calling tool"
    assert parsed.tool_calls == [
        ToolUseBlock(id="tool_456", name="lookup", input={"query": "iris"})
    ]
    assert parsed.metadata["stop_reason"] == "tool_use"


def test_conversation_uses_provider_formats() -> None:
    conversation = Conversation(
        messages=[
            Msg.system("You are concise."),
            Msg.user("Hello"),
            Msg.assistant(
                [ToolUseBlock(id="call_123", name="lookup", input={"q": "x"})]
            ),
            Msg.tool_result(tool_use_id="call_123", content="result"),
        ]
    )

    assert conversation.to_openai() == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hello"},
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "lookup",
            "arguments": '{"q":"x"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "result",
        },
    ]
    assert conversation.to_anthropic() == {
        "system": "You are concise.",
        "messages": [
            {"role": "user", "content": {"type": "text", "text": "Hello"}},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "lookup",
                        "input": {"q": "x"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "result",
                        "is_error": False,
                    }
                ],
            },
        ],
    }


def test_msg_from_dict_builds_typed_content_blocks() -> None:
    msg = Msg.from_dict(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Need a tool."},
                {
                    "type": "tool_use",
                    "id": "call_789",
                    "name": "lookup",
                    "input": {"query": "iris"},
                },
            ],
            "sender": "agent",
            "metadata": {"trace_id": "trace_123"},
        }
    )

    assert msg.role == Role.ASSISTANT
    assert msg.sender == "agent"
    assert msg.text == "Need a tool."
    assert msg.tool_calls == [
        ToolUseBlock(id="call_789", name="lookup", input={"query": "iris"})
    ]
    assert msg.metadata == {"trace_id": "trace_123"}
