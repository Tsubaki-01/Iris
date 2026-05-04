from iris.message import Conversation, Msg
from iris.providers import OpenAIMessageAdapter


def test_conversation_only_builds_provider_neutral_llm_request() -> None:
    conversation = Conversation(messages=[Msg.user("你好")])

    request = conversation.to_llm_request("gpt-4o")

    assert request.model == "gpt-4o"
    assert request.messages == conversation.messages


def test_openai_payload_is_created_by_provider_adapter() -> None:
    conversation = Conversation(messages=[Msg.user("你好")])
    request = conversation.to_llm_request("gpt-4o")

    payload = OpenAIMessageAdapter().to_provider_request(request)

    assert payload == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "你好", "name": "user"}],
    }
