"""同一 runtime 工具、持久化恢复与摘要契约经过真实双协议 adapter。"""

import base64
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import pytest
from PIL import Image

from iris.agents import CompactionConfig
from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import ImageBlock, LLMRequest, Msg, TextBlock, ToolSpec
from iris.providers import ProviderClient
from iris.providers.chat_completions import ChatCompletionsAdapter
from iris.providers.responses import ResponsesAdapter
from iris.runtime._compaction_summary import (
    consume_summary_response,
    next_summary_batch,
    serialize_history,
)
from iris.store import SQLiteStore
from iris.tools import ToolCapability, ToolRegistry, ToolResult
from iris.utils import TemplateRenderer

from .fakes import RecordingPublisher, build_runtime

ApiStyle = Literal["responses", "chat_completions"]


def _raw_response(style: ApiStyle, *, call: bool, text: str = "done") -> dict[str, Any]:
    if style == "responses":
        output = (
            {
                "type": "function_call",
                "id": "fc-save",
                "call_id": "call-save",
                "name": "save",
                "arguments": '{"value":"saved"}',
                "status": "completed",
            }
            if call
            else {
                "type": "message",
                "id": "msg-done",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
        return {
            "id": "resp-test",
            "model": "gpt-4o",
            "status": "completed",
            "output": (
                [
                    {
                        "type": "reasoning",
                        "id": "rs-saved",
                        "summary": [],
                        "encrypted_content": "saved-reasoning",
                    }
                ]
                if call
                else []
            )
            + [output],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }
    message = {"role": "assistant", "content": None if call else text}
    if call:
        message["reasoning_content"] = "saved reasoning"
        message["tool_calls"] = [
            {
                "id": "call-save",
                "type": "function",
                "function": {"name": "save", "arguments": '{"value":"saved"}'},
            }
        ]
    return {
        "id": "chat-test",
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if call else "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


async def _stream(style: ApiStyle, response: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    if style == "responses":
        yield {"type": "response.completed", "response": response}
    else:
        choice = response["choices"][0]
        delta = choice["message"]
        if "tool_calls" in delta:
            delta["tool_calls"][0]["index"] = 0
        yield {
            "id": response["id"],
            "model": response["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}],
        }
        yield {
            "id": response["id"],
            "model": response["model"],
            "choices": [],
            "usage": response["usage"],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["responses", "chat_completions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_protocol_tool_loop_resumes_from_sqlite_with_same_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, style: ApiStyle, stream: bool
) -> None:
    import litellm

    requests: list[dict[str, Any]] = []

    async def generate(self: object, kwargs: dict[str, Any]) -> Any:
        requests.append(kwargs)
        response = _raw_response(style, call=len(requests) == 1)
        return _stream(style, response) if kwargs.get("stream") else response

    async def wrong_protocol(self: object, kwargs: dict[str, Any]) -> None:
        pytest.fail("selected protocol must not switch")

    monkeypatch.setattr(
        ResponsesAdapter, "invoke", generate if style == "responses" else wrong_protocol
    )
    monkeypatch.setattr(
        ChatCompletionsAdapter,
        "invoke",
        generate if style == "chat_completions" else wrong_protocol,
    )
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 1)
    client = ProviderClient(provider="openai", api_key="test", api_style=style)
    writes: list[str] = []

    def save(value: str) -> ToolResult:
        writes.append(value)
        return ToolResult(
            tool_use_id="",
            tool_name="save",
            content=[TextBlock(text=value), tool_image, TextBlock(text="after image")],
        )

    registry = ToolRegistry()
    registry.register_function(save, description="保存", capabilities={ToolCapability.WRITE})
    database = tmp_path / "protocol.db"
    publisher = RecordingPublisher() if stream else None
    first_runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=client),
        store=SQLiteStore(database),
        live_publisher=publisher,
    )
    source = tmp_path / "source.png"
    with Image.new("RGB", (20, 10), "red") as pixels:
        pixels.save(source)
    user_image = await first_runner.import_image(source, session_id="default", name="user")
    buffer = BytesIO()
    with Image.new("RGB", (20, 10), "blue") as pixels:
        pixels.save(buffer, format="PNG")
    tool_image = await first_runner.import_image(
        buffer.getvalue(), session_id="default", name="tool"
    )
    user_content = [TextBlock(text="save a value"), user_image, TextBlock(text="after user image")]
    waiting = await first_runner.start(AgentRunRequest(input=user_content, run_id="protocol-run"))
    assert waiting.pending_interaction is not None
    assert writes == []
    before = first_runner.store.load_session("default")
    before_json = [message.model_dump_json() for message in before.messages]
    saved_assistant = before.messages[-1]
    assert saved_assistant.tool_calls[0].id == "call-save"
    assert style in saved_assistant.metadata
    await first_runner.aclose()
    source.write_bytes(b"original source is unavailable as an image")
    restarted = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=client),
        store=SQLiteStore(database),
        live_publisher=publisher,
    )
    result = await restarted.resume(
        "protocol-run",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert writes == ["saved"]
    assert result.run.usage.input_tokens == 6
    assert result.run.usage.output_tokens == 4
    history = restarted.store.load_session("default").messages
    assert [message.model_dump_json() for message in history[: len(before.messages)]] == before_json
    assert sum(message.content == user_content for message in history) == 1
    result_block = next(block for message in history for block in message.tool_results)
    assert result_block.tool_use_id == "call-save"
    assert result_block.content == [
        TextBlock(text="saved"),
        tool_image,
        TextBlock(text="after image"),
    ]
    assert (
        sum(isinstance(block, ImageBlock) for message in history for block in message.blocks) == 1
    )
    user_url = (
        f"data:image/png;base64,{base64.b64encode(user_image.model.path.read_bytes()).decode()}"
    )
    tool_url = (
        f"data:image/png;base64,{base64.b64encode(tool_image.model.path.read_bytes()).decode()}"
    )
    if style == "responses":
        receipt = next(
            item for item in requests[1]["input"] if item["type"] == "function_call_output"
        )
        assert receipt["call_id"] == "call-save"
        assert [part["type"] for part in receipt["output"]] == [
            "input_text",
            "input_image",
            "input_text",
        ]
        assert receipt["output"][1] == {
            "type": "input_image",
            "image_url": tool_url,
            "detail": "high",
        }
        user = next(item for item in requests[1]["input"] if item.get("role") == "user")
        assert [part["type"] for part in user["content"]] == [
            "input_text",
            "input_image",
            "input_text",
        ]
        assert user["content"][1]["image_url"] == user_url
        replay = next(item for item in requests[1]["input"] if item["type"] == "reasoning")
        assert replay["id"] == "rs-saved" and replay["encrypted_content"] == "saved-reasoning"
        call = next(item for item in requests[1]["input"] if item["type"] == "function_call")
        assert call["id"] == "fc-save" and call["call_id"] == "call-save"
    else:
        wire = requests[1]["messages"]
        receipt_index = next(index for index, item in enumerate(wire) if item["role"] == "tool")
        receipt = wire[receipt_index]
        assert receipt["tool_call_id"] == "call-save"
        assert "saved" in receipt["content"]
        assert "after image" in receipt["content"] and "call_id=call-save" in receipt["content"]
        projected = wire[receipt_index + 1]
        assert projected["role"] == "user" and "save" in projected["content"][0]["text"]
        assert projected["content"][1] == {
            "type": "image_url",
            "image_url": {"url": tool_url, "detail": "high"},
        }
        user = next(item for item in wire if item["role"] == "user")
        assert [part["type"] for part in user["content"]] == ["text", "image_url", "text"]
        assert user["content"][1]["image_url"]["url"] == user_url
        assert wire[receipt_index - 1]["reasoning_content"] == "saved reasoning"
    await restarted.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["responses", "chat_completions"])
async def test_protocol_summary_uses_same_client_and_normalized_finish_reason(
    monkeypatch: pytest.MonkeyPatch, style: ApiStyle
) -> None:
    seen: list[dict[str, Any]] = []

    async def generate(self: object, kwargs: dict[str, Any]) -> Any:
        seen.append(kwargs)
        return _raw_response(style, call=False, text="completed task summary")

    monkeypatch.setattr(
        ResponsesAdapter if style == "responses" else ChatCompletionsAdapter, "invoke", generate
    )
    client = ProviderClient(provider="openai", api_key="test", api_style=style)
    main = LLMRequest(
        model="gpt-4o",
        messages=[Msg.user("next")],
        stream=True,
        tools=[ToolSpec(name="save", input_schema={"type": "object"})],
        tool_choice={"name": "save"},
        response_format="json_object",
    )
    batch = next_summary_batch(
        main,
        None,
        serialize_history([Msg.user("old task")], start_index=0),
        (0, 0),
        CompactionConfig(),
        lambda request: 1,
        system_prompt="Summarize.",
        prompt_renderer=TemplateRenderer(),
    )
    response = await client.complete(batch.request)
    assert consume_summary_response(response) == "completed task summary"
    assert response.finish_reason == "stop" and response.total_tokens == 5
    assert batch.request.tools == [] and batch.request.tool_choice is None
    assert batch.request.response_format is None and not batch.request.stream
    assert seen[0]["num_retries"] == 0
    assert not {"tools", "tool_choice", "response_format", "text"} & seen[0].keys()
