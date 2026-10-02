"""同一 runtime 工具、持久化恢复与摘要契约经过真实双协议 adapter。"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import pytest

from iris.agents import CompactionConfig
from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import LLMRequest, Msg, ToolSpec
from iris.providers import ProviderClient
from iris.providers.chat_completions import ChatCompletionsAdapter
from iris.providers.responses import ResponsesAdapter
from iris.runtime._compaction_summary import (
    consume_summary_response,
    next_summary_batch,
    serialize_history,
)
from iris.store import SQLiteStore
from iris.tools import ToolCapability, ToolRegistry
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
            "output": [output],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }
    message = {"role": "assistant", "content": None if call else text}
    if call:
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

    def save(value: str) -> str:
        writes.append(value)
        return value

    registry = ToolRegistry()
    registry.register_function(save, description="保存", capabilities={ToolCapability.WRITE})
    database = tmp_path / "protocol.db"
    publisher = RecordingPublisher() if stream else None
    waiting = await AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=client),
        store=SQLiteStore(database),
        live_publisher=publisher,
    ).start(AgentRunRequest(input="save a value", run_id="protocol-run"))
    assert waiting.pending_interaction is not None
    assert writes == []
    result = await AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=client),
        store=SQLiteStore(database),
        live_publisher=publisher,
    ).resume(
        "protocol-run",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert writes == ["saved"]
    assert result.run.usage.input_tokens == 6
    assert result.run.usage.output_tokens == 4
    if style == "responses":
        receipt = next(
            item for item in requests[1]["input"] if item["type"] == "function_call_output"
        )
        assert receipt["call_id"] == "call-save"
        assert "saved" in str(receipt["output"])
    else:
        receipt = next(item for item in requests[1]["messages"] if item["role"] == "tool")
        assert receipt["tool_call_id"] == "call-save"
        assert "saved" in receipt["content"]


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
