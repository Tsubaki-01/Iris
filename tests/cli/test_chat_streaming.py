"""Chat CLI 流式输出集成测试。

Example:
    uv run pytest tests/cli/test_chat_streaming.py
"""

# region imports
from __future__ import annotations

import asyncio
import importlib
import threading
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris.cli.chat import ChatOptions, _ChatLiveOutput, run_chat_loop
from iris.harness import AgentRunner
from iris.message import (
    LLMRequest,
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockRef,
    ModelBlockStarted,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ProviderStreamError,
)
from iris.store import InMemoryLifecycleStore
from tests.harness.fakes import build_runtime, text_response
from tests.runtime.fakes import FakeStreamingProvider

# endregion


def _text_stream() -> list[ModelStreamEvent]:
    """构造两段文本增量和一个成功终态。"""
    response = text_response("你好")
    scope = ModelStreamScope(
        model_stream_id="stream-1",
        provider=response.provider,
        model=response.model,
        attempt=1,
    )
    block = ModelBlockRef(index=0, block_id="text-0", kind="text")
    occurred_at = datetime.now(UTC)
    return [
        ModelResponseStarted(
            scope=scope,
            sequence=1,
            occurred_at=occurred_at,
            response_id=response.id,
        ),
        ModelBlockStarted(
            scope=scope,
            sequence=2,
            occurred_at=occurred_at,
            block=block,
        ),
        ModelBlockDelta(
            scope=scope,
            sequence=3,
            occurred_at=occurred_at,
            block=block,
            channel="text",
            delta="你",
            snapshot="你",
        ),
        ModelBlockDelta(
            scope=scope,
            sequence=4,
            occurred_at=occurred_at,
            block=block,
            channel="text",
            delta="好",
            snapshot="你好",
        ),
        ModelBlockCompleted(
            scope=scope,
            sequence=5,
            occurred_at=occurred_at,
            block=block,
        ),
        ModelResponseCompleted(
            scope=scope,
            sequence=6,
            occurred_at=occurred_at,
            response=response,
            semantic_output_emitted=True,
        ),
    ]


def test_run_chat_displays_each_delta_before_reading_the_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """本地输出在每个增量到达时完成，不等待独立 consumer 调度。"""
    chat = importlib.import_module("iris.cli.chat")
    config = importlib.import_module("iris.config")
    provider_client = importlib.import_module("iris.providers.client")
    monkeypatch.setattr(config, "_config", config.Config(api_key="test-key"))
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "name: chat-test\nmodel:\n  provider: openai\n  name: test-model\nsystem: 请帮助用户\n",
        encoding="utf-8",
    )
    outputs: list[str] = []
    errors: list[str] = []
    observed: list[str] = []
    closed = threading.Event()
    finished = threading.Event()

    async def raw_chunks() -> AsyncIterator[dict[str, Any]]:
        try:
            for fragment in ("你", "好"):
                yield {"choices": [{"delta": {"content": fragment}}]}
                observed.append("".join(outputs))
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        finally:
            closed.set()

    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        return raw_chunks()

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        input_index += 1
        if input_index == 1:
            return "开始"
        assert finished.wait(2)
        return "/exit"

    def write_output(fragment: str) -> None:
        outputs.append(fragment)
        if fragment == "\n":
            finished.set()

    code = chat.run_chat(
        ChatOptions(config_path=config_path),
        input_func=read_input,
        output_func=write_output,
        error_func=errors.append,
    )

    assert code == 0
    assert observed == ["你", "你好"]
    assert "".join(outputs) == "你好\n"
    assert closed.is_set()
    assert errors == []


@pytest.mark.parametrize("text", ["完整结果", ""])
def test_chat_loop_displays_complete_response_when_no_text_delta_arrived(
    tmp_path: Path, text: str
) -> None:
    """未输出文本增量时，durable terminal 仍补显完整结果，包括空响应。"""
    [started, *_] = _text_stream()
    provider = FakeStreamingProvider(
        [
            [
                started,
                ModelResponseCompleted(
                    scope=started.scope,
                    sequence=2,
                    occurred_at=started.occurred_at,
                    response=text_response(text),
                    semantic_output_emitted=False,
                ),
            ]
        ]
    )
    streamed: list[str] = []
    outputs: list[str] = []
    errors: list[str] = []
    finished = threading.Event()
    live_output = _ChatLiveOutput(streamed.append)
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
        live_publisher=live_output,
    )
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        input_index += 1
        if input_index == 1:
            return "开始"
        assert finished.wait(2)
        return "/exit"

    def write_output(message: str) -> None:
        outputs.append(message)
        finished.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        live_output=live_output,
        input_func=read_input,
        output_func=write_output,
        error_func=errors.append,
    )

    assert code == 0
    assert outputs == [text]
    assert streamed == []
    assert errors == []


@pytest.mark.parametrize("terminal_kind", ["failed", "cancelled", "eof"])
def test_chat_loop_finishes_partial_line_before_reporting_stream_failure(
    tmp_path: Path, terminal_kind: str
) -> None:
    """失败、provider 取消和提前 EOF 都只补一次换行，再显示 durable 错误。"""
    events = _text_stream()[:3]
    started = events[0]
    if terminal_kind == "failed":
        events.append(
            ModelResponseFailed(
                scope=started.scope,
                sequence=4,
                occurred_at=started.occurred_at,
                error=ProviderStreamError(
                    code="PROVIDER_STREAM_ERROR", message="上游失败", retryable=False
                ),
                semantic_output_emitted=True,
            )
        )
    elif terminal_kind == "cancelled":
        events.append(
            ModelResponseCancelled(
                scope=started.scope,
                sequence=4,
                occurred_at=started.occurred_at,
                semantic_output_emitted=True,
            )
        )
    provider = FakeStreamingProvider([events])
    displayed: list[str] = []
    failed = threading.Event()
    live_output = _ChatLiveOutput(displayed.append)
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
        live_publisher=live_output,
    )
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        input_index += 1
        if input_index == 1:
            return "开始"
        assert failed.wait(2)
        return ""

    def write_error(message: str) -> None:
        displayed.append(message)
        failed.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        live_output=live_output,
        input_func=read_input,
        output_func=displayed.append,
        error_func=write_error,
    )

    assert code == 1
    assert displayed[:2] == ["你", "\n"]
    assert len(displayed) == 3
    assert displayed[2].startswith("provider:PROVIDER_STREAM_")


def test_chat_interrupt_closes_partial_stream_and_finishes_output_line(tmp_path: Path) -> None:
    """主线程中断会关闭尚无模型终态的 iterator，并为已显示文本收尾。"""
    displayed: list[str] = []
    errors: list[str] = []
    partial = threading.Event()
    closed = threading.Event()

    class BlockingStreamingProvider(FakeStreamingProvider):
        """输出部分文本后等待 host 取消的 provider。"""

        async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
            """保留真实异步生成器关闭行为。"""
            try:
                async for event in super().stream(request):
                    yield event
                await asyncio.Event().wait()
            finally:
                closed.set()

    provider = BlockingStreamingProvider([_text_stream()[:3]])

    def write_stream(fragment: str) -> None:
        displayed.append(fragment)
        if fragment == "你":
            partial.set()

    live_output = _ChatLiveOutput(write_stream)
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
        live_publisher=live_output,
    )
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        input_index += 1
        if input_index == 1:
            return "开始"
        assert partial.wait(2)
        raise KeyboardInterrupt

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        live_output=live_output,
        input_func=read_input,
        output_func=displayed.append,
        error_func=errors.append,
    )

    assert code == 130
    assert displayed == ["你", "\n"]
    assert closed.is_set()
    assert errors == []


def test_chat_loop_streams_text_without_reprinting_terminal_response(tmp_path: Path) -> None:
    """Chat 应消费文本增量，并跳过相同 durable assistant 的重复输出。"""
    provider = FakeStreamingProvider([_text_stream()])
    streamed: list[str] = []
    outputs: list[str] = []
    errors: list[str] = []
    finished = threading.Event()
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        del prompt
        input_index += 1
        if input_index == 1:
            return "开始"
        assert finished.wait(1)
        return "/exit"

    def write_stream(fragment: str) -> None:
        streamed.append(fragment)
        if fragment == "\n":
            finished.set()

    live_output = _ChatLiveOutput(write_stream)
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
        live_publisher=live_output,
    )
    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        live_output=live_output,
        input_func=read_input,
        output_func=outputs.append,
        error_func=errors.append,
    )

    assert code == 0
    assert "".join(streamed) == "你好\n"
    assert "你好" not in outputs
    assert len(provider.stream_requests) == 1
    assert provider.stream_requests[0].stream is True
    assert errors == []
