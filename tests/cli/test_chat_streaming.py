"""Chat CLI 流式输出集成测试。

Example:
    uv run pytest tests/cli/test_chat_streaming.py
"""

# region imports
from __future__ import annotations

import importlib
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from iris.cli.chat import ChatOptions, run_chat_loop
from iris.harness import AgentRunner
from iris.message import (
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockRef,
    ModelBlockStarted,
    ModelResponseCompleted,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
)
from iris.store import InMemoryLifecycleStore
from iris.streaming import LiveStreamBroker
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


def test_run_chat_injects_same_live_broker_into_runner_and_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """公开 chat 入口必须把同一 broker 交给 runner 与 terminal host。"""
    chat = importlib.import_module("iris.cli.chat")
    captured: dict[str, Any] = {}
    runner = cast(AgentRunner, object())

    def fake_from_config_path(
        cls: type[AgentRunner],
        path: Path,
        **kwargs: Any,
    ) -> AgentRunner:
        del cls
        captured["path"] = path
        captured["publisher"] = kwargs["live_publisher"]
        return runner

    def fake_run_chat_loop(**kwargs: Any) -> int:
        captured["loop_broker"] = kwargs["live_broker"]
        return 0

    monkeypatch.setattr(chat, "is_config_initialized", lambda: True)
    monkeypatch.setattr(
        chat.AgentRunner,
        "from_config_path",
        classmethod(fake_from_config_path),
    )
    monkeypatch.setattr(chat, "run_chat_loop", fake_run_chat_loop)

    code = chat.run_chat(ChatOptions(config_path=tmp_path / "agent.yaml"))

    assert code == 0
    assert isinstance(captured["publisher"], LiveStreamBroker)
    assert captured["loop_broker"] is captured["publisher"]


def test_chat_loop_streams_text_without_reprinting_terminal_response(tmp_path: Path) -> None:
    """Chat 应消费文本增量，并跳过相同 durable assistant 的重复输出。"""
    provider = FakeStreamingProvider([_text_stream()])
    broker = LiveStreamBroker(replay_capacity_per_scope=64, subscription_capacity=16)
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
        live_publisher=broker,
    )
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

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        live_broker=broker,
        input_func=read_input,
        output_func=outputs.append,
        stream_output_func=write_stream,
        error_func=errors.append,
    )

    assert code == 0
    assert "".join(streamed) == "你好\n"
    assert "你好" not in outputs
    assert len(provider.stream_requests) == 1
    assert provider.stream_requests[0].stream is True
    assert errors == []
