"""Fun-ASR task 协议、句子聚合及流生命周期契约。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from uuid import UUID

import pytest
from websockets.exceptions import ConnectionClosedOK

from iris.exceptions import IrisSpeechError
from iris.speech import SpeechClient, TranscriptionEvent
from iris.speech.adapters import _transport, dashscope_funasr
from iris.speech.adapters.dashscope_funasr import DashScopeFunASRAdapter

from .fakes import FakeConnect, FakeWebSocket, Frame


async def audio(*chunks: bytes) -> AsyncGenerator[bytes, None]:
    """提供已知有限音频。"""
    for chunk in chunks:
        yield chunk


def client() -> SpeechClient:
    """组合与服务商无关的公共调用入口。"""
    return SpeechClient(
        DashScopeFunASRAdapter(endpoint="wss://fun.test", model="fun-asr-realtime", api_key="key")
    )


def event(socket: FakeWebSocket, name: str, sentence: dict[str, object] | None = None) -> str:
    """按官方事件结构构造响应，任务 ID 来自实际 run-task。"""
    header = {
        "task_id": json.loads(socket.sent[0])["header"]["task_id"],
        "event": name,
        "attributes": {},
    }
    if name == "task-failed":
        header.update(error_code="CLIENT_ERROR", error_message="request timeout")
    payload = {} if sentence is None else {"output": {"sentence": sentence}, "usage": None}
    return json.dumps({"header": header, "payload": payload})


def sentence(identifier: int, text: str, *, confirmed: bool = False) -> dict[str, object]:
    """保留原生可选 heartbeat 缺省形状的句子 fixture。"""
    return {
        "begin_time": 0,
        "end_time": None,
        "sentence_id": identifier,
        "sentence_begin": True,
        "sentence_end": confirmed,
        "text": text,
        "words": [],
    }


def auto_respond(socket: FakeWebSocket, *sentences: dict[str, object]) -> None:
    """在启动和输入结束时投递独立的服务事件。"""

    async def on_send(frame: Frame) -> None:
        if isinstance(frame, bytes):
            return
        action = json.loads(frame)["header"]["action"]
        if action == "run-task":
            socket.incoming.put_nowait(event(socket, "task-started"))
        else:
            assert action == "finish-task"
            for result in sentences:
                socket.incoming.put_nowait(event(socket, "result-generated", result))
            socket.incoming.put_nowait(event(socket, "task-finished"))

    socket.on_send = on_send


@pytest.mark.asyncio
async def test_handshake_waits_for_started_then_sends_raw_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket()
    connect = FakeConnect(socket)
    monkeypatch.setattr(_transport, "connect", connect)
    speech = client()
    assert connect.calls == []
    stream = speech.stream(audio(b"abcd", b"ef"))
    task = asyncio.create_task(anext(stream))
    await socket.receive_waiting.wait()
    assert len(socket.sent) == 1
    assert connect.calls == [("wss://fun.test", {"Authorization": "Bearer key"}, 10)]
    request = json.loads(socket.sent[0])
    task_id = request["header"]["task_id"]
    assert str(UUID(task_id)) == task_id
    assert request == {
        "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": "fun-asr-realtime",
            "parameters": {"format": "pcm", "sample_rate": 16000, "heartbeat": True},
            "input": {},
        },
    }
    auto_respond(socket)
    socket.incoming.put_nowait(event(socket, "task-started"))
    assert await asyncio.wait_for(task, 1) == TranscriptionEvent("", True)
    assert socket.sent[1:3] == [b"abcd", b"ef"]
    assert json.loads(socket.sent[3]) == {
        "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {"input": {}},
    }
    await stream.aclose()
    assert socket.closed
    assert socket.active_receives == 0
    assert socket.max_receives == 1


@pytest.mark.asyncio
async def test_live_sentence_revisions_and_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    release_audio = asyncio.Event()
    source_waiting = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        yield b"ab"
        source_waiting.set()
        await release_audio.wait()

    auto_respond(socket)
    async with aclosing(client().stream(source())) as events:
        first = asyncio.create_task(anext(events))
        await source_waiting.wait()
        assert socket.sent[1] == b"ab"
        socket.incoming.put_nowait(event(socket, "result-generated", sentence(1, "初稿")))
        assert await first == TranscriptionEvent("初稿", False)
        heartbeat = sentence(0, "不应出现") | {"heartbeat": True}
        socket.incoming.put_nowait(event(socket, "result-generated", heartbeat))
        revised = sentence(1, " hello world ", confirmed=True) | {"begin_time": 170}
        socket.incoming.put_nowait(event(socket, "result-generated", revised))
        assert await anext(events) == TranscriptionEvent(" hello world ", False)
        socket.incoming.put_nowait(
            event(socket, "result-generated", sentence(2, " hello world ", confirmed=True))
        )
        assert await anext(events) == TranscriptionEvent(" hello world \n hello world ", False)
        assert not release_audio.is_set()
        release_audio.set()
        assert [item async for item in events] == [
            TranscriptionEvent(" hello world \n hello world ", True)
        ]
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sentences", "final"),
    [
        ((sentence(1, "末句"), sentence(1, "末句", confirmed=True)), "末句"),
        ((sentence(1, "确认", confirmed=True), sentence(2, "未确认")), "确认"),
        ((sentence(1, "未确认"),), ""),
        ((), ""),
        ((sentence(2, "第二", confirmed=True), sentence(1, "第一", confirmed=True)), "第一\n第二"),
    ],
)
async def test_only_confirmed_sentences_enter_final(
    monkeypatch: pytest.MonkeyPatch, sentences: tuple[dict[str, object], ...], final: str
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    auto_respond(socket, *sentences)
    events = [item async for item in client().stream(audio(b"ab"))]
    assert events[-1] == TranscriptionEvent(final, True)
    assert sum(item.is_final for item in events) == 1
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_service_failure_stops_waiting_input(
    monkeypatch: pytest.MonkeyPatch, started: bool
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    waiting = asyncio.Event()
    stopped = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        try:
            yield b"ab"
            waiting.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def consume() -> None:
        async with aclosing(source()) as source_stream:
            async for _item in client().stream(source_stream):
                pytest.fail("失败不应产生成功结果")

    task = asyncio.create_task(consume())
    await socket.receive_waiting.wait()
    if started:
        socket.incoming.put_nowait(event(socket, "task-started"))
        await waiting.wait()
    socket.incoming.put_nowait(event(socket, "task-failed"))
    with pytest.raises(IrisSpeechError) as caught:
        await asyncio.wait_for(task, 1)
    assert caught.value.context["provider_code"] == "CLIENT_ERROR"
    assert caught.value.context["provider_message"] == "request timeout"
    assert stopped.is_set()
    assert socket.closed
    assert socket.active_receives == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    ["not-json", "[]", '{"header":{}}', b"{}", '{"header":{"event":"unknown"}}'],
)
async def test_malformed_response_is_domain_error(
    monkeypatch: pytest.MonkeyPatch, invalid: Frame
) -> None:
    socket = FakeWebSocket()
    socket.incoming.put_nowait(invalid)
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    with pytest.raises(IrisSpeechError):
        async for _item in client().stream(audio(b"ab")):
            pass
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_sentence",
    [
        {},
        sentence(0, "无效"),
        sentence(1, "无效") | {"text": 42},
        sentence(1, "无效") | {"sentence_end": "true"},
    ],
)
async def test_invalid_sentence_is_domain_error(
    monkeypatch: pytest.MonkeyPatch, invalid_sentence: dict[str, object]
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    auto_respond(socket, invalid_sentence)
    with pytest.raises(IrisSpeechError):
        async for _item in client().stream(audio(b"ab")):
            pass
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["source", "send", "receive", "cancel", "break"])
async def test_failure_and_cancellation_close_both_directions(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    auto_respond(socket)
    waiting = asyncio.Event()
    stopped = asyncio.Event()
    failure = OSError("device or network failed")

    async def source() -> AsyncGenerator[bytes, None]:
        try:
            yield b"ab"
            waiting.set()
            if mode == "source":
                raise failure
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def consume() -> None:
        async with aclosing(source()) as source_stream:
            async with aclosing(client().stream(source_stream)) as events:
                async for _item in events:
                    break

    if mode == "send":

        async def fail_send(frame: Frame) -> None:
            if isinstance(frame, bytes):
                raise failure
            socket.incoming.put_nowait(event(socket, "task-started"))

        socket.on_send = fail_send
    task = asyncio.create_task(consume())
    if mode == "send":
        with pytest.raises(IrisSpeechError) as caught:
            await task
        assert caught.value.__cause__ is failure
    else:
        await waiting.wait()
        if mode == "source":
            with pytest.raises(OSError) as caught:
                await task
            assert caught.value is failure
        elif mode == "receive":
            socket.incoming.put_nowait(failure)
            with pytest.raises(IrisSpeechError) as caught:
                await asyncio.wait_for(task, 1)
            assert caught.value.__cause__ is failure
        elif mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            socket.incoming.put_nowait(event(socket, "result-generated", sentence(1, "预览")))
            await asyncio.wait_for(task, 1)
    assert stopped.is_set()
    assert socket.closed
    assert socket.active_receives == 0
    assert all(
        isinstance(frame, bytes) or json.loads(frame)["header"]["action"] != "finish-task"
        for frame in socket.sent
    )


@pytest.mark.asyncio
async def test_missing_started_times_out_without_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    monkeypatch.setattr(dashscope_funasr, "_START_TIMEOUT", 0.005)
    with pytest.raises(IrisSpeechError, match="超时"):
        async for _item in client().stream(audio(b"ab")):
            pass
    assert len(socket.sent) == 1
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_action", ["run-task", "finish-task"])
async def test_control_send_failure_is_not_success(
    monkeypatch: pytest.MonkeyPatch, failed_action: str
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    failure = OSError("control send failed")

    async def on_send(frame: Frame) -> None:
        if isinstance(frame, bytes):
            return
        action = json.loads(frame)["header"]["action"]
        if action == failed_action:
            raise failure
        socket.incoming.put_nowait(event(socket, "task-started"))

    socket.on_send = on_send
    with pytest.raises(IrisSpeechError) as caught:
        async for _item in client().stream(audio(b"ab")):
            pytest.fail("控制帧发送失败不应产生 final")
    assert caught.value.__cause__ is failure
    assert socket.closed
    assert socket.active_receives == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("early", [False, True])
async def test_terminal_before_finish_send_returns_waits_only_after_eof(
    monkeypatch: pytest.MonkeyPatch, early: bool
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    sending_finish = asyncio.Event()
    release_send = asyncio.Event()

    async def on_send(frame: Frame) -> None:
        if isinstance(frame, bytes):
            if early:
                socket.incoming.put_nowait(event(socket, "task-finished"))
            return
        if json.loads(frame)["header"]["action"] == "run-task":
            socket.incoming.put_nowait(event(socket, "task-started"))
        else:
            socket.incoming.put_nowait(event(socket, "task-finished"))
            sending_finish.set()
            await release_send.wait()

    async def source() -> AsyncGenerator[bytes, None]:
        yield b"ab"
        if early:
            await asyncio.Event().wait()

    socket.on_send = on_send

    async def consume() -> list[TranscriptionEvent]:
        return [item async for item in client().stream(source())]

    task = asyncio.create_task(consume())
    if early:
        with pytest.raises(IrisSpeechError):
            await asyncio.wait_for(task, 1)
    else:
        await sending_finish.wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.01)
        release_send.set()
        assert await asyncio.wait_for(task, 1) == [TranscriptionEvent("", True)]
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [False, True])
async def test_missing_terminal_failure_and_fresh_next_stream(
    monkeypatch: pytest.MonkeyPatch, closed: bool
) -> None:
    first, second = FakeWebSocket(), FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(first, second))
    monkeypatch.setattr(dashscope_funasr, "_FINAL_TIMEOUT", 0.02)

    async def on_send(frame: Frame) -> None:
        if isinstance(frame, bytes):
            return
        if json.loads(frame)["header"]["action"] == "run-task":
            first.incoming.put_nowait(event(first, "task-started"))
        else:
            first.incoming.put_nowait(event(first, "result-generated", sentence(1, "旧文本")))
            if closed:
                first.incoming.put_nowait(ConnectionClosedOK(None, None))

    first.on_send = on_send
    auto_respond(second)
    speech = client()
    async with aclosing(speech.stream(audio(b"ab"))) as events:
        assert await anext(events) == TranscriptionEvent("旧文本", False)
        with pytest.raises(IrisSpeechError):
            async with asyncio.timeout(1):
                while True:
                    first.incoming.put_nowait(event(first, "result-generated", sentence(1, "更新")))
                    await anext(events)
                    await asyncio.sleep(0.005)
    assert first.closed
    assert [item async for item in speech.stream(audio(b"cd"))] == [TranscriptionEvent("", True)]
    assert (
        json.loads(first.sent[0])["header"]["task_id"]
        != json.loads(second.sent[0])["header"]["task_id"]
    )
    assert second.closed
