"""豆包原生流的握手、并发、收尾和错误契约。"""

from __future__ import annotations

import asyncio
import gzip
import json
import struct
from collections.abc import AsyncGenerator
from contextlib import aclosing
from uuid import UUID

import pytest
from websockets.exceptions import ConnectionClosedOK

from iris.exceptions import IrisSpeechError
from iris.speech import SpeechClient, TranscriptionEvent
from iris.speech.adapters import _transport, doubao
from iris.speech.adapters.doubao import DoubaoASRAdapter

from .fakes import FakeConnect, FakeWebSocket, Frame


def response(text: str | None = None, *, final: bool = False, definite: bool = False) -> bytes:
    """按官方未压缩 full response 布局构造服务端 fixture。"""
    payload: dict[str, object] = {}
    if text is not None:
        payload["result"] = {"text": text, "utterances": [{"definite": definite}]}
    body = json.dumps(payload).encode()
    return bytes([0x11, 0x92 if final else 0x90, 0x10, 0]) + struct.pack(">I", len(body)) + body


def error_response() -> bytes:
    """使用官方 error 布局，不调用待测 codec。"""
    body = b'{"message":"quota exceeded"}'
    return b"\x11\xf0\x10\x00" + struct.pack(">II", 45000001, len(body)) + body


async def audio(*chunks: bytes) -> AsyncGenerator[bytes, None]:
    """依次提供有限音频。"""
    for chunk in chunks:
        yield chunk


def client() -> SpeechClient:
    """创建不依赖全局配置的直接组合客户端。"""
    return SpeechClient(
        DoubaoASRAdapter(endpoint="wss://speech.test", model="resource", api_key="key")
    )


def finish_with(socket: FakeWebSocket, *frames: Frame | Exception) -> None:
    """仅在真实末音频发出后投递服务端终段响应。"""

    async def on_send(frame: Frame) -> None:
        assert isinstance(frame, bytes)
        if frame[1] == 0x23:
            for incoming in frames:
                socket.incoming.put_nowait(incoming)

    socket.on_send = on_send


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [(b"ab",), (b"abcd", b"efgh", b"ij")])
async def test_handshake_and_real_last_audio_frame(
    monkeypatch: pytest.MonkeyPatch, chunks: tuple[bytes, ...]
) -> None:
    socket = FakeWebSocket()
    finish_with(socket, response("完成", final=True))
    connect = FakeConnect(socket)
    monkeypatch.setattr(_transport, "connect", connect)
    speech = client()
    assert connect.calls == []
    events = [event async for event in speech.stream(audio(*chunks))]
    assert events == [TranscriptionEvent("完成", True)]
    endpoint, headers, timeout = connect.calls[0]
    assert endpoint == "wss://speech.test"
    assert timeout == 10
    assert headers == {
        "X-Api-Key": "key",
        "X-Api-Resource-Id": "resource",
        "X-Api-Request-Id": str(UUID(headers["X-Api-Request-Id"])),
    }
    assert socket.closed
    assert socket.active_receives == 0
    assert socket.max_receives == 1
    packets = socket.sent
    assert len(packets) == len(chunks) + 1
    assert isinstance(packets[0], bytes)
    assert packets[0][:4] == b"\x11\x11\x11\x00"
    assert struct.unpack(">i", packets[0][4:8])[0] == 1
    assert json.loads(gzip.decompress(packets[0][12:])) == {
        "audio": {"format": "pcm", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1},
        "request": {
            "model_name": "bigmodel",
            "result_type": "full",
            "enable_nonstream": True,
            "enable_punc": True,
            "enable_itn": True,
            "enable_ddc": False,
        },
    }
    for index, (packet, chunk) in enumerate(zip(packets[1:], chunks, strict=True), start=2):
        assert isinstance(packet, bytes)
        is_last = index == len(packets)
        assert packet[1] == (0x23 if is_last else 0x21)
        assert struct.unpack(">i", packet[4:8])[0] == (-index if is_last else index)
        assert struct.unpack(">I", packet[8:12])[0] == len(packet[12:])
        assert gzip.decompress(packet[12:]) == chunk


@pytest.mark.asyncio
async def test_partial_revisions_arrive_before_source_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket()
    connect = FakeConnect(socket)
    monkeypatch.setattr(_transport, "connect", connect)
    release_audio = asyncio.Event()
    source_waiting = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        yield b"ab"
        yield b"cd"
        source_waiting.set()
        await release_audio.wait()

    async def on_send(frame: Frame) -> None:
        assert isinstance(frame, bytes)
        if frame[1] == 0x21:
            socket.incoming.put_nowait(response("初稿"))
            socket.incoming.put_nowait(response())
            socket.incoming.put_nowait(response("修正", definite=True))
        elif frame[1] == 0x23:
            socket.incoming.put_nowait(response("修正", final=True))

    socket.on_send = on_send
    async with aclosing(client().stream(source())) as events:
        first = await asyncio.wait_for(anext(events), 1)
        assert first == TranscriptionEvent("初稿", False)
        await source_waiting.wait()
        assert not release_audio.is_set()
        revised = await asyncio.wait_for(anext(events), 1)
        assert revised == TranscriptionEvent("修正", False)
        release_audio.set()
        rest = [event async for event in events]
    assert rest == [TranscriptionEvent("修正", True)]
    assert socket.max_receives == 1
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        ((response("初稿"), response("末修正", final=True)), "末修正"),
        ((response("初稿"), response(final=True)), "初稿"),
        ((response(), response(final=True)), ""),
    ],
)
async def test_final_uses_last_known_text(
    monkeypatch: pytest.MonkeyPatch, frames: tuple[bytes, ...], expected: str
) -> None:
    socket = FakeWebSocket()
    finish_with(socket, *frames)
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    events = [event async for event in client().stream(audio(b"ab"))]
    assert events[-1] == TranscriptionEvent(expected, True)
    assert sum(event.is_final for event in events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("incoming", [error_response(), b"\x11", OSError("connection reset")])
@pytest.mark.parametrize("after_audio", [False, True])
async def test_receive_failure_cancels_waiting_audio_source(
    monkeypatch: pytest.MonkeyPatch, incoming: Frame | Exception, after_audio: bool
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    source_waiting = asyncio.Event()
    source_closed = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        try:
            yield b"ab"
            if after_audio:
                yield b"cd"
            source_waiting.set()
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    async def consume() -> list[TranscriptionEvent]:
        return [event async for event in client().stream(source())]

    task = asyncio.create_task(consume())
    await source_waiting.wait()
    socket.incoming.put_nowait(incoming)
    with pytest.raises(IrisSpeechError) as caught:
        await asyncio.wait_for(task, 1)
    assert caught.value.context["adapter"] == "doubao_asr"
    assert caught.value.context["request_id"]
    assert source_closed.is_set()
    assert socket.closed
    assert socket.active_receives == 0


@pytest.mark.asyncio
async def test_final_deadline_starts_after_last_send_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    monkeypatch.setattr(doubao, "_FINAL_TIMEOUT", 0.005)
    sending_last = asyncio.Event()
    release_send = asyncio.Event()

    async def on_send(frame: Frame) -> None:
        assert isinstance(frame, bytes)
        if frame[1] == 0x23:
            sending_last.set()
            await release_send.wait()
            socket.incoming.put_nowait(response("完成", final=True))

    socket.on_send = on_send

    async def consume() -> list[TranscriptionEvent]:
        return [event async for event in client().stream(audio(b"ab"))]

    task = asyncio.create_task(consume())
    await sending_last.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), 0.02)
    release_send.set()
    assert await asyncio.wait_for(task, 1) == [TranscriptionEvent("完成", True)]
    assert socket.closed


@pytest.mark.asyncio
async def test_terminal_during_last_send_waits_for_send_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    sending_last = asyncio.Event()
    release_send = asyncio.Event()

    async def on_send(frame: Frame) -> None:
        assert isinstance(frame, bytes)
        if frame[1] == 0x23:
            socket.incoming.put_nowait(response("完成", final=True))
            sending_last.set()
            await release_send.wait()

    socket.on_send = on_send

    async def consume() -> list[TranscriptionEvent]:
        return [event async for event in client().stream(audio(b"ab"))]

    task = asyncio.create_task(consume())
    await sending_last.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), 0.01)
    release_send.set()
    assert await asyncio.wait_for(task, 1) == [TranscriptionEvent("完成", True)]
    assert socket.closed


@pytest.mark.asyncio
async def test_source_error_is_not_misclassified_as_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    failure = OSError("microphone failed")

    async def source() -> AsyncGenerator[bytes, None]:
        yield b"ab"
        await socket.receive_waiting.wait()
        raise failure

    with pytest.raises(OSError) as caught:
        async for _event in client().stream(source()):
            pass
    assert caught.value is failure
    assert len(socket.sent) == 1
    assert socket.closed
    assert socket.active_receives == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_packet", [0x11, 0x21, 0x23])
async def test_send_failure_unblocks_receiver(
    monkeypatch: pytest.MonkeyPatch, failed_packet: int
) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    failure = OSError("send failed")

    async def on_send(frame: Frame) -> None:
        assert isinstance(frame, bytes)
        if frame[1] == failed_packet:
            raise failure

    socket.on_send = on_send
    with pytest.raises(IrisSpeechError) as caught:
        async for _event in client().stream(audio(b"ab", b"cd")):
            pass
    assert caught.value.__cause__ is failure
    assert socket.closed
    assert socket.active_receives == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("dial failed"), TimeoutError("dial timeout")])
async def test_connection_failure_is_normalized(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    socket = FakeWebSocket()
    socket.open_error = failure
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    with pytest.raises(IrisSpeechError) as caught:
        async for _event in client().stream(audio(b"ab")):
            pass
    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_cancel_or_break_finishes_tasks_and_connection(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
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

    speech = client()

    async def consume() -> None:
        async with aclosing(speech.stream(source())) as events:
            async for _event in events:
                break

    task = asyncio.create_task(consume())
    await waiting.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        socket.incoming.put_nowait(response("预览"))
        await asyncio.wait_for(task, 1)
    assert stopped.is_set()
    assert socket.closed
    assert socket.active_receives == 0


@pytest.mark.asyncio
async def test_close_without_terminal_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket()
    finish_with(socket, response("未完成"), ConnectionClosedOK(None, None))
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    events = []
    with pytest.raises(IrisSpeechError):
        async for event in client().stream(audio(b"ab")):
            events.append(event)
    assert events == [TranscriptionEvent("未完成", False)]
    assert socket.closed


@pytest.mark.asyncio
async def test_premature_terminal_does_not_wait_for_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket()
    socket.incoming.put_nowait(response("提前结束", final=True))
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))

    async def source() -> AsyncGenerator[bytes, None]:
        yield b"ab"
        await asyncio.Event().wait()

    async def consume() -> None:
        async for _event in client().stream(source()):
            pytest.fail("提前终态不能成为成功事件")

    with pytest.raises(IrisSpeechError):
        await asyncio.wait_for(consume(), 1)
    assert socket.closed


@pytest.mark.asyncio
async def test_final_deadline_is_absolute_and_client_can_be_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = FakeWebSocket(), FakeWebSocket()
    connect = FakeConnect(first, second)
    monkeypatch.setattr(_transport, "connect", connect)
    monkeypatch.setattr(doubao, "_FINAL_TIMEOUT", 0.03)
    finish_with(first, response("旧预览"))
    finish_with(second, response(final=True))
    speech = client()
    async with aclosing(speech.stream(audio(b"ab"))) as events:
        assert await anext(events) == TranscriptionEvent("旧预览", False)
        with pytest.raises(IrisSpeechError, match="超时"):
            async with asyncio.timeout(1):
                while True:
                    first.incoming.put_nowait(response("仍未完成"))
                    await anext(events)
                    await asyncio.sleep(0.005)
    assert first.closed
    assert first.active_receives == 0
    result = [event async for event in speech.stream(audio(b"cd"))]
    assert result == [TranscriptionEvent("", True)]
    assert connect.calls[0][1]["X-Api-Request-Id"] != connect.calls[1][1]["X-Api-Request-Id"]
    assert second.closed
