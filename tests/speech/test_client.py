"""公共转录入口的音频、委托与关闭契约。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import aclosing

import pytest

from iris.exceptions import IrisProviderError, IrisSpeechError
from iris.speech import SpeechAdapter, SpeechClient, TranscriptionEvent


class RecordingAdapter:
    """记录音频并提供可观察关闭状态的内存 adapter。"""

    def __init__(
        self,
        events: tuple[TranscriptionEvent, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.started = 0
        self.closed = 0
        self.chunks: list[bytes] = []

    async def stream(self, audio: AsyncIterable[bytes]) -> AsyncGenerator[TranscriptionEvent, None]:
        """按消费顺序记录输入，随后转发测试事件。"""
        self.started += 1
        try:
            async for chunk in audio:
                self.chunks.append(chunk)
                yield TranscriptionEvent(text=str(len(self.chunks)), is_final=False)
            for event in self.events:
                yield event
            if self.error is not None:
                raise self.error
        finally:
            self.closed += 1


async def audio_source(*chunks: bytes) -> AsyncGenerator[bytes, None]:
    """逐块提供指定音频。"""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_stream_creation_does_not_read_audio_or_start_adapter() -> None:
    read = False

    async def source() -> AsyncGenerator[bytes, None]:
        nonlocal read
        read = True
        yield b"\x00\x00"

    adapter: SpeechAdapter = RecordingAdapter()
    client = SpeechClient(adapter)
    async with aclosing(source()) as audio, aclosing(client.stream(audio)):
        assert not read
        assert adapter.started == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [(b"ab",), (b"ab", b"cdef", b"gh")])
async def test_audio_keeps_order_and_is_consumed_lazily(chunks: tuple[bytes, ...]) -> None:
    adapter = RecordingAdapter()
    client = SpeechClient(adapter)
    async with aclosing(client.stream(audio_source(*chunks))) as events:
        await anext(events)
        assert adapter.chunks == [chunks[0]]
        assert adapter.closed == 0
        async for _event in events:
            pass
    assert adapter.chunks == list(chunks)
    assert adapter.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [(), (b"",), (b"a",)])
async def test_empty_or_invalid_first_chunk_does_not_start_adapter(
    chunks: tuple[bytes, ...],
) -> None:
    adapter = RecordingAdapter()
    with pytest.raises(IrisSpeechError):
        async for _event in SpeechClient(adapter).stream(audio_source(*chunks)):
            pass
    assert adapter.started == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_chunk", [b"", b"a"])
async def test_invalid_later_chunk_closes_adapter(invalid_chunk: bytes) -> None:
    adapter = RecordingAdapter()
    with pytest.raises(IrisSpeechError):
        async for _event in SpeechClient(adapter).stream(audio_source(b"ab", invalid_chunk)):
            pass
    assert adapter.chunks == [b"ab"]
    assert adapter.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected",
    [
        (
            TranscriptionEvent("识别初稿", False),
            TranscriptionEvent("识别修正", False),
            TranscriptionEvent("识别修正", True),
        ),
        (TranscriptionEvent("", True),),
    ],
)
async def test_events_are_passed_through_without_rebuilding(
    expected: tuple[TranscriptionEvent, ...],
) -> None:
    adapter = RecordingAdapter(expected)
    result = [event async for event in SpeechClient(adapter).stream(audio_source(b"ab"))]
    assert len(result) == len(expected) + 1
    assert all(actual is original for actual, original in zip(result[1:], expected, strict=True))
    assert adapter.closed == 1


@pytest.mark.asyncio
async def test_adapter_error_is_preserved_after_cleanup() -> None:
    error = IrisSpeechError("provider failed", adapter="test")
    adapter = RecordingAdapter(error=error)
    with pytest.raises(IrisSpeechError) as caught:
        async for event in SpeechClient(adapter).stream(audio_source(b"ab")):
            assert not event.is_final
    assert caught.value is error
    assert adapter.closed == 1
    assert isinstance(error, IrisProviderError)
    assert error.runtime_source == "provider"
    assert error.runtime_code == "SPEECH_ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("after_first_chunk", [False, True])
async def test_source_error_is_preserved(after_first_chunk: bool) -> None:
    error = OSError("microphone disconnected")

    async def source() -> AsyncGenerator[bytes, None]:
        if after_first_chunk:
            yield b"ab"
        raise error

    adapter = RecordingAdapter()
    with pytest.raises(OSError) as caught:
        async for event in SpeechClient(adapter).stream(source()):
            assert not event.is_final
    assert caught.value is error
    assert adapter.started == int(after_first_chunk)
    assert adapter.closed == int(after_first_chunk)


@pytest.mark.asyncio
async def test_cancellation_closes_adapter_and_propagates() -> None:
    waiting_for_audio = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        yield b"ab"
        waiting_for_audio.set()
        await asyncio.Event().wait()

    adapter = RecordingAdapter()

    async def consume() -> None:
        async with aclosing(SpeechClient(adapter).stream(source())) as events:
            async for event in events:
                assert not event.is_final

    task = asyncio.create_task(consume())
    await waiting_for_audio.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.closed == 1


@pytest.mark.asyncio
async def test_break_closes_adapter_but_host_retains_audio_source() -> None:
    source_closed = False

    async def source() -> AsyncGenerator[bytes, None]:
        nonlocal source_closed
        try:
            yield b"ab"
            yield b"cd"
        finally:
            source_closed = True

    adapter = RecordingAdapter()
    client = SpeechClient(adapter)
    async with aclosing(source()) as audio:
        async with aclosing(client.stream(audio)) as events:
            async for _event in events:
                break
        assert adapter.closed == 1
        assert not source_closed
        assert await anext(audio) == b"cd"
    assert source_closed

    async for _event in client.stream(audio_source(b"ef")):
        pass
    assert adapter.chunks == [b"ab", b"ef"]
    assert adapter.started == adapter.closed == 2
