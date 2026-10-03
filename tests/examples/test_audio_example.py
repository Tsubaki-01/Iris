"""同一 WAV 宿主经两种真实 ASR adapter 装配和提交普通文字。"""

from __future__ import annotations

import asyncio
import gzip
import json
import struct
import wave
from collections.abc import Callable
from contextlib import aclosing
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from examples.audio import transcribe_stream
from iris import config as global_config
from iris.agents import AgentConfig, load_agent_config
from iris.config import Config
from iris.exceptions import IrisSpeechError
from iris.harness import AgentRunner
from iris.lifecycle import RunStopReason
from iris.speech.adapters import _transport
from tests.harness.fakes import StaticProvider, text_response
from tests.speech.fakes import FakeConnect, FakeWebSocket

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "audio"


@pytest.fixture(autouse=True)
def speech_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离凭据，不读取真实账号，也不初始化聊天模型 key。"""
    monkeypatch.setattr(
        global_config,
        "_config",
        Config(
            api_key=None,
            provider_api_keys={"doubao_asr": "doubao-test", "dashscope_funasr": "ali-test"},
        ),
    )


def _config(tmp_path: Path, adapter: str) -> Path:
    name = "doubao.yaml" if adapter == "doubao_asr" else "dashscope.yaml"
    path = tmp_path / name
    path.write_text((EXAMPLE_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _wav(
    tmp_path: Path, *, channels: int = 1, rate: int = 16000, width: int = 2
) -> tuple[Path, bytes]:
    path = tmp_path / "sample.wav"
    pcm = b"\x01\x02" * 1602
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(pcm)
    return path, pcm


def _track_files(monkeypatch: pytest.MonkeyPatch) -> list[Mock]:
    original = wave.open
    closers: list[Mock] = []

    def open_wav(path: str, mode: str) -> wave.Wave_read:
        reader = original(path, mode)
        close = Mock(wraps=reader.close)
        reader.close = close
        closers.append(close)
        return reader

    monkeypatch.setattr(transcribe_stream.wave, "open", open_wav)
    return closers


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    adapter: str,
    *,
    final_text: str = "最终文字",
    fail: bool = False,
    block: bool = False,
) -> tuple[FakeWebSocket, list[bytes], asyncio.Event]:
    socket = FakeWebSocket()
    audio: list[bytes] = []
    waiting = asyncio.Event()
    task_id = ""

    def response(text: str, *, final: bool = False) -> bytes:
        data = json.dumps({"result": {"text": text}}).encode()
        return bytes((0x11, 0x92 if final else 0x90, 0x10, 0)) + struct.pack(">I", len(data)) + data

    def event(name: str, *, text: str | None = None, final: bool = False) -> str:
        payload = (
            {}
            if text is None
            else {"output": {"sentence": {"sentence_id": 1, "text": text, "sentence_end": final}}}
        )
        return json.dumps({"header": {"event": name, "task_id": task_id}, "payload": payload})

    async def send(frame: bytes | str) -> None:
        nonlocal task_id
        if adapter == "doubao_asr":
            assert isinstance(frame, bytes)
            if frame[1] >> 4 != 2:
                return
            audio.append(gzip.decompress(frame[12:]))
            await socket.incoming.put(response("预览文字"))
            finished = bool(frame[1] & 2)
        elif isinstance(frame, str):
            request = json.loads(frame)
            task_id = request["header"]["task_id"]
            if request["header"]["action"] == "run-task":
                await socket.incoming.put(event("task-started"))
                return
            finished = True
        else:
            audio.append(frame)
            await socket.incoming.put(event("result-generated", text="预览文字"))
            finished = False
        waiting.set()
        if fail:
            await socket.incoming.put(OSError("recognizer disconnected"))
        elif block:
            await asyncio.Event().wait()
        elif finished and adapter == "doubao_asr":
            await socket.incoming.put(response(final_text, final=True))
        elif finished:
            await socket.incoming.put(event("result-generated", text=final_text, final=True))
            await socket.incoming.put(event("task-finished"))

    socket.on_send = send
    monkeypatch.setattr(_transport, "connect", FakeConnect(socket))
    return socket, audio, waiting


def _runners(
    monkeypatch: pytest.MonkeyPatch,
    provider: StaticProvider,
    before_create: Callable[[], None],
) -> tuple[list[AgentRunner], list[Path]]:
    original = AgentRunner.from_config
    runners: list[AgentRunner] = []
    paths: list[Path] = []

    def from_config(config: AgentConfig, *, config_path: Path) -> AgentRunner:
        before_create()
        paths.append(config_path)
        runner = original(config, config_path=config_path, provider=provider)
        runner.aclose = AsyncMock(wraps=runner.aclose)
        runners.append(runner)
        return runner

    monkeypatch.setattr(AgentRunner, "from_config", staticmethod(from_config))
    return runners, paths


@pytest.mark.parametrize("adapter", ["doubao_asr", "dashscope_funasr"])
def test_both_complete_yaml_examples_load(adapter: str, tmp_path: Path) -> None:
    config = load_agent_config(_config(tmp_path, adapter))
    assert config.name == "voice-agent"
    assert config.model.provider == "deepseek" and config.model.name == "deepseek-flash"
    assert config.system == "你是一个助手，根据用户提交的文字回答问题。\n"
    assert config.speech.enabled and config.speech.adapter == adapter
    if adapter == "doubao_asr":
        assert config.speech.model == "volc.seedasr.sauc.duration"
        assert config.speech.endpoint == "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
    else:
        assert config.speech.model == "fun-asr-realtime"
        assert config.speech.endpoint == (
            "wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["doubao_asr", "dashscope_funasr"])
@pytest.mark.parametrize("submit", [False, True])
async def test_same_example_sends_only_pcm_and_submits_final_after_cleanup(
    adapter: str,
    submit: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, adapter)
    audio, pcm = _wav(tmp_path)
    closers = _track_files(monkeypatch)
    socket, transmitted, _ = _wire(monkeypatch, adapter)
    provider = StaticProvider(text_response("Agent 回答"))

    def before_create() -> None:
        assert socket.closed
        assert all(close.called for close in closers)

    runners, paths = _runners(monkeypatch, provider, before_create)
    result = await transcribe_stream.run_transcription(
        config_path=config, audio_path=audio, submit=submit, session_id="voice-test"
    )
    assert b"".join(transmitted) == pcm
    assert [len(chunk) for chunk in transmitted] == [3200, 4]
    assert socket.closed and all(close.called for close in closers)
    output = capsys.readouterr().out
    assert "预览文字" in output and "最终文字" in output
    if not submit:
        assert result is None and runners == [] and provider.requests == []
        return
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.assistant_message.text == "Agent 回答"
    assert paths == [config]
    assert len(provider.requests) == 1
    user = [message for message in provider.requests[0].messages if message.role == "user"]
    assert len(user) == 1 and user[0].content == "最终文字"
    assert runners[0].store.load_run(result.run.run_id).request.input == "最终文字"
    runners[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_does_not_open_audio_read_keys_or_create_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "disabled.yaml"
    config.write_text(
        "name: voice-agent\nmodel: deepseek/deepseek-flash\nsystem: 你是一个助手。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(global_config, "_config", None)
    monkeypatch.setattr(
        transcribe_stream.wave, "open", Mock(side_effect=AssertionError("file opened"))
    )
    create = Mock(side_effect=AssertionError("runner created"))
    monkeypatch.setattr(AgentRunner, "from_config", create)
    assert (
        await transcribe_stream.run_transcription(
            config_path=config, audio_path=tmp_path / "missing.wav", submit=True
        )
        is None
    )
    create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["doubao_asr", "dashscope_funasr"])
@pytest.mark.parametrize("outcome", ["empty", "error", "cancel"])
async def test_empty_failed_or_cancelled_recognition_never_creates_agent(
    adapter: str,
    outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, adapter)
    audio, _ = _wav(tmp_path)
    closers = _track_files(monkeypatch)
    socket, _, waiting = _wire(
        monkeypatch, adapter, final_text="", fail=outcome == "error", block=outcome == "cancel"
    )
    create = Mock(side_effect=AssertionError("runner created"))
    monkeypatch.setattr(AgentRunner, "from_config", create)
    task = asyncio.create_task(
        transcribe_stream.run_transcription(config_path=config, audio_path=audio, submit=True)
    )
    if outcome == "cancel":
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(IrisSpeechError):
            await task
    else:
        assert await task is None
    create.assert_not_called()
    assert socket.closed and all(close.called for close in closers)


@pytest.mark.asyncio
async def test_agent_failure_keeps_transcript_and_closes_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, "doubao_asr")
    audio, _ = _wav(tmp_path)
    socket, _, _ = _wire(monkeypatch, "doubao_asr")
    provider = StaticProvider()
    runners, _ = _runners(monkeypatch, provider, lambda: None)
    result = await transcribe_stream.run_transcription(
        config_path=config, audio_path=audio, submit=True
    )
    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None
    assert "最终文字" in capsys.readouterr().out
    assert len(provider.requests) == 1 and socket.closed
    runners[0].aclose.assert_awaited_once()


@pytest.mark.parametrize("succeed", [False, True])
def test_main_outputs_complete_run_result_and_status(
    succeed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, "doubao_asr")
    audio, _ = _wav(tmp_path)
    _wire(monkeypatch, "doubao_asr")
    provider = StaticProvider(*([text_response("Agent 回答")] if succeed else []))
    runners, _ = _runners(monkeypatch, provider, lambda: None)
    exit_code = transcribe_stream.main(
        ["--config", str(config), "--audio", str(audio), "--submit", "--session-id", "main-test"]
    )
    assert exit_code == (0 if succeed else 1)
    output = capsys.readouterr().out
    assert "最终文字" in output
    assert f'"stop_reason": "{"completed" if succeed else "failed"}"' in output
    assert '"session_id": "main-test"' in output
    runners[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("channels, rate, width", [(2, 16000, 2), (1, 8000, 2), (1, 16000, 1)])
async def test_host_rejects_wrong_wav_format_and_closes_file(
    channels: int,
    rate: int,
    width: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _wav(tmp_path, channels=channels, rate=rate, width=width)
    closers = _track_files(monkeypatch)
    with pytest.raises(IrisSpeechError):
        async with aclosing(transcribe_stream.pcm_chunks(path)) as chunks:
            await anext(chunks)
    assert all(close.called for close in closers)


@pytest.mark.asyncio
async def test_wav_pacing_accounts_for_elapsed_processing_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, pcm = _wav(tmp_path)
    monotonic = Mock(side_effect=[10.0, 10.025, 10.08])
    sleep = AsyncMock()
    monkeypatch.setattr(transcribe_stream, "monotonic", monotonic)
    monkeypatch.setattr(transcribe_stream, "sleep", sleep)
    assert b"".join([chunk async for chunk in transcribe_stream.pcm_chunks(path)]) == pcm
    assert [call.args[0] for call in sleep.await_args_list] == pytest.approx([0.075, 0.020125])
