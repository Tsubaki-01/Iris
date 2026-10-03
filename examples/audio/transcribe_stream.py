"""按 YAML 选择语音服务转录 WAV，可显式将最终文字交给 Agent。"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from asyncio import sleep
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from pathlib import Path
from time import monotonic
from uuid import uuid4

from iris.agents import load_agent_config
from iris.config import init_config, is_config_initialized
from iris.exceptions import IrisSpeechError
from iris.harness import AgentRunner, AgentRunRequest, RunResult
from iris.lifecycle import RunStopReason
from iris.speech import create_speech_client


async def pcm_chunks(path: Path) -> AsyncGenerator[bytes, None]:
    """在宿主拥有的文件作用域中读取 PCM，并按累计音频时长发送。

    Args:
        path: PCM、16 kHz、16-bit、单声道 WAV 文件。

    Yields:
        100 ms 音频块，最后一块可以更短，不包含 WAV 文件头。

    Raises:
        IrisSpeechError: WAV 格式无法读取或音频参数不符。
    """
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getcomptype(),
                source.getframerate(),
                source.getsampwidth(),
                source.getnchannels(),
            ) != ("NONE", 16000, 2, 1):
                raise IrisSpeechError("WAV 必须为 PCM、16 kHz、16-bit、单声道", path=str(path))
            started = monotonic()
            duration = 0.0
            while chunk := source.readframes(1600):
                duration += len(chunk) / 32000
                delay = started + duration - monotonic()
                if delay > 0:
                    await sleep(delay)
                yield chunk
    except (wave.Error, EOFError) as exc:
        raise IrisSpeechError("无法读取 WAV 音频", path=str(path)) from exc


async def run_transcription(
    *,
    config_path: Path,
    audio_path: Path,
    submit: bool = False,
    session_id: str = "voice-input",
) -> RunResult | None:
    """运行同一个转录入口，仅在显式提交时创建并关闭 Agent runner。

    Args:
        config_path: 完整 Agent YAML 路径，保留用于 runner 相对路径解析。
        audio_path: 音频文件路径；相对路径以当前工作目录解析。
        submit: 是否将正常收尾后的非空最终文字作为普通用户输入提交。
        session_id: 显式提交时使用的会话标识。

    Returns:
        提交后的完整 RunResult；关闭语音、仅转录或无文字时返回 None。
    """
    config = load_agent_config(config_path)
    client = create_speech_client(config.speech)
    if client is None:
        print("语音功能已关闭。")
        return None

    final_text = ""
    async with aclosing(pcm_chunks(audio_path)) as audio, aclosing(client.stream(audio)) as events:
        async for event in events:
            label = "最终转录" if event.is_final else "当前转录"
            print(f"[{label}] {event.text}", flush=True)
            if event.is_final:
                final_text = event.text
    if not submit or not final_text.strip():
        return None

    runner = AgentRunner.from_config(config, config_path=config_path)
    try:
        return await runner.start(AgentRunRequest(input=final_text, session_id=session_id))
    finally:
        await runner.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """输出转录快照，显式提交时同时输出完整 RunResult JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--session-id", default=f"voice-{uuid4().hex}")
    args = parser.parse_args(argv)
    if not is_config_initialized():
        init_config(env_file=str(args.env_file) if args.env_file is not None else None)
    result = asyncio.run(
        run_transcription(
            config_path=args.config,
            audio_path=args.audio,
            submit=args.submit,
            session_id=args.session_id,
        )
    )
    if result is None:
        return 0
    print(result.model_dump_json(indent=2))
    return 0 if result.run.stop_reason is RunStopReason.COMPLETED else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
