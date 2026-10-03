"""阿里 DashScope Fun-ASR 的 task 协议与句子转录聚合。"""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..models import TranscriptionEvent
from ._transport import WebSocketTransport, open_transport, received_messages

_START_TIMEOUT = 10.0
_FINAL_TIMEOUT = 10.0


class _Sentence(BaseModel):
    """仅在服务响应边界解析一次的原生句子。"""

    model_config = ConfigDict(frozen=True, strict=True)

    heartbeat: bool = False
    sentence_id: int = Field(ge=0)
    text: str
    sentence_end: bool

    @model_validator(mode="after")
    def _validate_identifier(self) -> Self:
        if not self.heartbeat and self.sentence_id == 0:
            raise ValueError("非心跳句子的 sentence_id 必须从 1 开始")
        return self


type _Event = Literal["task-started", "task-finished"] | _Sentence


def _decode(frame: bytes | str, transport: WebSocketTransport) -> _Event:
    if not isinstance(frame, str):
        raise transport.error("Fun-ASR 响应必须为 JSON 文本帧")
    try:
        raw = json.loads(frame)
        header = raw["header"]
        name = header["event"]
        if header["task_id"] != transport.request_id:
            raise transport.error("Fun-ASR 响应 task_id 与当前任务不符")
        if name == "task-failed":
            raise transport.error(
                "Fun-ASR 任务失败",
                provider_code=header["error_code"],
                provider_message=header["error_message"],
            )
        if name == "result-generated":
            return _Sentence.model_validate(raw["payload"]["output"]["sentence"])
        if name == "task-started":
            return "task-started"
        if name == "task-finished":
            return "task-finished"
        raise transport.error("Fun-ASR 响应事件不受支持", event=name)
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise transport.error("无法解析 Fun-ASR 服务响应") from exc


def _task_message(action: str, task_id: str, payload: dict[str, object]) -> str:
    return json.dumps(
        {
            "header": {"action": action, "task_id": task_id, "streaming": "duplex"},
            "payload": payload,
        }
    )


def _text(sentences: dict[int, _Sentence], *, confirmed: bool = False) -> str:
    return "\n".join(
        item.text
        for _, item in sorted(sentences.items())
        if item.text and (not confirmed or item.sentence_end)
    )


async def _send_audio(
    transport: WebSocketTransport, audio: AsyncIterable[bytes], ending: asyncio.Event
) -> float:
    async for chunk in audio:
        await transport.send(chunk)
    ending.set()
    await transport.send(_task_message("finish-task", transport.request_id, {"input": {}}))
    return asyncio.get_running_loop().time()


@dataclass(frozen=True, slots=True, kw_only=True)
class DashScopeFunASRAdapter:
    """保存 Fun-ASR 参数，每次识别独立使用一个 task 和连接。

    Args:
        endpoint: 指定地域和业务空间的完整 WebSocket 地址。
        model: Fun-ASR-Realtime 模型名称。
        api_key: 阿里百炼 API Key。
    """

    endpoint: str
    model: str
    api_key: str = field(repr=False)

    async def stream(self, audio: AsyncIterable[bytes]) -> AsyncGenerator[TranscriptionEvent, None]:
        """就绪后发送 PCM，并将句子更新投影为统一全文事件。"""
        task_id = str(uuid4())
        async with open_transport(
            self.endpoint,
            {"Authorization": f"Bearer {self.api_key}"},
            adapter="dashscope_funasr",
            request_id=task_id,
        ) as transport:
            await transport.send(
                _task_message(
                    "run-task",
                    task_id,
                    {
                        "task_group": "audio",
                        "task": "asr",
                        "function": "recognition",
                        "model": self.model,
                        "parameters": {"format": "pcm", "sample_rate": 16000, "heartbeat": True},
                        "input": {},
                    },
                )
            )
            try:
                async with asyncio.timeout(_START_TIMEOUT):
                    started = _decode(await transport.receive(), transport)
            except TimeoutError as exc:
                raise transport.error("等待 Fun-ASR task-started 超时") from exc
            if started != "task-started":
                raise transport.error("Fun-ASR 尚未确认任务启动")

            ending = asyncio.Event()
            sender = asyncio.create_task(_send_audio(transport, audio, ending))
            sentences: dict[int, _Sentence] = {}
            async with aclosing(
                received_messages(transport, sender, final_timeout=_FINAL_TIMEOUT)
            ) as messages:
                async for frame in messages:
                    response = _decode(frame, transport)
                    match response:
                        case "task-finished":
                            if not ending.is_set():
                                raise transport.error("Fun-ASR 在音频输入结束前返回终态")
                            await sender
                            yield TranscriptionEvent(
                                text=_text(sentences, confirmed=True), is_final=True
                            )
                            return
                        case "task-started":
                            raise transport.error("Fun-ASR 重复返回 task-started")
                        case _Sentence() as result:
                            if not result.heartbeat:
                                sentences[result.sentence_id] = result
                                yield TranscriptionEvent(text=_text(sentences), is_final=False)


__all__ = ["DashScopeFunASRAdapter"]
