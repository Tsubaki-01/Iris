"""豆包流式识别二进制协议的唯一编解码边界。"""

from __future__ import annotations

import gzip
import json
import struct
import zlib
from dataclasses import dataclass

from ...exceptions import IrisSpeechError


@dataclass(frozen=True, slots=True)
class DoubaoResponse:
    """已解析的全文更新及服务端末响应标志。"""

    text: str | None
    is_final: bool


def _encode_packet(payload: bytes, sequence: int, message_flags: int) -> bytes:
    compressed = gzip.compress(payload, mtime=0)
    return (
        bytes((0x11, message_flags, 0x11, 0))
        + struct.pack(">iI", sequence, len(compressed))
        + compressed
    )


def encode_request() -> bytes:
    """编码序号为 1 的固定 PCM 识别配置首包。"""
    payload = {
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
    return _encode_packet(json.dumps(payload).encode("utf-8"), 1, 0x11)


def encode_audio(chunk: bytes, sequence: int, *, final: bool = False) -> bytes:
    """编码已校验的音频，末包使用负序号并保留真实音频。"""
    return _encode_packet(chunk, -sequence if final else sequence, 0x23 if final else 0x21)


def decode_response(frame: bytes | str, *, request_id: str) -> DoubaoResponse:
    """将外部帧解析为内部响应，保留厂商错误及本次请求标识。

    Args:
        frame: WebSocket 收到的原始帧。
        request_id: 本次连接使用的请求标识。

    Returns:
        当前文字更新以及从 header flags 取得的末响应标志。

    Raises:
        IrisSpeechError: 帧损坏、字段不符或服务端返回错误帧。
    """
    context: dict[str, object] = {"adapter": "doubao_asr", "request_id": request_id}
    if isinstance(frame, str) or len(frame) < 4:
        raise IrisSpeechError("豆包响应必须包含完整二进制帧头", **context)

    version, header_words = frame[0] >> 4, frame[0] & 0x0F
    message_type, flags = frame[1] >> 4, frame[1] & 0x0F
    serialization, compression = frame[2] >> 4, frame[2] & 0x0F
    if version != 1 or header_words < 1:
        raise IrisSpeechError("豆包响应帧头版本或长度无效", **context)
    if message_type not in (0x9, 0xF):
        raise IrisSpeechError("豆包响应消息类型不受支持", message_type=message_type, **context)
    if compression not in (0, 1) or serialization not in (0, 1):
        raise IrisSpeechError("豆包响应压缩或序列化格式不受支持", **context)
    if message_type == 0x9 and serialization != 1:
        raise IrisSpeechError("豆包识别响应必须使用 JSON", **context)

    # 序号和 event 是可选线协议字段，均不决定整段识别是否完成。
    offset = header_words * 4 + (4 if flags & 1 else 0) + (4 if flags & 4 else 0)
    try:
        if message_type == 0xF:
            context["provider_code"] = struct.unpack_from(">I", frame, offset)[0]
            offset += 4
        size = struct.unpack_from(">I", frame, offset)[0]
        payload = frame[offset + 4 :]
        if len(payload) != size:
            raise IrisSpeechError("豆包响应 payload 长度不符", **context)
        if compression == 1:
            payload = gzip.decompress(payload)
        decoded = json.loads(payload) if serialization == 1 else payload.decode("utf-8")
    except (struct.error, OSError, EOFError, zlib.error, UnicodeError, json.JSONDecodeError) as exc:
        raise IrisSpeechError("无法解析豆包语音响应", **context) from exc

    if message_type == 0xF:
        raise IrisSpeechError("豆包语音服务返回错误", provider_message=decoded, **context)
    if not isinstance(decoded, dict):
        raise IrisSpeechError("豆包识别响应必须是 JSON 对象", **context)
    text: str | None = None
    if "result" in decoded:
        result = decoded["result"]
        if not isinstance(result, dict):
            raise IrisSpeechError("豆包响应 result 必须是对象", **context)
        if "text" in result:
            text = result["text"]
            if not isinstance(text, str):
                raise IrisSpeechError("豆包响应 result.text 必须是字符串", **context)
    return DoubaoResponse(text=text, is_final=bool(flags & 2))
