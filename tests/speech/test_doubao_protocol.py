"""豆包官方二进制帧格式与外部响应解析契约。"""

from __future__ import annotations

import gzip
import json
import struct

import pytest

from iris.exceptions import IrisSpeechError
from iris.speech.adapters._doubao_protocol import (
    DoubaoResponse,
    decode_response,
    encode_audio,
    encode_request,
)


def server_frame(
    payload: bytes,
    *,
    flags: int = 1,
    compression: int = 0,
    sequence: int = 1,
    extended_header: bool = False,
) -> bytes:
    """根据官方响应布局独立构建 fixture，不调用待测编码函数。"""
    header = bytes((0x12 if extended_header else 0x11, 0x90 | flags, 0x10 | compression, 0))
    if extended_header:
        header += b"\x00\x00\x00\x00"
    fields = struct.pack(">i", sequence) if flags & 1 else b""
    if flags & 4:
        fields += struct.pack(">i", 150)
    data = gzip.compress(payload) if compression == 1 else payload
    return header + fields + struct.pack(">I", len(data)) + data


def test_request_matches_current_demo_header_and_payload() -> None:
    frame = encode_request()
    assert frame[:8] == b"\x11\x11\x11\x00\x00\x00\x00\x01"
    assert struct.unpack(">I", frame[8:12])[0] == len(frame[12:])
    assert json.loads(gzip.decompress(frame[12:])) == {
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


@pytest.mark.parametrize("sequence, final", [(2, False), (3, True), (2, True)])
def test_audio_preserves_bytes_and_marks_only_final_sequence(sequence: int, final: bool) -> None:
    audio = b"\x00\x01\xff\xfe"
    frame = encode_audio(audio, sequence, final=final)
    assert frame[:4] == (b"\x11\x23\x11\x00" if final else b"\x11\x21\x11\x00")
    assert struct.unpack(">i", frame[4:8])[0] == (-sequence if final else sequence)
    assert struct.unpack(">I", frame[8:12])[0] == len(frame[12:])
    assert gzip.decompress(frame[12:]) == audio


@pytest.mark.parametrize("compression", [0, 1])
@pytest.mark.parametrize("flags", [0, 1, 2, 3, 7])
def test_response_uses_header_flags_and_optional_fields(compression: int, flags: int) -> None:
    frame = server_frame(
        b'{"result":{"text":"hello","utterances":[{"definite":true}]}}',
        flags=flags,
        compression=compression,
        sequence=-8,
        extended_header=True,
    )
    assert decode_response(frame, request_id="request-1") == DoubaoResponse(
        text="hello", is_final=bool(flags & 2)
    )


@pytest.mark.parametrize("payload", [b"{}", b'{"audio_info":{"duration":100}}', b'{"result":{}}'])
@pytest.mark.parametrize("flags", [1, 3])
def test_state_frame_without_text_does_not_erase_previous_snapshot(
    payload: bytes, flags: int
) -> None:
    assert decode_response(server_frame(payload, flags=flags), request_id="request-1") == (
        DoubaoResponse(text=None, is_final=bool(flags & 2))
    )


def test_empty_transcript_is_an_explicit_text_update() -> None:
    frame = server_frame(b'{"result":{"text":""}}', flags=3)
    assert decode_response(frame, request_id="request-1") == DoubaoResponse("", True)


@pytest.mark.parametrize("compression", [0, 1])
@pytest.mark.parametrize("serialization", [0, 1])
def test_error_frame_keeps_vendor_code_message_and_request_context(
    compression: int, serialization: int
) -> None:
    payload = b'{"message":"quota exceeded"}' if serialization else b"quota exceeded"
    data = gzip.compress(payload) if compression else payload
    frame = (
        bytes((0x11, 0xF0, serialization << 4 | compression, 0))
        + struct.pack(">II", 45000030, len(data))
        + data
    )
    with pytest.raises(IrisSpeechError) as caught:
        decode_response(frame, request_id="request-error")
    assert caught.value.context["adapter"] == "doubao_asr"
    assert caught.value.context["request_id"] == "request-error"
    assert caught.value.context["provider_code"] == 45000030
    assert "quota exceeded" in str(caught.value.context["provider_message"])


@pytest.mark.parametrize(
    "payload",
    [b"[]", b'{"result":null}', b'{"result":[]}', b'{"result":{"text":3}}', b"not json"],
)
def test_invalid_json_shapes_fail_instead_of_returning_empty_final(payload: bytes) -> None:
    with pytest.raises(IrisSpeechError) as caught:
        decode_response(server_frame(payload, flags=3), request_id="request-invalid")
    assert caught.value.context["request_id"] == "request-invalid"


@pytest.mark.parametrize(
    "frame",
    [
        "unexpected text frame",
        b"\x11\x91",
        b"\x10\x91\x10\x00",
        b"\x11\x81\x10\x00\x00\x00\x00\x01\x00\x00\x00\x02{}",
        b"\x11\x91\x12\x00\x00\x00\x00\x01\x00\x00\x00\x02{}",
        b"\x11\x91\x20\x00\x00\x00\x00\x01\x00\x00\x00\x02{}",
        b"\x11\x91\x10\x00\x00\x00\x00\x01\x00\x00\x00\x03{}",
        b"\x11\x91\x10\x00\x00\x00\x00\x01\x00\x00\x00\x01{}",
        b"\x11\x97\x10\x00\x00\x00\x00\x01",
        b"\x11\x91\x11\x00\x00\x00\x00\x01\x00\x00\x00\x02{}",
        b"\x11\xf0\x10\x00\x00\x00\x00\x01",
    ],
)
def test_malformed_binary_frames_are_domain_errors(frame: bytes | str) -> None:
    with pytest.raises(IrisSpeechError) as caught:
        decode_response(frame, request_id="request-invalid")
    assert caught.value.context["adapter"] == "doubao_asr"
    assert caught.value.context["request_id"] == "request-invalid"
