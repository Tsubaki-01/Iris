"""双流固定头尾预算与原始字节统计。"""

from iris.command._output import OutputBuffer


def test_streams_keep_independent_heads_and_tails() -> None:
    output = OutputBuffer(limit=12)
    output.append(1, b"abc123456XYZ")
    output.append(2, b"error")
    assert output.stdout.startswith("abc") and output.stdout.endswith("XYZ")
    assert "省略" in output.stdout
    assert output.stderr == "error"
    assert output.stats.stdout_bytes == 12
    assert output.stats.stdout_retained_bytes == 6
    assert output.stats.stderr_bytes == output.stats.stderr_retained_bytes == 5
    assert output.stats.truncation_reasons == frozenset({"byte_limit"})


def test_exact_budget_has_no_duplicate_or_false_truncation() -> None:
    output = OutputBuffer(limit=12)
    for data in (b"ab", b"", b"cdef"):
        output.append(1, data)
    assert output.stdout == "abcdef"
    assert not output.truncated
    output.append(1, b"gh")
    assert output.stdout.startswith("abc") and output.stdout.endswith("fgh")
    assert output.stats.stdout_bytes == 8
    assert output.stats.stdout_retained_bytes == 6


def test_utf8_chunks_without_gap_are_decoded_together() -> None:
    output = OutputBuffer(limit=12)
    data = "中文".encode()
    output.append(1, data[:2])
    output.append(1, data[2:])
    output.append(2, b"\xff")
    assert output.stdout == "中文"
    assert output.stderr == "\ufffd"
    assert not output.truncated


def test_gap_does_not_join_utf8_bytes_and_reasons_accumulate() -> None:
    output = OutputBuffer(limit=8)
    output.append(1, "中A文".encode())
    output.mark_truncated("drain_timeout")
    output.mark_truncated("stream_error")
    output.mark_truncated("stream_closed")
    assert output.stdout.count("\ufffd") == 3
    assert output.stats.stdout_bytes == 7
    assert output.stats.stdout_retained_bytes == 4
    assert output.stats.truncation_reasons == frozenset(
        {"byte_limit", "drain_timeout", "stream_error", "stream_closed"}
    )


def test_single_stream_does_not_borrow_other_stream_budget() -> None:
    output = OutputBuffer()
    output.append(1, b"x" * (1024 * 1024))
    output.append(2, b"late traceback")
    assert output.stats.stdout_retained_bytes == 512 * 1024
    assert output.stderr == "late traceback"
