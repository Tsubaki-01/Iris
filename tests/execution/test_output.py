"""两个管道共享字节预算，截断后仍可持续消费。"""

from iris.execution._output import OutputBuffer


def test_streams_share_one_byte_limit() -> None:
    output = OutputBuffer(limit=6)
    output.append(1, b"out")
    output.append(2, b"error")
    output.append(1, b"ignored")
    assert output.stdout == "out"
    assert output.stderr == "err"
    assert output.truncated


def test_exact_budget_is_not_truncated_until_data_is_dropped() -> None:
    output = OutputBuffer(limit=3)
    output.append(1, b"abc")
    output.append(2, b"")
    assert not output.truncated
    output.append(2, b"d")
    assert output.truncated


def test_utf8_chunks_are_decoded_after_collection() -> None:
    output = OutputBuffer()
    data = "中文".encode()
    output.append(1, data[:2])
    output.append(1, data[2:])
    output.append(2, b"\xff")
    assert output.stdout == "中文"
    assert output.stderr == "\ufffd"


def test_partial_utf8_and_incomplete_drain_are_explicit() -> None:
    output = OutputBuffer(limit=2)
    output.append(1, "中".encode())
    assert output.stdout == "\ufffd"
    assert output.truncated

    output = OutputBuffer()
    output.append(1, b"partial")
    output.mark_truncated()
    assert output.stdout == "partial"
    assert output.truncated
