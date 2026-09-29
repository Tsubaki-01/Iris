"""Native 与 Docker 共用的有限输出保留，不阻止继续排空管道。"""

from .models import CommandOutputStats, OutputTruncationReason


class _StreamBuffer:
    """一条流的固定头尾，连续部分只在展示时解码。"""

    def __init__(self, limit: int) -> None:
        self.head_limit = limit // 2
        self.tail_limit = limit - self.head_limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def append(self, data: bytes) -> None:
        """保留首次头部和最新尾部，不按输入 chunk 大小扩张缓冲。"""
        self.total += len(data)
        head_bytes = min(len(data), self.head_limit - len(self.head))
        self.head.extend(data[:head_bytes])
        if len(data) - head_bytes >= self.tail_limit:
            self.tail[:] = data[-self.tail_limit :] if self.tail_limit else b""
        else:
            self.tail.extend(data[head_bytes:])
            if len(self.tail) > self.tail_limit:
                del self.tail[: len(self.tail) - self.tail_limit]

    @property
    def retained(self) -> int:
        """返回当前仍持有的原始字节数。"""
        return len(self.head) + len(self.tail)

    @property
    def text(self) -> str:
        """分别解码缺口两侧，连续字节则合并解码。"""
        if self.total <= self.retained:
            return (self.head + self.tail).decode("utf-8", errors="replace")
        return (
            self.head.decode("utf-8", errors="replace")
            + "\n\n[中间输出已省略]\n\n"
            + self.tail.decode("utf-8", errors="replace")
        )


class OutputBuffer:
    """stdout/stderr 固定各占总预算一半，不借用另一条流的额度。"""

    def __init__(self, limit: int = 1024 * 1024) -> None:
        self._stdout = _StreamBuffer(limit // 2)
        self._stderr = _StreamBuffer(limit // 2)
        self._reasons: set[OutputTruncationReason] = set()

    def append(self, stream: int, data: bytes) -> None:
        """累计采集字节并保留头尾；调用方继续读取其余输出。"""
        buffer = self._stdout if stream == 1 else self._stderr
        buffer.append(data)
        if buffer.total > buffer.retained:
            self._reasons.add("byte_limit")

    def mark_truncated(self, reason: OutputTruncationReason) -> None:
        """登记后端实际观察到的不完整输出原因。"""
        self._reasons.add(reason)

    @property
    def stats(self) -> CommandOutputStats:
        """投影当前采集统计，不重复累加或访问原始流。"""
        return CommandOutputStats(
            stdout_bytes=self._stdout.total,
            stderr_bytes=self._stderr.total,
            stdout_retained_bytes=self._stdout.retained,
            stderr_retained_bytes=self._stderr.retained,
            truncation_reasons=frozenset(self._reasons),
        )

    @property
    def truncated(self) -> bool:
        """由原因集合派生是否截断。"""
        return bool(self._reasons)

    @property
    def stdout(self) -> str:
        """以 UTF-8 replacement 解码已经保留的标准输出。"""
        return self._stdout.text

    @property
    def stderr(self) -> str:
        """以 UTF-8 replacement 解码已经保留的标准错误。"""
        return self._stderr.text
