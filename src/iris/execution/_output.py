"""Native 与 Docker 共用的有限输出保留，不阻止继续排空管道。"""


class OutputBuffer:
    """stdout/stderr 共用一个字节预算，解码前保留跨 chunk 字符。"""

    def __init__(self, limit: int = 1024 * 1024) -> None:
        self._remaining = limit
        self._stdout = bytearray()
        self._stderr = bytearray()
        self.truncated = False

    def append(self, stream: int, data: bytes) -> None:
        """保留预算内字节；调用方继续读取其余输出。"""
        retained = data[: self._remaining]
        (self._stdout if stream == 1 else self._stderr).extend(retained)
        self._remaining -= len(retained)
        self.truncated |= len(retained) < len(data)

    def mark_truncated(self) -> None:
        """标记管道未能在收尾期限内完全排空。"""
        self.truncated = True

    @property
    def stdout(self) -> str:
        """以 UTF-8 replacement 解码已经保留的标准输出。"""
        return self._stdout.decode("utf-8", errors="replace")

    @property
    def stderr(self) -> str:
        """以 UTF-8 replacement 解码已经保留的标准错误。"""
        return self._stderr.decode("utf-8", errors="replace")
