"""Lifecycle 人工 smoke 使用的最小同步工具。"""

from __future__ import annotations

import sys
import time


def wait_for_seconds(seconds: int) -> str:
    """打印 durable-claim 后的 smoke 标记，阻塞指定秒数并返回完成文本。"""
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 60:
        raise ValueError("seconds 必须是 1 到 60 之间的整数")
    print(
        f"IRIS_LIFECYCLE_SMOKE_TOOL_STARTED seconds={seconds}",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(seconds)
    return f"waited {seconds} seconds"


__all__ = ["wait_for_seconds"]
