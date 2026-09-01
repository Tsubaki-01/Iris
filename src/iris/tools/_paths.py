"""工具包共用的私有路径片段处理。

外部调用标识在进入 artifact 路径前统一投影为单个安全文件名片段。

Example:
    segment = safe_path_segment("call/1")
"""

from __future__ import annotations

import re


def safe_path_segment(value: str) -> str:
    """把外部 ID 投影为单个安全路径片段。"""
    segment = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return segment.strip("_") or "default"
