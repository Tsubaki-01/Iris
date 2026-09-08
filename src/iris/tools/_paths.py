"""工具包共用的私有路径片段处理。

外部调用标识在进入 artifact 路径前统一投影为单个安全文件名片段。

Example:
    segment = safe_path_segment("call/1")
"""

from __future__ import annotations


def safe_path_segment(value: str) -> str:
    """编码完整 UTF-8 ID，使大小写不敏感的文件系统也不会合并不同 ID。"""
    return f"id_{value.encode('utf-8').hex()}"
