"""Skill 已解析路径的轻量 containment 判断。

Example:
    is_resolved_within(skill_file, skill_root)
"""

from __future__ import annotations

# region imports
from pathlib import Path

# endregion


def is_resolved_within(path: Path, boundary: Path) -> bool:
    """判断已解析路径是否位于已解析边界内。"""
    try:
        path.relative_to(boundary)
    except ValueError:
        return False
    return True


__all__ = ["is_resolved_within"]
