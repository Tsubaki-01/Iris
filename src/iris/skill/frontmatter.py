"""SKILL.md frontmatter 的纯解析函数。"""

from __future__ import annotations

from typing import Any

import yaml

from ..exceptions import IrisSkillFormatError

_FRONTMATTER_DELIMITER = "---"


def _line_content(line: str) -> str:
    """移除单行的 LF/CRLF 结尾，保留其他字符。"""
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def split_frontmatter(text: str) -> tuple[str, str]:
    """把完整 SKILL.md 切分为 frontmatter 与 Markdown 正文。"""
    lines = text.splitlines(keepends=True)
    if not lines or _line_content(lines[0]) != _FRONTMATTER_DELIMITER:
        return "", text

    for index, line in enumerate(lines[1:], start=1):
        if _line_content(line) == _FRONTMATTER_DELIMITER:
            return "".join(lines[1:index]), "".join(lines[index + 1 :])

    raise IrisSkillFormatError("SKILL.md frontmatter 缺少闭合分隔符")


def parse_frontmatter(text: str) -> dict[str, Any]:
    """安全解析不含 delimiter 的 YAML frontmatter。"""
    if not text.strip():
        return {}

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise IrisSkillFormatError(
            "SKILL.md frontmatter YAML 解析失败",
            reason=str(exc),
        ) from exc

    if not isinstance(parsed, dict):
        raise IrisSkillFormatError("SKILL.md frontmatter 顶层必须是对象")
    if any(not isinstance(key, str) for key in parsed):
        raise IrisSkillFormatError("SKILL.md frontmatter 键必须是字符串")
    return parsed


__all__ = ["parse_frontmatter", "split_frontmatter"]
