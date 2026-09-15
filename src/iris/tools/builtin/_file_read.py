"""按行和行内字符位置读取有界文本页，并生成模型可见的继续位置。"""

from __future__ import annotations

from typing import TextIO

from ...exceptions import IrisToolExecutionError


def _position_reader(handle: TextIO, offset: int, column: int) -> None:
    """分块跳过前置行和字符，不将超长行整体载入内存。"""
    for _ in range(offset):
        while True:
            text = handle.readline(8192)
            if not text:
                return
            if text.endswith("\n"):
                break
    remaining = column
    while remaining:
        text = handle.readline(min(remaining, 8192))
        if not text or text.endswith("\n"):
            raise IrisToolExecutionError("COLUMN_OUT_OF_RANGE: column 超出起始行的字符范围")
        remaining -= len(text)


def _page_footer(
    offset: int, column: int, next_offset: int, next_column: int, has_more: bool
) -> str:
    """用同一格式声明零基起点、结束位置和剩余内容。"""
    return (
        f"\n\n[read_file: offset={offset}, column={column}; "
        f"next_offset={next_offset}, next_column={next_column}; "
        f"has_more={str(has_more).lower()}]"
    )


def read_text_page(
    handle: TextIO,
    *,
    offset: int,
    column: int,
    limit: int,
    with_line_numbers: bool,
    max_chars: int,
) -> str:
    """读取预算内的源文本，在正文中返回实际继续位置。

    保留文本流解码后的换行；行内偏移按 Unicode 字符计数。
    最多额外观察一个字符判断是否结束，不扫描全文统计行数。

    Args:
        handle: 已通过文件边界检查的 UTF-8 文本流。
        offset: 零基起始行偏移。
        column: 起始行内的零基字符偏移。
        limit: 最多读取的逻辑行数，零表示只查询当前位置。
        with_line_numbers: 是否为每行片段附加行号。
        max_chars: 包含行号和继续提示的最终正文预算。

    Returns:
        文件片段与可见的分页提示。

    Raises:
        IrisToolExecutionError: 预算不足以返回一页，或 column 超出起始行范围。
    """
    # 按最大可能坐标预留提示空间，避免后续通用截断吞掉正文或游标。
    footer_budget = len(_page_footer(offset, column, offset + limit, column + max_chars, False))
    remaining = max_chars - footer_budget
    max_prefix = len(f"L{offset + limit + 1:04d} | ") if with_line_numbers else 0
    minimum = max_prefix + 1 if limit else 0
    if remaining < minimum:
        raise IrisToolExecutionError("READ_BUDGET_TOO_SMALL: 结果预算不足以容纳文本和续读提示")

    _position_reader(handle, offset, column)
    next_offset, next_column = offset, column
    parts: list[str] = []
    for _ in range(limit):
        prefix = f"L{next_offset + 1:04d} | " if with_line_numbers else ""
        if remaining <= len(prefix):
            break
        text = handle.readline(remaining - len(prefix))
        if not text:
            break
        parts.append(prefix + text)
        remaining -= len(prefix) + len(text)
        if text.endswith("\n"):
            next_offset += 1
            next_column = 0
        else:
            next_column += len(text)
            break
    has_more = bool(handle.read(1))
    return "".join(parts) + _page_footer(offset, column, next_offset, next_column, has_more)
