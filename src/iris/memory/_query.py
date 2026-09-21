"""FTS 索引与普通文本查询共用的词法处理。"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence, Set

_SEGMENTS = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def iter_token_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """按原文顺序产生索引词及 Python 字符起止位置。"""
    for match in _SEGMENTS.finditer(text):
        segment = match.group()
        start, end = match.span()
        if segment.isascii():
            yield segment.lower(), start, end
        elif len(segment) == 1:
            yield segment, start, end
        else:
            for index in range(len(segment) - 1):
                yield segment[index : index + 2], start + index, start + index + 2


def tokenize_text(text: str) -> list[str]:
    """生成保留出现顺序和重复频次的英文词或中文双字片段。"""
    return [term for term, _, _ in iter_token_spans(text)]


def prepare_fts_query(terms: Sequence[str], required_terms: Sequence[str] = ()) -> str:
    """将普通 OR 查询与有序且保留重复词项的必要词组取交集。"""
    query = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))
    if not required_terms:
        return query
    phrases = [f'"{" ".join(tokenize_text(phrase))}"' for phrase in required_terms]
    return f"({query}) AND " + " AND ".join(phrases)


def make_snippet(text: str, query_terms: Set[str]) -> tuple[str, bool]:
    """返回全文或首个命中词附近的连续 300 字符原文窗口。"""
    if len(text) <= 300:
        return text, True
    hit_start = next(start for term, start, _ in iter_token_spans(text) if term in query_terms)
    start = max(0, min(hit_start - 150, len(text) - 300))
    return text[start : start + 300], False


__all__ = ["iter_token_spans", "make_snippet", "prepare_fts_query", "tokenize_text"]
