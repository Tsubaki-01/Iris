"""FTS 索引与普通文本查询共用的词法处理。"""

from __future__ import annotations

import re

_SEGMENTS = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize_text(text: str) -> list[str]:
    """生成保留出现顺序和重复频次的英文词或中文双字片段。"""
    terms: list[str] = []
    for match in _SEGMENTS.finditer(text):
        segment = match.group()
        if segment.isascii():
            terms.append(segment.lower())
        elif len(segment) == 1:
            terms.append(segment)
        else:
            terms.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return terms


def prepare_fts_query(text: str, *, max_query_terms: int | None = None) -> str:
    """把普通文本转换为字面量 OR 查询，可选按首尾不同词项分配预算。"""
    occurrences = tokenize_text(text)
    terms = list(dict.fromkeys(occurrences))
    if max_query_terms is not None and len(terms) > max_query_terms:
        head_count = max_query_terms // 2
        tail_count = max_query_terms - head_count
        tail = list(dict.fromkeys(reversed(occurrences)))[:tail_count]
        terms = list(dict.fromkeys([*terms[:head_count], *reversed(tail)]))
    return " OR ".join(f'"{term}"' for term in terms)


__all__ = ["prepare_fts_query", "tokenize_text"]
