"""保存实际读取过的配置来源，不为描述快照重新访问文件。"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Literal
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """实际来源文本或明确的不可展示状态。"""

    kind: str
    original_path: str | None
    text: str | None
    status: Literal["captured", "redacted", "not_utf8"] = "captured"
    document_id: str = field(default_factory=lambda: f"source_{uuid4().hex}")


_SOURCES: ContextVar[list[SourceDocument] | None] = ContextVar(
    "iris_configuration_sources", default=None
)


def capture_source_reads[**P, R](operation: Callable[P, R]) -> Callable[P, R]:
    """让同步装配的多个真实 loader 共用局部来源集合。"""

    @wraps(operation)
    def capture(*args: P.args, **kwargs: P.kwargs) -> R:
        if _SOURCES.get() is not None:
            return operation(*args, **kwargs)
        token = _SOURCES.set([])
        try:
            return operation(*args, **kwargs)
        finally:
            _SOURCES.reset(token)

    return capture


def capture_document(document: SourceDocument) -> None:
    """记录本次真实读取，普通独立 loader 不保留全局历史。"""
    sources = _SOURCES.get()
    if sources is not None:
        sources.append(document)


def captured_documents() -> tuple[SourceDocument, ...]:
    """返回当前装配已经读入的来源。"""
    return tuple(_SOURCES.get() or ())


def read_source_text(path: Path, *, kind: str, encoding: str = "utf-8") -> str:
    """在原文件读取边界保留同一份正文，错误仍由调用 loader 归属。"""
    text = path.read_text(encoding=encoding)
    capture_document(SourceDocument(kind, str(path.resolve()), text))
    return text


__all__ = [
    "SourceDocument",
    "capture_source_reads",
    "captured_documents",
    "capture_document",
    "read_source_text",
]
