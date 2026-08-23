"""P1 文件遍历与逐行 grep 的本机前后观测。"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO

import pytest

from iris.tools import (
    GrepSearchInput,
    ListFilesInput,
    ToolExecutionContext,
    WorkspaceFileService,
)


class _CountingScandir:
    """为真实 scandir iterator 增加测试侧 entry 计数。"""

    def __init__(self, iterator: os.ScandirIterator[str], counters: dict[str, int]) -> None:
        self._iterator = iterator
        self._counters = counters

    def __enter__(self) -> _CountingScandir:
        self._iterator.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._iterator.__exit__(*args)

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        for entry in self._iterator:
            self._counters["entries_yielded"] += 1
            yield entry

    def close(self) -> None:
        self._iterator.close()


class _CountingReader:
    """统计真实文本文件被消费的行数。"""

    def __init__(self, handle: TextIO, counters: dict[str, int]) -> None:
        self._handle = handle
        self._counters = counters

    def __enter__(self) -> _CountingReader:
        self._handle.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._handle.__exit__(*args)

    def __iter__(self) -> _CountingReader:
        return self

    def __next__(self) -> str:
        line = next(self._handle)
        self._counters["lines_consumed"] += 1
        return line

    def read(self, *args: Any, **kwargs: Any) -> str:
        content = self._handle.read(*args, **kwargs)
        self._counters["lines_consumed"] += len(content.splitlines())
        return content

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


@pytest.mark.performance_timing
def test_p1_list_files_early_stop_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    """记录 max_results=1 时实际枚举的目录项数量。"""
    file_count = 2000
    for index in range(file_count):
        (tmp_path / f"{index:04d}.txt").write_text("x", encoding="utf-8")

    original_scandir = os.scandir
    counters = {"scandir_calls": 0, "entries_yielded": 0}

    def counting_scandir(path: str | os.PathLike[str]) -> _CountingScandir:
        counters["scandir_calls"] += 1
        return _CountingScandir(original_scandir(path), counters)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    service = WorkspaceFileService()
    context = ToolExecutionContext(workspace_root=tmp_path)
    samples: list[float] = []
    for _ in range(5):
        started_at = time.perf_counter()
        result = service.list_files(
            ListFilesInput(max_results=1),
            context,
        )
        samples.append((time.perf_counter() - started_at) * 1000)
        assert len(result.splitlines()) == 1

    assert counters == {"scandir_calls": 5, "entries_yielded": 5}

    record_observation(
        scenario="p1_list_files_first_result",
        perf_ids=("PERF-02", "PERF-14"),
        fixture={"file_count": file_count, "max_results": 1, "shape": "flat"},
        samples_ms=tuple(samples),
        counters=counters,
    )


@pytest.mark.performance_timing
def test_p1_grep_first_line_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    """记录首行命中时实际消费的文本行数。"""
    line_count = 50000
    target = tmp_path / "large.txt"
    target.write_text(
        "needle\n" + "".join(f"line {index}\n" for index in range(line_count - 1)),
        encoding="utf-8",
    )
    original_open = Path.open
    counters = {"open_calls": 0, "lines_consumed": 0}

    def counting_open(path: Path, *args: Any, **kwargs: Any) -> TextIO | _CountingReader:
        handle = original_open(path, *args, **kwargs)
        if path.resolve() != target.resolve():
            return handle
        counters["open_calls"] += 1
        return _CountingReader(handle, counters)

    monkeypatch.setattr(Path, "open", counting_open)
    service = WorkspaceFileService()
    context = ToolExecutionContext(workspace_root=tmp_path)
    samples: list[float] = []
    for _ in range(5):
        started_at = time.perf_counter()
        result = service.grep_search(
            GrepSearchInput(pattern="needle", path=str(target), max_results=1),
            context,
        )
        samples.append((time.perf_counter() - started_at) * 1000)
        assert result == "large.txt:1: needle"

    assert counters == {"open_calls": 5, "lines_consumed": 5}

    record_observation(
        scenario="p1_grep_first_line",
        perf_ids=("PERF-02", "PERF-14"),
        fixture={"line_count": line_count, "max_results": 1, "shape": "single_file"},
        samples_ms=tuple(samples),
        counters=counters,
    )
