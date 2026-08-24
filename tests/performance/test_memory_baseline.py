"""P4 Memory query 与 mirror 重建的结构计数和同机计时观测。"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.memory import (
    FileMemoryMirror,
    MemoryAccessPolicy,
    MemoryBackend,
    MemoryConfig,
    MemoryItem,
    MemoryScope,
    MemorySearchTool,
    MemorySearchToolInput,
    SQLiteMemoryStore,
    build_memory_service_from_config,
)
from iris.tools import ToolExecutionContext

_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_ROW_COUNT = 30_000
_TARGET_SCOPE = MemoryScope(workspace_id="workspace-0", agent_id="agent")


class _MirrorStore:
    def __init__(self, items: list[MemoryItem]) -> None:
        self.items = items

    def list_items(
        self,
        scope: MemoryScope,
        *,
        limit: int | None = 50,
        **_: object,
    ) -> list[MemoryItem]:
        del scope
        return self.items if limit is None else self.items[:limit]

    def list_events(
        self,
        scope: MemoryScope,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[object]:
        del scope, item_id, limit
        return []


def _seed_query_rows(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executemany(
            """INSERT INTO memory_items(
                id, scope_workspace_id, scope_agent_id, scope_collection,
                scope_visibility, scope_session_id, episode_id, level, category,
                kind, text, status, source_type, source_id, reason, confidence,
                importance, artifacts_json, metadata_json, created_at, updated_at,
                deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            _item_rows(),
        )
        connection.executemany(
            """INSERT INTO memory_events(
                id, scope_workspace_id, scope_agent_id, scope_collection,
                scope_visibility, scope_session_id, event_type, actor, item_id,
                episode_id, reason, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            _event_rows(),
        )


def _item_rows() -> Iterator[tuple[object, ...]]:
    for index in range(_ROW_COUNT):
        timestamp = (_NOW + timedelta(seconds=index)).isoformat()
        yield (
            f"mem_{index:05d}",
            f"workspace-{index % 100}",
            "agent",
            "default",
            "agent",
            "",
            None,
            "l2",
            "user",
            "fact",
            f"memory-{index}",
            "active",
            "sdk",
            f"source-{index}",
            "baseline",
            None,
            None,
            "[]",
            "{}",
            timestamp,
            timestamp,
            None,
        )


def _event_rows() -> Iterator[tuple[object, ...]]:
    for index in range(_ROW_COUNT):
        timestamp = (_NOW + timedelta(seconds=index)).isoformat()
        yield (
            f"evt_{index:05d}",
            f"workspace-{index % 100}",
            "agent",
            "default",
            "agent",
            "",
            "add",
            "sdk",
            f"mem_{index:05d}",
            None,
            "baseline",
            "{}",
            timestamp,
        )


def _query_plan_counters(path: Path, *, table: str) -> dict[str, int]:
    columns = (
        "scope_workspace_id = ? AND scope_agent_id = ? AND scope_collection = ? "
        "AND scope_visibility = ? AND scope_session_id = ?"
    )
    if table == "items":
        sql = (
            f"SELECT * FROM memory_items WHERE {columns} AND status = ? "
            "ORDER BY updated_at DESC, id DESC LIMIT ?"
        )
        params: list[object] = ["workspace-0", "agent", "default", "agent", "", "active", 100]
        expected_index = "idx_memory_items_scope_status_updated"
    else:
        sql = (
            f"SELECT * FROM memory_events WHERE {columns} ORDER BY created_at DESC, id DESC LIMIT ?"
        )
        params = ["workspace-0", "agent", "default", "agent", "", 100]
        expected_index = "idx_memory_events_scope_created"
    with sqlite3.connect(path) as connection:
        details = [str(row[3]) for row in connection.execute(f"EXPLAIN QUERY PLAN {sql}", params)]
    return {
        "target_index_search": int(any(expected_index in detail for detail in details)),
        "temporary_order_sort": int(
            any("USE TEMP B-TREE FOR ORDER BY" in detail for detail in details)
        ),
        "full_table_scan": int(any(f"SCAN memory_{table}" in detail for detail in details)),
    }


def _mirror_items(size: int) -> list[MemoryItem]:
    return [
        MemoryItem(
            id=f"mem_{index:05d}",
            scope=_TARGET_SCOPE,
            text=f"memory-{index}",
            created_at=(_NOW + timedelta(seconds=index)).isoformat(),
            updated_at=(_NOW + timedelta(seconds=index)).isoformat(),
        )
        for index in range(size)
    ]


@pytest.mark.asyncio
@pytest.mark.performance_timing
async def test_p4_configured_memory_tool_event_loop_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    service = build_memory_service_from_config(
        MemoryConfig(
            backend=MemoryBackend.SQLITE,
            path=".iris/memory/memory.db",
            root=".iris/memory",
            search={"use_fts": False},
        ),
        tmp_path,
    )
    assert service is not None
    scopes = [
        MemoryScope(workspace_id="workspace", agent_id="agent-a"),
        MemoryScope(workspace_id="workspace", agent_id="agent-b"),
    ]
    search_threads: list[int] = []
    to_thread_calls = 0
    heartbeat_ticks = 0
    finished = False
    loop_thread = threading.get_ident()
    original_to_thread = asyncio.to_thread

    def search(query: object) -> list[object]:
        del query
        search_threads.append(threading.get_ident())
        time.sleep(0.12)
        return []

    async def to_thread(
        function: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal to_thread_calls
        to_thread_calls += 1
        return await original_to_thread(function, *args, **kwargs)

    async def heartbeat() -> None:
        nonlocal heartbeat_ticks
        while not finished:
            await asyncio.sleep(0.01)
            if not finished:
                heartbeat_ticks += 1

    monkeypatch.setattr(service.store, "search", search)
    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    tool = MemorySearchTool(
        service=service,
        access_policy_factory=lambda context: MemoryAccessPolicy(
            actor_agent_id=context.agent_id,
            write_scope=scopes[0],
            read_scopes=scopes,
        ),
    )
    heartbeat_task = asyncio.create_task(heartbeat())
    started_at = time.perf_counter()
    try:
        await tool.arun(
            MemorySearchToolInput(query="memory", limit=8),
            ToolExecutionContext(workspace_root=tmp_path, agent_id="agent-a"),
        )
    finally:
        finished = True
        await heartbeat_task
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    record_observation(
        scenario="p4_configured_sqlite_memory_tool_event_loop",
        perf_ids=("PERF-09",),
        fixture={"scopes": 2, "blocking_ms_per_scope": 120},
        samples_ms=(elapsed_ms,),
        counters={
            "heartbeat_ticks": heartbeat_ticks,
            "to_thread_calls": to_thread_calls,
            "search_calls": len(search_threads),
            "distinct_search_threads": len(set(search_threads)),
            "searches_on_loop_thread": sum(
                thread_id == loop_thread for thread_id in search_threads
            ),
        },
    )


@pytest.mark.performance_timing
def test_p4_memory_scope_query_observation(
    tmp_path: Path,
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory-query.db", use_fts=False)
    _seed_query_rows(store.path)

    item_samples: list[float] = []
    event_samples: list[float] = []
    for _ in range(5):
        started_at = time.perf_counter()
        items = store.list_items(_TARGET_SCOPE, limit=100)
        item_samples.append((time.perf_counter() - started_at) * 1000)

        started_at = time.perf_counter()
        events = store.list_events(_TARGET_SCOPE, limit=100)
        event_samples.append((time.perf_counter() - started_at) * 1000)

    assert len(items) == 100
    assert len(events) == 100
    record_observation(
        scenario="p4_memory_items_scope_order_query",
        perf_ids=("PERF-05",),
        fixture={"rows": _ROW_COUNT, "target_rows": 300, "limit": 100},
        samples_ms=tuple(item_samples),
        counters=_query_plan_counters(store.path, table="items"),
    )
    record_observation(
        scenario="p4_memory_events_scope_order_query",
        perf_ids=("PERF-05",),
        fixture={"rows": _ROW_COUNT, "target_rows": 300, "limit": 100},
        samples_ms=tuple(event_samples),
        counters=_query_plan_counters(store.path, table="events"),
    )


@pytest.mark.performance_timing
@pytest.mark.parametrize("item_count", [50, 200, 500])
def test_p4_memory_mirror_rebuild_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_performance_timing: None,
    record_observation: Callable[..., None],
    item_count: int,
) -> None:
    root = tmp_path / f"mirror-{item_count}"
    mirror = FileMemoryMirror(root)
    mirror.initialize_layout()
    target_paths = {path.resolve(strict=False) for path in root.rglob("*") if path.is_file()}
    counters = {"target_reads": 0, "target_direct_writes": 0, "atomic_replaces": 0}
    original_read_text = Path.read_text
    original_write_text = Path.write_text
    original_replace = os.replace

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve(strict=False) in target_paths:
            counters["target_reads"] += 1
        return original_read_text(path, *args, **kwargs)

    def write_text(path: Path, data: str, *args: object, **kwargs: object) -> int:
        if path.resolve(strict=False) in target_paths:
            counters["target_direct_writes"] += 1
        return original_write_text(path, data, *args, **kwargs)

    def replace(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None:
        if Path(destination).resolve(strict=False) in target_paths:
            counters["atomic_replaces"] += 1
        original_replace(source, destination)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(os, "replace", replace)
    started_at = time.perf_counter()
    mirror.rebuild_from_store(_MirrorStore(_mirror_items(item_count)), _TARGET_SCOPE)  # type: ignore[arg-type]
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    record_observation(
        scenario="p4_memory_mirror_single_target_rebuild",
        perf_ids=("PERF-06",),
        fixture={"items": item_count, "changed_targets": 1},
        samples_ms=(elapsed_ms,),
        counters=counters,
    )
