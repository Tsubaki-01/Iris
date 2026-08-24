from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MemoryAccessPolicy,
    MemoryBackend,
    MemoryConfig,
    MemoryGetTool,
    MemoryGetToolInput,
    MemoryIOExecutionMode,
    MemoryListTool,
    MemoryListToolInput,
    MemoryQuery,
    MemoryScope,
    MemorySearchTool,
    MemorySearchToolInput,
    MemoryService,
    MemoryStore,
    MemoryWriteInput,
    SQLiteMemoryStore,
    build_memory_service_from_config,
)
from iris.tools import ToolExecutionContext


def _scope(agent_id: str = "agent") -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id=agent_id)


@pytest.mark.asyncio
async def test_async_read_wrappers_preserve_sync_results_and_default_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "inline.db", use_fts=False)
    service = MemoryService(store)
    scope = _scope()
    item = service.remember(MemoryWriteInput(scope=scope, text="用户偏好简洁回答", reason="test"))
    query = MemoryQuery(scope=scope, text="简洁", limit=5)
    loop_thread = threading.get_ident()
    search_threads: list[int] = []
    original_search = store.search

    def search(value: MemoryQuery):
        search_threads.append(threading.get_ident())
        return original_search(value)

    monkeypatch.setattr(store, "search", search)

    assert service.io_execution_mode is MemoryIOExecutionMode.INLINE
    assert await service.arecall(query) == service.recall(query)
    assert await service.aget_item(item.id, scope) == service.get_item(item.id, scope)
    assert await service.alist_items(scope) == service.list_items(scope)
    assert await service.alist_events(scope) == service.list_events(scope)
    assert await service.abuild_context(query, max_chars=100) == service.build_context(
        query,
        max_chars=100,
    )
    assert search_threads == [loop_thread, loop_thread, loop_thread, loop_thread]
    with pytest.raises(AttributeError):
        service.io_execution_mode = MemoryIOExecutionMode.THREAD  # type: ignore[misc]


@pytest.mark.asyncio
async def test_thread_read_uses_one_worker_and_keeps_connection_lifecycle_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "thread.db", use_fts=False)
    service = MemoryService(store, io_execution_mode=MemoryIOExecutionMode.THREAD)
    scope = _scope()
    service.remember(MemoryWriteInput(scope=scope, text="thread item", reason="test"))
    loop_thread = threading.get_ident()
    to_thread_calls = 0
    create_threads: list[int] = []
    execute_threads: list[int] = []
    close_threads: list[int] = []
    original_to_thread = asyncio.to_thread
    original_connection = store._connection

    async def to_thread(
        function: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal to_thread_calls
        to_thread_calls += 1
        return await original_to_thread(function, *args, **kwargs)

    @contextmanager
    def connection() -> Iterator[object]:
        create_threads.append(threading.get_ident())
        with original_connection() as opened:
            opened.set_trace_callback(lambda _: execute_threads.append(threading.get_ident()))
            yield opened
        close_threads.append(threading.get_ident())

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    monkeypatch.setattr(store, "_connection", connection)

    items = await service.alist_items(scope)

    assert [item.text for item in items] == ["thread item"]
    assert to_thread_calls == 1
    assert len(create_threads) == 1
    assert create_threads == close_threads
    assert set(execute_threads) == set(create_threads)
    assert create_threads[0] != loop_thread


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [MemoryIOExecutionMode.INLINE, MemoryIOExecutionMode.THREAD],
)
async def test_async_read_preserves_memory_exception_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: MemoryIOExecutionMode,
) -> None:
    store = SQLiteMemoryStore(tmp_path / f"error-{mode.value}.db", use_fts=False)
    service = MemoryService(store, io_execution_mode=mode)
    error = IrisMemoryError("injected memory read failure")

    def fail(query: MemoryQuery) -> list[object]:
        del query
        raise error

    monkeypatch.setattr(store, "search", fail)

    with pytest.raises(IrisMemoryError) as captured:
        await service.arecall(MemoryQuery(scope=_scope(), text="error"))

    assert captured.value is error


@pytest.mark.asyncio
async def test_cancelled_thread_read_does_not_publish_late_result(tmp_path: Path) -> None:
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "cancel.db", use_fts=False),
        io_execution_mode=MemoryIOExecutionMode.THREAD,
    )
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def operation() -> str:
        started.set()
        release.wait(timeout=2)
        finished.set()
        return "late-result"

    task = asyncio.create_task(service.run_async_read(operation))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(finished.wait, 1)
    assert task.cancelled()


def test_configured_sqlite_uses_thread_but_direct_service_stays_inline(tmp_path: Path) -> None:
    configured = build_memory_service_from_config(
        MemoryConfig(
            backend=MemoryBackend.SQLITE,
            path=".iris/memory/memory.db",
            root=".iris/memory",
            search={"use_fts": False},
        ),
        tmp_path,
    )
    assert configured is not None

    assert configured.io_execution_mode is MemoryIOExecutionMode.THREAD
    assert (
        MemoryService(SQLiteMemoryStore(tmp_path / "direct.db", use_fts=False)).io_execution_mode
        is MemoryIOExecutionMode.INLINE
    )


@pytest.mark.asyncio
async def test_custom_store_keeps_default_thread_affinity() -> None:
    loop_thread = threading.get_ident()
    store = _CustomReadStore()
    service = MemoryService(cast(MemoryStore, store))

    assert await service.arecall(MemoryQuery(scope=_scope(), text="custom")) == []
    assert store.search_threads == [loop_thread]


@pytest.mark.asyncio
async def test_memory_tools_keep_policy_on_loop_and_submit_one_job_per_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "tools.db", use_fts=False)
    service = MemoryService(store, io_execution_mode=MemoryIOExecutionMode.THREAD)
    first_scope = _scope("agent-a")
    second_scope = _scope("agent-b")
    first = service.remember(
        MemoryWriteInput(scope=first_scope, text="shared first", reason="test")
    )
    second = service.remember(
        MemoryWriteInput(scope=second_scope, text="shared second", reason="test")
    )
    loop_thread = threading.get_ident()
    policy_threads: list[int] = []
    store_threads: list[int] = []
    to_thread_calls = 0
    original_to_thread = asyncio.to_thread
    original_search = store.search
    original_list = store.list_items
    original_get = store.get_item

    def policy(context: ToolExecutionContext) -> MemoryAccessPolicy:
        policy_threads.append(threading.get_ident())
        return MemoryAccessPolicy(
            actor_agent_id=context.agent_id,
            write_scope=first_scope,
            read_scopes=[first_scope, second_scope],
        )

    def search(query: MemoryQuery):
        store_threads.append(threading.get_ident())
        return original_search(query)

    def list_items(scope: MemoryScope, **kwargs: object):
        store_threads.append(threading.get_ident())
        return original_list(scope, **kwargs)

    def get_item(item_id: str, scope: MemoryScope):
        store_threads.append(threading.get_ident())
        return original_get(item_id, scope)

    async def to_thread(
        function: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal to_thread_calls
        to_thread_calls += 1
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(store, "search", search)
    monkeypatch.setattr(store, "list_items", list_items)
    monkeypatch.setattr(store, "get_item", get_item)
    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    context = ToolExecutionContext(workspace_root=tmp_path, agent_id="agent-a")

    search_result = await MemorySearchTool(
        service=service,
        access_policy_factory=policy,
    ).arun(MemorySearchToolInput(query="shared", limit=8), context)
    list_result = await MemoryListTool(
        service=service,
        access_policy_factory=policy,
    ).arun(MemoryListToolInput(limit=8), context)
    get_result = await MemoryGetTool(
        service=service,
        access_policy_factory=policy,
    ).arun(MemoryGetToolInput(item_id=first.id), context)

    search_payload = json.loads(search_result.content[0].text)
    list_payload = json.loads(list_result.content[0].text)
    get_payload = json.loads(get_result.content[0].text)
    assert [item["id"] for item in search_payload["results"]] == [first.id, second.id]
    assert [item["id"] for item in list_payload["items"]] == [first.id, second.id]
    assert get_payload == {
        "found": True,
        "item": get_payload["item"],
    }
    assert get_payload["item"]["id"] == first.id
    assert policy_threads == [loop_thread, loop_thread, loop_thread]
    assert to_thread_calls == 3
    assert len(store_threads) == 5
    assert all(thread_id != loop_thread for thread_id in store_threads)


class _CustomReadStore:
    def __init__(self) -> None:
        self.search_threads: list[int] = []

    def search(self, query: MemoryQuery) -> list[object]:
        del query
        self.search_threads.append(threading.get_ident())
        return []
