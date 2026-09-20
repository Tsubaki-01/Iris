from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryAccessPolicy,
    MemoryBackend,
    MemoryConfig,
    MemoryGetTool,
    MemoryGetToolInput,
    MemoryIOExecutionMode,
    MemoryItem,
    MemoryItemPatch,
    MemoryListTool,
    MemoryListToolInput,
    MemoryOverviewContent,
    MemoryQuery,
    MemorySearchResult,
    MemorySearchTool,
    MemorySearchToolInput,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
    build_memory_service_from_config,
)
from iris.tools import ToolExecutionContext


@pytest.mark.asyncio
async def test_thread_read_uses_one_worker_and_keeps_connection_lifecycle_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "thread.db")
    service = MemoryService(store, io_execution_mode=MemoryIOExecutionMode.THREAD)
    namespace = "project"
    service.remember(MemoryWriteInput(namespace=namespace, text="thread item", reason="test"))
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

    items = await service.alist_items([namespace])

    assert [item.text for item in items] == ["thread item"]
    assert to_thread_calls == 1
    assert len(create_threads) == 1
    assert create_threads == close_threads
    assert set(execute_threads) == set(create_threads)
    assert create_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_overviews_whole_read_runs_all_namespaces_in_one_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """所有 namespace 的实际概览读取共用一个 worker job。"""
    service = build_memory_service_from_config(MemoryConfig(backend="sqlite"), tmp_path)
    assert service is not None and service.mirror is not None
    for namespace in ("first", "second"):
        service.remember(MemoryWriteInput(namespace=namespace, text="fact", reason="seed"))
        service.mirror.publish_overview(
            service.store,
            service.store.read_namespace_snapshot(namespace),
            MemoryOverviewContent(core_facts="fact", knowledge_scope="项目资料"),
        )
    loop_thread = threading.get_ident()
    read_threads: list[int] = []
    jobs = 0
    original_read = service.mirror.read_overview
    original_to_thread = asyncio.to_thread

    def read(namespace: str) -> tuple[int, str, str] | None:
        read_threads.append(threading.get_ident())
        return original_read(namespace)

    async def to_thread(
        function: Callable[..., object], /, *args: object, **kwargs: object
    ) -> object:
        nonlocal jobs
        jobs += 1
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(service.mirror, "read_overview", read)
    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    documents = await service.aload_overviews(["second", "first"])
    assert [document.namespace for document in documents] == ["second", "first"]
    assert jobs == 1 and len(read_threads) == 2
    assert len(set(read_threads)) == 1 and loop_thread not in read_threads
    assert all(not document.path.is_absolute() for document in documents)
    assert all(document.source_revision == 1 for document in documents)


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
    store = SQLiteMemoryStore(tmp_path / f"error-{mode.value}.db")
    service = MemoryService(store, io_execution_mode=mode)
    error = IrisMemoryError("injected memory read failure")

    def fail(query: MemoryQuery) -> list[object]:
        del query
        raise error

    monkeypatch.setattr(store, "search", fail)

    with pytest.raises(IrisMemoryError) as captured:
        await service.arecall(MemoryQuery(namespaces=["project"], text="error"))

    assert captured.value is error


@pytest.mark.asyncio
async def test_cancelled_thread_read_does_not_publish_late_result(tmp_path: Path) -> None:
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "cancel.db"),
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

    task = asyncio.create_task(service.run_async_io(operation))
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
        ),
        tmp_path,
    )
    assert configured is not None

    assert configured.io_execution_mode is MemoryIOExecutionMode.THREAD
    assert (
        MemoryService(SQLiteMemoryStore(tmp_path / "direct.db")).io_execution_mode
        is MemoryIOExecutionMode.INLINE
    )


@pytest.mark.asyncio
async def test_memory_tools_keep_policy_on_loop_and_submit_one_job_per_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "tools.db")
    service = MemoryService(store, io_execution_mode=MemoryIOExecutionMode.THREAD)
    first_namespace = "private-a"
    second_namespace = "private-b"
    first = service.remember(
        MemoryWriteInput(namespace=first_namespace, text="shared first", reason="test")
    )
    second = service.remember(
        MemoryWriteInput(namespace=second_namespace, text="shared second", reason="test")
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
            read_namespaces=[first_namespace, second_namespace],
        )

    def search(query: MemoryQuery) -> list[MemorySearchResult]:
        store_threads.append(threading.get_ident())
        return original_search(query)

    def list_items(namespaces: Sequence[str], **kwargs: object) -> list[MemoryItem]:
        store_threads.append(threading.get_ident())
        return original_list(namespaces, **kwargs)

    def get_item(item_id: str, namespaces: Sequence[str]) -> MemoryItem | None:
        store_threads.append(threading.get_ident())
        return original_get(item_id, namespaces)

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
    assert {item["id"] for item in search_payload["results"]} == {first.id, second.id}
    assert {item["id"] for item in list_payload["items"]} == {first.id, second.id}
    assert get_payload == {
        "found": True,
        "item": get_payload["item"],
    }
    assert get_payload["item"]["id"] == first.id
    assert policy_threads == [loop_thread, loop_thread, loop_thread]
    assert to_thread_calls == 3
    assert len(store_threads) == 3
    assert all(thread_id != loop_thread for thread_id in store_threads)


class _CustomReadStore:
    def __init__(self) -> None:
        self.search_threads: list[int] = []

    def search(self, query: MemoryQuery) -> list[object]:
        del query
        self.search_threads.append(threading.get_ident())
        return []


@pytest.mark.asyncio
async def test_direct_custom_store_remains_inline() -> None:
    store = _CustomReadStore()
    service = MemoryService(store)  # type: ignore[arg-type]
    assert await service.arecall(MemoryQuery(text="anything")) == []
    assert store.search_threads == [threading.get_ident()]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["remember", "update", "forget"])
async def test_async_write_and_mirror_refresh_share_one_worker_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    store = SQLiteMemoryStore(tmp_path / "writes.db")
    service = MemoryService(store, mirror=mirror, io_execution_mode=MemoryIOExecutionMode.THREAD)
    seeded = service.remember(MemoryWriteInput(text="seed", reason="seed"))
    loop_thread = threading.get_ident()
    worker_jobs = 0
    connection_threads: list[int] = []
    mirror_threads: list[int] = []
    original_connection = store._connection
    original_replace = mirror._atomic_replace
    original_to_thread = asyncio.to_thread

    @contextmanager
    def connection() -> Iterator[object]:
        connection_threads.append(threading.get_ident())
        with original_connection() as opened:
            yield opened

    def replace(relative_path: str, content: str) -> None:
        mirror_threads.append(threading.get_ident())
        original_replace(relative_path, content)

    async def to_thread(
        function: Callable[..., object], /, *args: object, **kwargs: object
    ) -> object:
        nonlocal worker_jobs
        worker_jobs += 1
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(store, "_connection", connection)
    monkeypatch.setattr(mirror, "_atomic_replace", replace)
    monkeypatch.setattr(asyncio, "to_thread", to_thread)

    if operation == "remember":
        result = await service.aremember(MemoryWriteInput(text="new fact", reason="new"))
        assert result.text == "new fact"
    elif operation == "update":
        result = await service.aupdate(
            seeded.id, "project", MemoryItemPatch(text="updated fact"), reason="edit"
        )
        assert result.id == seeded.id and result.text == "updated fact"
    else:
        assert await service.aforget(seeded.id, "project", reason="done") is True

    assert worker_jobs == 1
    assert connection_threads and mirror_threads
    assert set(connection_threads) == set(mirror_threads)
    assert loop_thread not in connection_threads
