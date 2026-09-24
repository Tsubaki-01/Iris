"""维护取消后，已经派发的数据库工作仍由服务跟踪到完成。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import iris.memory.generation as memory_generation
import iris.memory.service as memory_service
from iris.memory import (
    FileMemoryMirror,
    MemoryIOExecutionMode,
    MemoryItemPatch,
    MemoryObserveInput,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.message import LLMRequest, LLMResponse, TextBlock


class _Provider:
    """按实际阶段输入返回确定性结果，并保留真实调用次数。"""

    def __init__(self) -> None:
        self.calls = 0

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        return 10

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        source = json.loads(request.messages[1].text)
        if "records" in source:
            content = {"observations": []}
        elif "observations" in source:
            item = source["items"][0]
            content = {
                "operations": [
                    {
                        "action": "update",
                        "target_id": item["id"],
                        "text": "updated fact",
                        "evidence": item["evidence"],
                        "reason": "整理既有事实",
                    }
                ],
                "resolutions": [],
            }
        else:
            content = {"core_facts": "fact", "knowledge_scope": "facts"}
        return LLMResponse(
            provider="controlled",
            content=[TextBlock(text=json.dumps(content))],
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )


def _service(tmp_path: Path) -> tuple[MemoryService, _Provider]:
    """使用真实 SQLite 与 THREAD 路径验证提交和落账。"""
    provider = _Provider()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror"),
        generation_provider=provider,
        generation_model="test",
        overview_provider=provider,
        overview_model="test",
        io_execution_mode=MemoryIOExecutionMode.THREAD,
    )
    return service, provider


def _stage_results(service: MemoryService, stage: str) -> list[dict[str, Any]]:
    """读取全部阶段尝试，以免最近一条状态掩盖重复成本。"""
    with sqlite3.connect(service.store.path) as connection:
        return [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload FROM memory_generation_results WHERE stage=?", (stage,)
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["flush", "dream", "overview"])
@pytest.mark.parametrize("boundary", ["template", "prepare", "parse"])
async def test_slow_generation_processing_leaves_loop_and_foreground_reads_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, boundary: str
) -> None:
    """扣住同步生成计算，前台仍能读库并取消等待，迟到结果不消费材料。"""
    service, provider = _service(tmp_path)
    item = service.remember(MemoryWriteInput(text="original fact", reason="seed"))
    service.observe(MemoryObserveInput(text="new source"))
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    started = asyncio.Event()
    release = threading.Event()
    worker_threads: list[int] = []
    model_threads: list[int] = []
    original_complete = provider.complete

    async def complete(request: LLMRequest) -> LLMResponse:
        model_threads.append(threading.get_ident())
        return await original_complete(request)

    monkeypatch.setattr(provider, "complete", complete)
    if boundary == "template":
        target, attribute = service.prompt_renderer, "render_file"
        original = service.prompt_renderer.render_file
    elif boundary == "prepare":
        target, attribute = provider, "estimate_input_tokens"
        original = provider.estimate_input_tokens
    elif stage == "overview":
        target, attribute = memory_service, "complete_overview_content"
        original = memory_service.complete_overview_content
    else:
        target, attribute = memory_generation, "_parse"
        original = memory_generation._parse

    def slow(*args: Any, **kwargs: Any) -> Any:
        worker_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5), "同步生成计算占用了前台事件循环"
        return original(*args, **kwargs)

    monkeypatch.setattr(target, attribute, slow)
    operation = {
        "flush": service.flush,
        "dream": service.dream,
        "overview": service.refresh_overview,
    }[stage]
    task = asyncio.create_task(operation("project"))
    try:
        await asyncio.wait_for(started.wait(), 2)
        current = await asyncio.wait_for(service.aget_item(item.id, ["project"]), 1)
        assert current == item
        assert not task.done()
        assert worker_threads and loop_thread not in worker_threads
        assert all(thread_id == loop_thread for thread_id in model_threads)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await service.wait_pending_io()
    assert provider.calls == (1 if boundary == "parse" else 0)
    state = service.generation_state("project")
    assert state.pending_episodes == state.pending_changes == 1
    assert service.get_item(item.id, ["project"]) == item
    assert _stage_results(service, stage)[0]["status"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["flush", "dream", "overview_publish", "overview_record"])
async def test_cancelled_short_commit_finishes_and_records_model_cost_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """前台重复取消不拆开已开始的短提交，成功记录不能另写一次 cancelled 成本。"""
    service, provider = _service(tmp_path)
    started, release = threading.Event(), threading.Event()
    stage = boundary.split("_")[0]
    operation: Callable[[], Awaitable[object]]
    if stage == "flush":
        service.observe(MemoryObserveInput(text="source input"))
        target, method = service.store, "commit_flush"
        operation = partial(service.flush, "project")
    elif stage == "dream":
        service.remember(MemoryWriteInput(text="original fact", reason="seed"))
        target, method = service.store, "commit_dream"
        operation = partial(service.dream, "project")
    else:
        service.remember(MemoryWriteInput(text="original fact", reason="seed"))
        target, method = (
            (service.mirror, "publish_overview")
            if boundary == "overview_publish"
            else (service.store, "record_generation_result")
        )
        operation = partial(service.refresh_overview, "project")
    original = getattr(target, method)

    def delayed(*args: object, **kwargs: object) -> object:
        started.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, delayed)
    task = asyncio.create_task(operation())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await service.wait_pending_io()
    results = _stage_results(service, stage)
    assert len(results) == provider.calls == 1
    assert results[0]["status"] == "completed"
    assert results[0]["usage"]["total_tokens"] == 15
    state = service.generation_state("project")
    if stage == "flush":
        assert state.pending_episodes == 0
    elif stage == "dream":
        assert state.pending_changes == 0
        assert service.list_items(["project"])[0].text == "updated fact"
    else:
        assert service.load_overviews(["project"])[0].source_revision == state.item_revision


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["state", "mirror"])
async def test_cancellation_after_dream_commit_does_not_relabel_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """权威事务已完成，后续状态读取或投影等待的取消不能重记模型费用。"""
    service, provider = _service(tmp_path)
    service.remember(MemoryWriteInput(text="original fact", reason="seed"))
    started, release = threading.Event(), threading.Event()
    target, method = (
        (service.store, "generation_state")
        if boundary == "state"
        else (service, "_rebuild_committed")
    )
    original = getattr(target, method)

    def delayed(*args: object, **kwargs: object) -> object:
        started.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, delayed)
    task = asyncio.create_task(service.dream("project"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await service.wait_pending_io()
    results = _stage_results(service, "dream")
    assert len(results) == provider.calls == 1
    assert results[0]["status"] == "completed"
    assert results[0]["usage"]["total_tokens"] == 15
    assert service.generation_state("project").pending_changes == 0
    assert service.list_items(["project"])[0].text == "updated fact"


@pytest.mark.asyncio
async def test_cancelled_conflicting_commit_records_one_conflict_without_consuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """短提交取消后仍以真实 CAS 结果落账，不把冲突或成本再写一遍。"""
    service, provider = _service(tmp_path)
    item = service.remember(MemoryWriteInput(text="original fact", reason="seed"))
    started, release = threading.Event(), threading.Event()
    original = service.store.commit_dream

    def delayed(*args: object, **kwargs: object) -> bool:
        started.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.store, "commit_dream", delayed)
    task = asyncio.create_task(service.dream("project"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        service.update(
            item.id, "project", MemoryItemPatch(text="foreground fact"), reason="correct"
        )
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await service.wait_pending_io()
    results = _stage_results(service, "dream")
    assert len(results) == provider.calls == 1
    assert results[0]["status"] == "conflict"
    assert results[0]["usage"]["total_tokens"] == 15
    assert service.generation_state("project").pending_changes == 2
    assert service.get_item(item.id, ["project"]).text == "foreground fact"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["dream", "overview"])
async def test_model_cancellation_before_commit_keeps_original_knowledge(
    tmp_path: Path, stage: str
) -> None:
    """provider 吞取消并返回响应也不得启动提交，已知 usage 仍只记录一次。"""
    service, _ = _service(tmp_path)
    item = service.remember(MemoryWriteInput(text="original fact", reason="seed"))
    started = asyncio.Event()

    class LateProvider(_Provider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return await super().complete(request)

    provider = LateProvider()
    service.generation_provider = service.overview_provider = provider
    task = asyncio.create_task(
        service.dream("project") if stage == "dream" else service.refresh_overview("project")
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await service.wait_pending_io()
    results = _stage_results(service, stage)
    assert len(results) == provider.calls == 1
    assert results[0]["status"] == "cancelled"
    assert results[0]["usage"]["total_tokens"] == 15
    assert service.generation_state("project").pending_changes == 1
    assert service.get_item(item.id, ["project"]) == item
    assert service.load_overviews(["project"])[0].source_revision is None


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_orphan_database_job(tmp_path: Path) -> None:
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        io_execution_mode=MemoryIOExecutionMode.THREAD,
    )
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def operation() -> None:
        started.set()
        release.wait(5)
        finished.set()

    task = asyncio.create_task(service.run_async_io(operation))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    close_wait = asyncio.create_task(service.wait_pending_io())
    await asyncio.sleep(0)
    assert not close_wait.done()
    release.set()
    await asyncio.wait_for(close_wait, 5)
    assert finished.is_set()
