"""真实文件与 SQLite 下，工具本地 IO 的取消和超时回执收口。"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RunPhase,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolCallPhase,
)
from iris.message import TextBlock, ToolUseBlock
from iris.store import SQLiteStore
from iris.tools import (
    BaseTool,
    DefaultPermissionPolicy,
    ToolArtifact,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    WorkspaceFileService,
    register_file_tools,
)
from iris.tools.artifacts import ToolArtifactStore

from .fakes import StaticProvider, build_runtime, tool_batch_response, tool_response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interrupt", "fails"),
    [
        ("signal", False),
        ("task", False),
        ("timeout", False),
        ("deadline", False),
        ("task", True),
        ("timeout", True),
    ],
)
async def test_file_io_receipt_is_durable_before_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: str, fails: bool
) -> None:
    """真实写入完成后保存结果和 read_state，再结算取消或超时。"""
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    worker_threads: list[int] = []
    service = WorkspaceFileService()
    original_write = service.atomic_write

    def blocked_write(path: Path, content: str) -> None:
        worker_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(5)
            if fails:
                raise OSError("受控写入失败")
            original_write(path, content)
        finally:
            finished.set()

    monkeypatch.setattr(service, "atomic_write", blocked_write)
    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="write-1",
                name="write_file",
                input={"file_path": "output.txt", "content": "完成"},
            )
        )
    )
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=register_file_tools(file_service=service),
            permission_policy=DefaultPermissionPolicy(write_mode="allow"),
            provider=provider,
        ),
        store=store,
    )
    options = AgentRunOptions(
        runtime=RuntimeExecutionOptions(
            tool_timeout_seconds=0.15 if interrupt == "timeout" else None
        ),
        limits=RunLimits(
            deadline_at=datetime.now(UTC) + timedelta(milliseconds=700)
            if interrupt == "deadline"
            else None
        ),
    )
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="写入", run_id="file-io"), options=options)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert len(worker_threads) == 1 and worker_threads[0] != loop_thread
        assert store.load_tool_call("file-io", "write-1").phase is ToolCallPhase.CLAIMED
        if interrupt == "signal":
            runner.request_cancel("file-io")
        elif interrupt == "task":
            running.cancel()
            await asyncio.sleep(0.01)
            running.cancel()
        else:
            await asyncio.sleep(0.8 if interrupt == "deadline" else 0.25)
        await asyncio.sleep(0.01)
        assert not running.done()
        assert not finished.is_set()
        assert store.load_result("file-io") is None

        release.set()
        if interrupt == "task":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=2)
            # 外部 task 取消不冒充公开 durable cancellation；已知结果仍可恢复。
            assert store.load_run("file-io").phase is RunPhase.ACTIVE
            assert store.load_result("file-io") is None
        else:
            result = await asyncio.wait_for(running, timeout=2)
            expected = {
                "signal": RunStopReason.CANCELLED,
                "timeout": RunStopReason.FAILED,
                "deadline": RunStopReason.DEADLINE_EXCEEDED,
            }
            assert result.run.stop_reason is expected[interrupt]
            if interrupt == "timeout":
                assert result.error is not None and result.error.code == "TOOL_TIMEOUT"

        target = (tmp_path / "output.txt").resolve()
        assert finished.is_set()
        [record] = store.list_tool_calls("file-io")
        assert record.phase is ToolCallPhase.COMMITTED
        assert record.result is not None and record.result.is_error is fails
        checkpoint = store.load_checkpoint("file-io")
        assert checkpoint is not None
        if fails:
            assert not target.exists()
            assert "受控写入失败" in record.result.model_content
            assert checkpoint.engine_cursor["read_state"] is None
        else:
            assert target.read_text(encoding="utf-8") == "完成"
            observed = checkpoint.engine_cursor["read_state"]["files"][str(target)]
            assert observed["size_bytes"] == target.stat().st_size
        assert len(provider.requests) == 1
        assert len(worker_threads) == 1
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["signal", "task", "timeout"])
async def test_parallel_artifacts_commit_in_order_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: str
) -> None:
    """两个真实 artifact 并行写盘，反向完成时也按原序保存已知结果。"""
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = {call_id: asyncio.Event() for call_id in ("read-1", "read-2")}
    releases = {call_id: threading.Event() for call_id in entered}
    finished = {call_id: asyncio.Event() for call_id in entered}
    worker_threads: dict[str, int] = {}
    original_persist = ToolArtifactStore._persist_text

    def blocked_persist(
        self: ToolArtifactStore,
        tool_use_id: str,
        content: str,
        *,
        suffix: str,
        mime_type: str,
        preview: str,
    ) -> ToolArtifact:
        worker_threads[tool_use_id] = threading.get_ident()
        loop.call_soon_threadsafe(entered[tool_use_id].set)
        assert releases[tool_use_id].wait(5)
        result = original_persist(
            self, tool_use_id, content, suffix=suffix, mime_type=mime_type, preview=preview
        )
        loop.call_soon_threadsafe(finished[tool_use_id].set)
        return result

    class LargeReadTool(BaseTool):
        """直接返回大文本，让真实 executor 负责 artifact 落盘。"""

        definition = ToolDefinition(
            name="large_read",
            description="返回大文本",
            input_schema={"type": "object", "properties": {}},
            capabilities={ToolCapability.READ},
            max_result_chars=1000,
            preview_chars=100,
        )

        async def arun(
            self, params: BaseModel | dict[str, object], context: ToolExecutionContext
        ) -> ToolResult:
            """正文包含调用标识，验证不同并发结果没有串线。"""
            return ToolResult(
                tool_use_id=context.call_id,
                tool_name=context.tool_name,
                content=[TextBlock(text=context.call_id * 1000)],
            )

    monkeypatch.setattr(ToolArtifactStore, "_persist_text", blocked_persist)
    registry = ToolRegistry()
    registry.register(LargeReadTool())
    provider = StaticProvider(
        tool_batch_response(*(ToolUseBlock(id=key, name="large_read", input={}) for key in entered))
    )
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider), store=store
    )
    running = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="读取", run_id="artifact-io"),
            options=AgentRunOptions(
                runtime=RuntimeExecutionOptions(
                    tool_timeout_seconds=0.15 if interrupt == "timeout" else None
                )
            ),
        )
    )
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 2)
        assert all(thread_id != loop_thread for thread_id in worker_threads.values())
        if interrupt == "signal":
            runner.request_cancel("artifact-io")
        elif interrupt == "task":
            running.cancel()
            await asyncio.sleep(0.01)
            running.cancel()
        else:
            await asyncio.sleep(0.25)
        assert not running.done()
        releases["read-2"].set()
        await asyncio.wait_for(finished["read-2"].wait(), 2)
        assert not running.done()
        releases["read-1"].set()
        if interrupt == "task":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 2)
        else:
            result = await asyncio.wait_for(running, 2)
            assert result.run.stop_reason is (
                RunStopReason.FAILED if interrupt == "timeout" else RunStopReason.CANCELLED
            )
            if interrupt == "timeout":
                assert result.error is not None and result.error.code == "TOOL_TIMEOUT"
        records = store.list_tool_calls("artifact-io")
        assert [record.tool_call_id for record in records] == ["read-1", "read-2"]
        for record in records:
            assert record.phase is ToolCallPhase.COMMITTED
            assert record.result is not None and record.result.artifact is not None
            assert (
                record.result.artifact.path.read_text(encoding="utf-8")
                == record.tool_call_id * 1000
            )
        assert len(provider.requests) == 1
    finally:
        for release in releases.values():
            release.set()
        await asyncio.gather(running, return_exceptions=True)
        await runner.aclose()
