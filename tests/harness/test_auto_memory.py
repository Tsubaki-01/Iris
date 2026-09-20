"""普通run和恢复复用durable概览，只有成功压缩重新采用发布物。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.agents import CompactionConfig
from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response
from .test_context_compaction import CompactionProvider
from .test_runner_memory import OverviewService, overview, with_memory


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_kind", ["hitl", "recover"])
async def test_resume_uses_saved_window_without_loading_or_reselecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_kind: str
) -> None:
    """新runner的更小memory比例不改变已经初始化的旧窗口。"""
    service = OverviewService(tmp_path / "memory.db")
    registry = ToolRegistry()
    registry.register_function(
        lambda: "写入", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    store = InMemoryLifecycleStore()
    first = (
        StaticProvider(tool_response(ToolUseBlock(id="write", name="write", input={})))
        if resume_kind == "hitl"
        else BlockingProvider()
    )
    runner = AgentRunner(
        runtime=with_memory(build_runtime(tmp_path, provider=first, registry=registry), service),
        store=store,
    )
    task = asyncio.create_task(runner.start(AgentRunRequest(input="部署", run_id="resume-window")))
    if resume_kind == "recover":
        await asyncio.wait_for(first.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        waiting = None
    else:
        waiting = await task
        assert waiting.pending_interaction is not None, waiting.error
    saved = store.load_session("default").context_window
    assert saved is not None

    async def forbidden(namespaces: object) -> None:
        raise AssertionError("恢复不能重新加载概览")

    monkeypatch.setattr(service, "aload_overviews", forbidden)
    provider = StaticProvider(text_response())
    runtime = with_memory(build_runtime(tmp_path, provider=provider, registry=registry), service)
    memory = runtime.environment.agent_config.memory
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={
            "memory": memory.model_copy(
                update={
                    "overview": memory.overview.model_copy(update={"system_budget_ratio": 0.000001})
                }
            )
        }
    )
    resumed = AgentRunner(runtime=runtime, store=store)
    if waiting is not None:
        result = await resumed.resume(
            "resume-window",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    else:
        run = store.load_run("resume-window")
        result = await resumed.recover(
            "resume-window", expected_activation_id=run.current_activation_id
        )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert store.load_session("default").context_window == saved
    assert provider.requests[0].messages[0].text.endswith(saved.memory_overview)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "navigation", "too_small"])
async def test_compaction_atomically_adopts_and_sends_the_new_window(
    tmp_path: Path, mode: str
) -> None:
    """成功时即时请求与提交窗口相同，容量失败时旧窗口保持。"""
    service = OverviewService(tmp_path / "memory.db")
    store = InMemoryLifecycleStore()
    initial = AgentRunner(
        runtime=with_memory(
            build_runtime(tmp_path, provider=StaticProvider(text_response())), service
        ),
        store=store,
    )
    first = await initial.start(AgentRunRequest(input="旧中性背景" * 4000, run_id="history"))
    assert first.run.stop_reason is RunStopReason.COMPLETED, first.error
    old_window = store.load_session("default").context_window
    service.documents = (overview("新版 B" if mode == "full" else "新版事实" * 500, revision=2),)
    provider = CompactionProvider(text_response())
    runtime = with_memory(build_runtime(tmp_path, provider=provider), service)
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={
            "compaction": CompactionConfig(
                input_budget_tokens=1000 if mode == "too_small" else 20000
            )
        }
    )
    result = await AgentRunner(runtime=runtime, store=store).start(
        AgentRunRequest(input="现在确认", run_id="compact-window")
    )
    snapshot = store.load_session("default")
    assert len(provider.summary_requests) == 1
    assert service.reads == [("project",), ("project",)]
    if mode == "too_small":
        assert result.run.stop_reason is RunStopReason.FAILED
        assert result.error.source == "context"
        assert snapshot.context_window == old_window
        assert snapshot.compaction is None
        assert not provider.requests
    else:
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
        assert snapshot.context_window.mode == mode
        assert snapshot.context_window.sources[0].source_revision == 2
        assert snapshot.compaction is not None
        assert (
            provider.requests[0].messages[0].text.endswith(snapshot.context_window.memory_overview)
        )
        assert "事实 A" not in snapshot.context_window.memory_overview
