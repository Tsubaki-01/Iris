"""每轮自动记忆召回、可见原文去重和恢复的真实 Runner 回归。"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunStopReason, RuntimeExecutionOptions
from iris.memory import (
    MemoryConfig,
    MemoryItemPatch,
    MemoryQuery,
    MemorySearchResult,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.message import Msg, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response
from .test_context_compaction import CompactionProvider
from .test_context_compaction import _runtime as compaction_runtime


def _memories(messages: list[Msg]) -> list[Msg]:
    return [message for message in messages if message.metadata.get("context_kind") == "memory"]


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))


@pytest.mark.asyncio
async def test_yaml_recall_and_model_search_share_configured_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """从 YAML 到模型请求、补查工具和历史提交贯通同一 SQLite 服务。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: memory-agent\nmodel: openai/test\nsystem: instructions\n"
        "memory:\n  backend: sqlite\n  max_query_terms: 1\n",
        encoding="utf-8",
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="lookup", name="memory_search", input={"query": "部署"})),
        text_response(),
    )
    runner = AgentRunner.from_config_path(path, provider=provider)
    service = runner.runtime.environment.memory_service
    assert service is not None
    item = service.remember(MemoryWriteInput(text="部署使用 uv", reason="项目约定"))
    calls: list[MemoryQuery] = []
    original = service.arecall

    async def recall(query: MemoryQuery) -> list[MemorySearchResult]:
        calls.append(query)
        return await original(query)

    monkeypatch.setattr(service, "arecall", recall)
    result = await runner.start(AgentRunRequest(input="部署", run_id="yaml-recall"))
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert [query.max_query_terms for query in calls] == [1, None]
    assert len(provider.requests) == 2
    assert all(len(_memories(request.messages)) == 1 for request in provider.requests)
    tool_results = [
        block for message in provider.requests[1].messages for block in message.tool_results
    ]
    assert json.loads(tool_results[0].content)["results"][0]["id"] == item.id
    assert len(_memories(runner.store.load_session("default").messages)) == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_new_runs_recall_once_and_only_append_new_visible_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """相同原文复用、更新追加、软删除不改旧历史，工具循环不重复召回。"""
    service = _service(tmp_path)
    item = service.remember(MemoryWriteInput(text="项目部署使用 uv", reason="测试"))
    registry = ToolRegistry()
    registry.register_function(lambda: "工具完成", name="read", description="读取")
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="read", name="read", input={})),
        *[text_response() for _ in range(4)],
    )
    runtime = build_runtime(tmp_path, provider=provider, registry=registry)
    runtime.environment.memory_service = service
    calls: list[MemoryQuery] = []
    original = service.arecall

    async def recall(query: MemoryQuery) -> list[MemorySearchResult]:
        calls.append(query)
        return await original(query)

    monkeypatch.setattr(service, "arecall", recall)
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=runtime, store=store)
    for run_id in ("first", "second"):
        result = await runner.start(AgentRunRequest(input="项目部署", run_id=run_id))
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert len(calls) == 2
    assert all(query.text == "项目部署" and query.max_query_terms is None for query in calls)
    assert len(_memories(store.load_session("default").messages)) == 1
    assert all(len(_memories(request.messages)) == 1 for request in provider.requests)

    service.update(item.id, "project", MemoryItemPatch(text="项目部署使用 uv sync"), reason="更新")
    await runner.start(AgentRunRequest(input="项目部署", run_id="updated"))
    assert len(_memories(store.load_session("default").messages)) == 2
    assert "uv sync" in _memories(provider.requests[-1].messages)[-1].text
    service.forget(item.id, "project", reason="忘记")
    await runner.start(AgentRunRequest(input="项目部署", run_id="forgotten"))
    assert len(calls) == 4
    assert len(_memories(store.load_session("default").messages)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["manual", "disabled", "empty_results", "explicit_query"])
async def test_explicit_sources_and_manual_mode_override_automatic_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """显式空结果也能关闭本轮召回；显式查询不继承自动预算。"""
    service = _service(tmp_path)
    service.remember(MemoryWriteInput(text="部署日志", reason="测试"))
    provider = StaticProvider(text_response())
    runtime = build_runtime(tmp_path, provider=provider)
    runtime.environment.memory_service = None if mode == "disabled" else service
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={
            "memory": MemoryConfig(
                recall_mode="manual" if mode == "manual" else "on_turn", max_query_terms=1
            )
        }
    )

    async def forbidden_auto(query: MemoryQuery) -> list[MemorySearchResult]:
        pytest.fail("本轮不应自动查询")

    monkeypatch.setattr(service, "arecall", forbidden_auto)
    options = RuntimeExecutionOptions()
    if mode == "empty_results":
        options = RuntimeExecutionOptions(memory_results=[])
    elif mode == "explicit_query":
        options = RuntimeExecutionOptions(
            memory_query=MemoryQuery(text="部署").model_dump(mode="json")
        )
    store = InMemoryLifecycleStore()
    result = await AgentRunner(runtime=runtime, store=store).start(
        AgentRunRequest(input="本轮正文无关键词", run_id=mode),
        options=AgentRunOptions(runtime=options),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert len(_memories(store.load_session("default").messages)) == (mode == "explicit_query")


@pytest.mark.asyncio
async def test_auto_query_uses_configured_namespace_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """自动查询只使用当前输入，配置预算和命名空间传给唯一查询 owner。"""
    service = _service(tmp_path)
    service.remember(MemoryWriteInput(namespace="notes", text="部署日志", reason="测试"))
    service.remember(MemoryWriteInput(text="部署日志不能跨范围读取", reason="测试"))
    runtime = build_runtime(tmp_path, provider=StaticProvider(text_response()))
    runtime.environment.memory_service = service
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={"memory": MemoryConfig(read_namespaces=["notes"], max_query_terms=3)}
    )
    calls: list[MemoryQuery] = []
    original = service.arecall

    async def recall(query: MemoryQuery) -> list[MemorySearchResult]:
        calls.append(query)
        return await original(query)

    monkeypatch.setattr(service, "arecall", recall)
    store = InMemoryLifecycleStore()
    result = await AgentRunner(runtime=runtime, store=store).start(
        AgentRunRequest(input="查询部署日志", run_id="configured")
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert len(calls) == 1
    assert calls[0].namespaces == ["notes"]
    assert calls[0].max_query_terms == 3
    assert calls[0].text == "查询部署日志"
    messages = _memories(store.load_session("default").messages)
    assert len(messages) == 1 and messages[0].metadata["namespace"] == "notes"


@pytest.mark.asyncio
async def test_auto_read_failure_warns_and_commits_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """读取故障有带 run_id 的 WARNING，但不阻断本轮主模型调用。"""
    service = _service(tmp_path)

    async def fail(query: MemoryQuery) -> list[MemorySearchResult]:
        raise IrisMemoryError("数据库暂时不可读取")

    monkeypatch.setattr(service, "arecall", fail)
    runtime = build_runtime(tmp_path, provider=StaticProvider(text_response()))
    runtime.environment.memory_service = service
    store = InMemoryLifecycleStore()
    with caplog.at_level(logging.WARNING):
        result = await AgentRunner(runtime=runtime, store=store).start(
            AgentRunRequest(input="继续任务", run_id="auto-read-failure")
        )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert [message.text for message in store.load_session("default").messages] == [
        "继续任务",
        "完成",
    ]
    assert any(
        record.levelno == logging.WARNING
        and "auto-read-failure" in record.message
        and "memory" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_compacted_memory_is_recalled_again_without_original_pinning(tmp_path: Path) -> None:
    """durable 原文存在不等于模型可见；压缩之后允许重新注入相同条目。"""
    service = _service(tmp_path)
    service.remember(MemoryWriteInput(text="部署用 uv", reason="测试"))
    provider = CompactionProvider(text_response("旧" * 1100), text_response(), text_response())
    runtime = compaction_runtime(tmp_path, provider)
    runtime.environment.memory_service = service
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=runtime, store=store)
    first = await runner.start(AgentRunRequest(input="部署", run_id="original"))
    assert first.run.stop_reason is RunStopReason.COMPLETED, first.error
    second = await runner.start(
        AgentRunRequest(input="另一任务", run_id="compact"),
        options=AgentRunOptions(runtime=RuntimeExecutionOptions(memory_results=[])),
    )
    assert second.run.stop_reason is RunStopReason.COMPLETED, second.error
    assert provider.summary_requests
    assert not _memories(provider.requests[1].messages)
    assert len(_memories(store.load_session("default").messages)) == 1
    third = await runner.start(AgentRunRequest(input="部署", run_id="recall-again"))
    assert third.run.stop_reason is RunStopReason.COMPLETED, third.error
    assert len(_memories(store.load_session("default").messages)) == 2
    assert len(_memories(provider.requests[2].messages)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_kind", ["hitl", "recover"])
async def test_auto_memory_replays_after_resume_without_querying_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_kind: str
) -> None:
    """恢复使用输入提交的历史；不重新执行自动查询。"""
    service = _service(tmp_path)
    service.remember(MemoryWriteInput(text="部署资料", reason="测试"))
    registry = ToolRegistry()
    registry.register_function(
        lambda: "写入", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    store = InMemoryLifecycleStore()
    first_provider = (
        StaticProvider(tool_response(ToolUseBlock(id="write", name="write", input={})))
        if resume_kind == "hitl"
        else BlockingProvider()
    )
    runtime = build_runtime(tmp_path, provider=first_provider, registry=registry)
    runtime.environment.memory_service = service
    runner = AgentRunner(runtime=runtime, store=store)
    task = asyncio.create_task(runner.start(AgentRunRequest(input="部署", run_id="resume-auto")))
    if isinstance(first_provider, BlockingProvider):
        await asyncio.wait_for(first_provider.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        waiting = None
    else:
        waiting = await task
        assert waiting.pending_interaction is not None, waiting.error
    saved = _memories(store.load_session("default").messages)
    assert len(saved) == 1

    async def forbidden(query: MemoryQuery) -> list[MemorySearchResult]:
        pytest.fail("恢复不应再次自动查询")

    monkeypatch.setattr(service, "arecall", forbidden)
    provider = StaticProvider(text_response())
    resumed_runtime = build_runtime(tmp_path, provider=provider, registry=registry)
    resumed_runtime.environment.memory_service = service
    resumed = AgentRunner(runtime=resumed_runtime, store=store)
    if waiting is not None:
        assert waiting.pending_interaction is not None
        result = await resumed.resume(
            "resume-auto",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    else:
        run = store.load_run("resume-auto")
        assert run is not None and run.current_activation_id is not None
        result = await resumed.recover(
            "resume-auto", expected_activation_id=run.current_activation_id
        )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert _memories(provider.requests[0].messages) == saved
    assert _memories(store.load_session("default").messages) == saved
