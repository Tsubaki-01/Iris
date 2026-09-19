"""动态记忆随 run 输入持久化、恢复及故障边界的集成测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.context import ContextSection, ContextSlot
from iris.exceptions import IrisProviderError
from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    LifecycleStore,
    RunStopReason,
    RuntimeExecutionOptions,
)
from iris.memory import (
    MemoryContextBundle,
    MemoryItem,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.message import LLMRequest, LLMResponse, Msg, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response


@pytest.fixture(params=["memory", "sqlite"])
def lifecycle_store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """输入归档和恢复同时覆盖两个真实 lifecycle store。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "lifecycle.db")
    return InMemoryLifecycleStore()


def _result(item_id: str, text: str) -> MemorySearchResult:
    return MemorySearchResult(
        item=MemoryItem(
            id=item_id,
            scope=MemoryScope(workspace_id="workspace", agent_id="agent"),
            text=text,
        )
    )


def _with_context(runtime: AgentRuntime) -> AgentRuntime:
    runtime.environment.context_input = runtime.environment.context_input.model_copy(
        update={
            "memory": ContextSection(slots=[ContextSlot(name="static", content="固定记忆")]),
            "before_current_input": ContextSection(
                slots=[ContextSlot(name="environment", content="本轮环境")]
            ),
        }
    )
    return runtime


def _memory_messages(messages: list[Msg] | tuple[Msg, ...]) -> list[Msg]:
    return [message for message in messages if message.metadata.get("context_kind") == "memory"]


def _query_service(tmp_path: Path) -> tuple[MemoryService, MemoryQuery]:
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False))
    service.remember(MemoryWriteInput(scope=scope, text="动态资料", reason="测试输入归档"))
    return service, MemoryQuery(scope=scope, text="动态资料")


@pytest.mark.asyncio
async def test_dynamic_results_are_committed_before_provider_and_replayed_in_tool_loop(
    tmp_path: Path, lifecycle_store: LifecycleStore
) -> None:
    """逐条记忆和 BCI/user 先入库，静态槽位每次装配但不进入历史。"""
    registry = ToolRegistry()
    registry.register_function(lambda: "工具结果", name="read", description="读取资料")

    class InspectingProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            session = lifecycle_store.load_session("default")
            checkpoint = lifecycle_store.load_checkpoint("memory-loop")
            assert checkpoint is not None
            assert checkpoint.engine_cursor["position"] == "before_model"
            assert checkpoint.session_revision == session.revision
            assert [
                message.metadata["item_id"] for message in _memory_messages(session.messages)
            ] == [
                "first",
                "second",
            ]
            assert sum(message.text == "比较资料" for message in session.messages) == 1
            assert sum("本轮环境" in message.text for message in session.messages) == 1
            assert all("固定记忆" not in message.text for message in session.messages)
            return await super().complete(request)

    provider = InspectingProvider(
        tool_response(ToolUseBlock(id="read-1", name="read", input={})), text_response()
    )
    runner = AgentRunner(
        runtime=_with_context(build_runtime(tmp_path, provider=provider, registry=registry)),
        store=lifecycle_store,
    )
    result = await runner.start(
        AgentRunRequest(input="比较资料", run_id="memory-loop"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(
                memory_results=[
                    _result("first", "第一条动态资料").model_dump(mode="json"),
                    _result("second", "第二条动态资料").model_dump(mode="json"),
                ]
            )
        ),
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert result.run.usage.model_steps_reserved == result.run.usage.model_steps_committed == 2
    assert len(provider.requests) == 2
    for request in provider.requests:
        assert len(_memory_messages(request.messages)) == 2
        assert sum("固定记忆" in message.text for message in request.messages) == 1
    session = lifecycle_store.load_session("default")
    assert [message.metadata.get("context_kind") for message in session.messages[:4]] == [
        "memory",
        "memory",
        "before_current_input",
        None,
    ]
    assert all(message.sender == "context" for message in session.messages[:3])
    assert len(_memory_messages(session.messages)) == 2


@pytest.mark.asyncio
async def test_hitl_resume_uses_saved_memory_without_a_second_query(
    tmp_path: Path, lifecycle_store: LifecycleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """换 runner 后的权限恢复重放已保存快照，不再次调用长期记忆服务。"""
    service, query = _query_service(tmp_path)
    calls: list[MemoryQuery] = []
    original = service.abuild_context

    async def read_once(query: MemoryQuery, *, max_chars: int) -> MemoryContextBundle:
        calls.append(query)
        assert len(calls) == 1, "恢复不能重新读取 memory"
        return await original(query, max_chars=max_chars)

    monkeypatch.setattr(service, "abuild_context", read_once)
    registry = ToolRegistry()
    registry.register_function(
        lambda: "已写入", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    first_provider = StaticProvider(tool_response(ToolUseBlock(id="write", name="write", input={})))
    runtime = build_runtime(tmp_path, provider=first_provider, registry=registry)
    runtime.environment.memory_service = service
    waiting = await AgentRunner(runtime=runtime, store=lifecycle_store).start(
        AgentRunRequest(input="使用资料写入", run_id="memory-wait"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(memory_query=query.model_dump(mode="json"))
        ),
    )
    assert waiting.pending_interaction is not None, waiting.error
    saved = _memory_messages(lifecycle_store.load_session("default").messages)
    assert len(saved) == 1

    resumed_provider = StaticProvider(text_response())
    resumed_runtime = build_runtime(tmp_path, provider=resumed_provider, registry=registry)
    resumed_runtime.environment.memory_service = service
    result = await AgentRunner(runtime=resumed_runtime, store=lifecycle_store).resume(
        "memory-wait",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert len(calls) == 1
    assert _memory_messages(first_provider.requests[0].messages) == saved
    assert _memory_messages(resumed_provider.requests[0].messages) == saved
    assert _memory_messages(lifecycle_store.load_session("default").messages) == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt_at", ["input", "model"])
async def test_recovery_queries_only_when_input_was_not_committed(
    tmp_path: Path,
    lifecycle_store: LifecycleStore,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_at: str,
) -> None:
    """输入准备前崩溃可重读；输入提交后崩溃复用快照及模型 reservation。"""
    service, query = _query_service(tmp_path)
    input_started = asyncio.Event()
    release_input = asyncio.Event()
    query_calls = 0
    original = service.abuild_context

    async def controlled_query(query: MemoryQuery, *, max_chars: int) -> MemoryContextBundle:
        nonlocal query_calls
        query_calls += 1
        input_started.set()
        if interrupt_at == "input" and query_calls == 1:
            await release_input.wait()
        if interrupt_at == "model":
            assert query_calls == 1, "before_model 恢复不能重新读取 memory"
        return await original(query, max_chars=max_chars)

    monkeypatch.setattr(service, "abuild_context", controlled_query)
    provider = BlockingProvider()
    runtime = _with_context(build_runtime(tmp_path, provider=provider))
    runtime.environment.memory_service = service
    running = asyncio.create_task(
        AgentRunner(runtime=runtime, store=lifecycle_store).start(
            AgentRunRequest(input="恢复资料", run_id="memory-recover"),
            options=AgentRunOptions(
                runtime=RuntimeExecutionOptions(memory_query=query.model_dump(mode="json"))
            ),
        )
    )
    await asyncio.wait_for(
        (input_started if interrupt_at == "input" else provider.started).wait(), 2
    )
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    checkpoint = lifecycle_store.load_checkpoint("memory-recover")
    crashed = lifecycle_store.load_run("memory-recover")
    assert checkpoint is not None and crashed is not None
    assert crashed.current_activation_id is not None
    assert checkpoint.checkpoint_version == 2
    assert checkpoint.engine_cursor["position"] == (
        "before_input" if interrupt_at == "input" else "before_model"
    )
    before = lifecycle_store.load_session("default")
    assert len(before.messages) == (0 if interrupt_at == "input" else 3)
    assert crashed.usage.model_steps_reserved == (0 if interrupt_at == "input" else 1)

    restored_provider = StaticProvider(text_response())
    restored_runtime = _with_context(build_runtime(tmp_path, provider=restored_provider))
    restored_runtime.environment.memory_service = service
    result = await AgentRunner(runtime=restored_runtime, store=lifecycle_store).recover(
        "memory-recover", expected_activation_id=crashed.current_activation_id
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert query_calls == (2 if interrupt_at == "input" else 1)
    assert result.run.usage.model_steps_reserved == result.run.usage.model_steps_committed == 1
    session = lifecycle_store.load_session("default")
    assert len(_memory_messages(session.messages)) == 1
    assert sum(message.text == "恢复资料" for message in session.messages) == 1
    assert _memory_messages(restored_provider.requests[0].messages) == _memory_messages(
        session.messages
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_fails", [False, True])
async def test_empty_memory_results_still_commit_input_before_provider(
    tmp_path: Path, lifecycle_store: LifecycleStore, provider_fails: bool
) -> None:
    """空记忆仍完成输入提交，provider 失败也不撤销已提交用户输入。"""

    class InspectingProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            session = lifecycle_store.load_session("default")
            assert [message.text for message in session.messages] == ["空记忆输入"]
            if provider_fails:
                raise IrisProviderError("provider 失败", provider="fake")
            return await super().complete(request)

    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=InspectingProvider(text_response())),
        store=lifecycle_store,
    )
    result = await runner.start(
        AgentRunRequest(input="空记忆输入", run_id="empty-memory"),
        options=AgentRunOptions(runtime=RuntimeExecutionOptions(memory_results=[])),
    )
    assert result.run.stop_reason is (
        RunStopReason.FAILED if provider_fails else RunStopReason.COMPLETED
    )
    session = lifecycle_store.load_session("default")
    assert session.messages[0].text == "空记忆输入"
    assert len(session.messages) == (1 if provider_fails else 2)
    assert _memory_messages(session.messages) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["input", "model"])
async def test_durable_cancellation_keeps_only_committed_input(
    tmp_path: Path,
    lifecycle_store: LifecycleStore,
    monkeypatch: pytest.MonkeyPatch,
    cancel_at: str,
) -> None:
    """取消输入准备不写部分输入，取消 provider 不撤销完整输入组。"""
    service, query = _query_service(tmp_path)
    input_started = asyncio.Event()
    release_input = asyncio.Event()
    original = service.abuild_context

    async def controlled_query(query: MemoryQuery, *, max_chars: int) -> MemoryContextBundle:
        input_started.set()
        if cancel_at == "input":
            await release_input.wait()
        return await original(query, max_chars=max_chars)

    monkeypatch.setattr(service, "abuild_context", controlled_query)
    provider = BlockingProvider()
    runtime = _with_context(build_runtime(tmp_path, provider=provider))
    runtime.environment.memory_service = service
    runner = AgentRunner(runtime=runtime, store=lifecycle_store)
    running = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="取消资料处理", run_id="memory-cancel"),
            options=AgentRunOptions(
                runtime=RuntimeExecutionOptions(memory_query=query.model_dump(mode="json"))
            ),
        )
    )
    started = input_started if cancel_at == "input" else provider.started
    await asyncio.wait_for(started.wait(), 2)
    result = await runner.cancel("memory-cancel", settlement_timeout=2)
    assert await running == result

    assert result.run.stop_reason is RunStopReason.CANCELLED
    session = lifecycle_store.load_session("default")
    assert len(session.messages) == (0 if cancel_at == "input" else 3)
    assert len(_memory_messages(session.messages)) == (0 if cancel_at == "input" else 1)
    assert result.run.usage.model_steps_reserved == (0 if cancel_at == "input" else 1)
    assert result.run.usage.model_steps_committed == 0


@pytest.mark.asyncio
async def test_explicit_query_failure_leaves_no_partial_input(
    tmp_path: Path, lifecycle_store: LifecycleStore
) -> None:
    """缺少显式查询服务是输入准备失败，不提交 BCI/user 或预留模型调用。"""
    provider = StaticProvider()
    query = MemoryQuery(
        scope=MemoryScope(workspace_id="workspace", agent_id="agent"), text="动态资料"
    )
    runner = AgentRunner(
        runtime=_with_context(build_runtime(tmp_path, provider=provider)), store=lifecycle_store
    )
    result = await runner.start(
        AgentRunRequest(input="不能准备记忆", run_id="memory-failed"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(memory_query=query.model_dump(mode="json"))
        ),
    )

    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None
    assert "memory_service" in result.error.message
    assert provider.requests == []
    assert lifecycle_store.load_session("default").messages == []
    assert result.run.usage.model_steps_reserved == result.run.usage.model_steps_committed == 0
