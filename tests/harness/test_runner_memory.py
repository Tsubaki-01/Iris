"""概览窗口随首次输入原子提交，并独立于普通消息历史持久重放。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from iris.agents import CompactionConfig
from iris.context import ContextSection, ContextSlot
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    LifecycleStore,
    RunStopReason,
    RuntimeExecutionOptions,
)
from iris.memory import MemoryAccessPolicy, MemoryService, SQLiteMemoryStore, register_memory_tools
from iris.memory.models import MemoryOverviewDocument
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response


@pytest.fixture(params=["memory", "sqlite"])
def lifecycle_store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """窗口持久化同时覆盖两个真实store。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "lifecycle.db")
    return InMemoryLifecycleStore()


class OverviewService(MemoryService):
    """只替换已发布文档读取，运行和提交继续使用真实runtime/store。"""

    def __init__(self, path: Path, text: str = "事实 A") -> None:
        super().__init__(SQLiteMemoryStore(path))
        self.documents = (overview(text),)
        self.reads: list[tuple[str, ...]] = []

    async def aload_overviews(
        self, namespaces: Sequence[str]
    ) -> tuple[MemoryOverviewDocument, ...]:
        """记录唯一的发布物加载入口。"""
        self.reads.append(tuple(namespaces))
        return self.documents


def overview(text: str, *, revision: int | None = 1) -> MemoryOverviewDocument:
    """构造一个含核心事实与知识范围的已发布文档或确定性缺产物说明。"""
    content = (
        text + "\n\n知识范围：资料对比与回答偏好。"
        if revision is not None
        else "当前没有已发布概览，本窗口暂不使用长期记忆。"
    )
    return MemoryOverviewDocument(
        namespace="project",
        path=Path(".iris/memory/namespaces/ns_project/Memory.md"),
        source_revision=revision,
        text=content,
        navigation=("知识范围：资料对比与回答偏好。" if revision is not None else content),
    )


def with_memory(runtime: AgentRuntime, service: MemoryService) -> AgentRuntime:
    """保留静态memory和BCI，以检查新概览不混入这两个历史位置。"""
    runtime.environment.memory_service = service
    runtime.environment.context_input = runtime.environment.context_input.model_copy(
        update={
            "memory": ContextSection(slots=[ContextSlot(name="static", content="固定记忆")]),
            "before_current_input": ContextSection(
                slots=[ContextSlot(name="environment", content="本轮环境")]
            ),
        }
    )
    return runtime


@pytest.mark.asyncio
async def test_window_is_committed_before_provider_and_replayed_in_tool_loop(
    tmp_path: Path, lifecycle_store: LifecycleStore
) -> None:
    """输入提交已经包含实际窗口，工具步骤只重放且不产生memory历史。"""
    service = OverviewService(tmp_path / "memory.db")
    registry = ToolRegistry()
    registry.register_function(lambda: "工具结果", name="read", description="读取资料")

    class InspectingProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            session = lifecycle_store.load_session("default")
            assert session.context_window is not None
            assert request.messages[0].text.endswith(session.context_window.memory_overview)
            checkpoint = lifecycle_store.load_checkpoint("memory-loop")
            assert checkpoint is not None and checkpoint.session_revision == session.revision
            assert not any(
                message.metadata.get("context_kind") == "memory" for message in session.messages
            )
            return await super().complete(request)

    provider = InspectingProvider(
        tool_response(ToolUseBlock(id="read-1", name="read", input={})), text_response()
    )
    result = await AgentRunner(
        runtime=with_memory(build_runtime(tmp_path, provider=provider, registry=registry), service),
        store=lifecycle_store,
    ).start(AgentRunRequest(input="比较资料", run_id="memory-loop"))
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert len(provider.requests) == 2
    assert provider.requests[0].messages[0].text == provider.requests[1].messages[0].text
    assert service.reads == [("project",)]
    session = lifecycle_store.load_session("default")
    assert "事实 A" in session.context_window.memory_overview
    assert sum(message.text == "比较资料" for message in session.messages) == 1
    assert all("事实 A" not in message.text for message in session.messages)
    assert all("固定记忆" not in message.text for message in session.messages)
    for request in provider.requests:
        assert "固定记忆" in request.messages[1].text
        assert request.messages[1].sender == "context"
        assert "事实 A" not in request.messages[1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("include_tools", [False, True])
async def test_window_tool_guidance_uses_effective_request_capabilities(
    tmp_path: Path, include_tools: bool
) -> None:
    """实际注册 Search/Fetch 后，include_tools 同时控制请求 schema 和首次窗口指引。"""
    service = OverviewService(tmp_path / "memory.db")
    registry = register_memory_tools(
        service=service,
        access_policy_factory=lambda context: MemoryAccessPolicy(),
        tool_names=["memory.search", "memory.fetch"],
    )
    provider = StaticProvider(text_response())
    result = await AgentRunner(
        runtime=with_memory(build_runtime(tmp_path, provider=provider, registry=registry), service),
        store=InMemoryLifecycleStore(),
    ).start(
        AgentRunRequest(input="查看覆盖主题"),
        options=AgentRunOptions(runtime=RuntimeExecutionOptions(include_tools=include_tools)),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    request = provider.requests[0]
    for tool_name in ("memory_search", "memory_fetch"):
        assert (tool_name in request.messages[0].text) is include_tools
    assert bool(request.tools) is include_tools


@pytest.mark.asyncio
async def test_same_window_keeps_old_overview_and_new_session_loads_current(
    tmp_path: Path, lifecycle_store: LifecycleStore
) -> None:
    service = OverviewService(tmp_path / "memory.db")
    provider = StaticProvider(text_response(), text_response(), text_response())
    runner = AgentRunner(
        runtime=with_memory(build_runtime(tmp_path, provider=provider), service),
        store=lifecycle_store,
    )
    assert (
        await runner.start(AgentRunRequest(input="第一次"))
    ).run.stop_reason is RunStopReason.COMPLETED
    service.documents = (overview("事实 B", revision=2),)
    assert (
        await runner.start(AgentRunRequest(input="同窗口"))
    ).run.stop_reason is RunStopReason.COMPLETED
    assert (
        await runner.start(AgentRunRequest(input="新窗口", session_id="new"))
    ).run.stop_reason is RunStopReason.COMPLETED
    assert service.reads == [("project",), ("project",)]
    assert provider.requests[0].messages[0].text == provider.requests[1].messages[0].text
    assert "事实 B" in provider.requests[2].messages[0].text
    assert lifecycle_store.load_session("new").context_window.sources[0].source_revision == 2


@pytest.mark.asyncio
async def test_missing_overview_disables_long_term_queries_for_the_window(
    tmp_path: Path, lifecycle_store: LifecycleStore
) -> None:
    """未生成概览正常聊天，但已初始化窗口不查询长期记忆或反复加载。"""
    service = OverviewService(tmp_path / "memory.db")
    service.documents = (overview("", revision=None),)
    provider = StaticProvider(text_response(), text_response())
    runner = AgentRunner(
        runtime=with_memory(build_runtime(tmp_path, provider=provider), service),
        store=lifecycle_store,
    )
    await runner.start(AgentRunRequest(input="第一次"))
    result = await runner.start(AgentRunRequest(input="第二次"))
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    window = lifecycle_store.load_session("default").context_window
    assert window.mode == "navigation"
    assert "没有概览则本窗口暂不使用长期记忆" in window.memory_overview
    assert "不查询长期记忆" in window.memory_overview
    assert "当前没有专用数据库读取工具" in window.memory_overview
    assert "read_file" not in window.memory_overview
    assert window.sources[0].source_revision is None
    assert service.reads == [("project",)]
    assert provider.requests[0].messages[0].text == provider.requests[1].messages[0].text


@pytest.mark.asyncio
async def test_namespaces_share_one_addendum_allowance(tmp_path: Path) -> None:
    """多个有效概览合并超限时保留每个 namespace 的知识范围和来源。"""
    service = OverviewService(tmp_path / "memory.db")
    service.documents = tuple(
        MemoryOverviewDocument(
            namespace=name,
            path=Path(f".iris/memory/namespaces/{name}/Memory.md"),
            source_revision=1,
            text="事实" * 500,
            navigation=f"知识范围：{name} 的资料对比。",
        )
        for name in ("project", "research")
    )

    class CountingProvider(StaticProvider):
        def estimate_input_tokens(self, request: LLMRequest) -> int:
            return sum(len(message.text) for message in request.messages)

    provider = CountingProvider(text_response())
    runtime = with_memory(build_runtime(tmp_path, provider=provider), service)
    memory = runtime.environment.agent_config.memory.model_copy(
        update={"read_namespaces": ["project", "research"]}
    )
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={"memory": memory}
    )
    store = InMemoryLifecycleStore()
    result = await AgentRunner(runtime=runtime, store=store).start(
        AgentRunRequest(input="比较资料")
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    window = store.load_session("default").context_window
    assert window.mode == "navigation"
    assert [source.namespace for source in window.sources] == ["project", "research"]
    assert "知识范围：project 的资料对比。" in window.memory_overview
    assert "知识范围：research 的资料对比。" in window.memory_overview
    assert service.reads == [("project", "research")]


@pytest.mark.asyncio
async def test_window_allowance_uses_floor_of_compaction_input_budget(tmp_path: Path) -> None:
    """按真实待发送请求采用：额度向下取整，分母不是生成预算或压缩触发额度。"""
    service = OverviewService(tmp_path / "memory.db")

    class BoundaryProvider(StaticProvider):
        def estimate_input_tokens(self, request: LLMRequest) -> int:
            assert any(message.text == "本次输入" for message in request.messages)
            system = request.messages[0].text
            if "事实 A" in system:
                return 20 + 201
            if "# Memory overview" in system:
                return 20 + 200
            return 20

    provider = BoundaryProvider(text_response())
    runtime = with_memory(build_runtime(tmp_path, provider=provider), service)
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={"compaction": CompactionConfig(input_budget_tokens=10001)}
    )
    store = InMemoryLifecycleStore()
    result = await AgentRunner(runtime=runtime, store=store).start(
        AgentRunRequest(input="本次输入")
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    window = store.load_session("default").context_window
    assert window.mode == "navigation"
    assert provider.requests[0].messages[0].text.endswith(window.memory_overview)


@pytest.mark.asyncio
@pytest.mark.parametrize("too_large", [False, True])
async def test_first_window_navigation_selection_is_durable_or_fails_before_input(
    tmp_path: Path, lifecycle_store: LifecycleStore, too_large: bool
) -> None:
    service = OverviewService(tmp_path / "memory.db", "超长事实" * 1000)
    if too_large:
        document = service.documents[0]
        service.documents = (
            MemoryOverviewDocument(
                namespace=document.namespace,
                path=document.path,
                source_revision=1,
                text=document.text,
                navigation="主题范围 " * 500,
            ),
        )

    class CountingProvider(StaticProvider):
        def estimate_input_tokens(self, request: LLMRequest) -> int:
            return sum(len(message.text) for message in request.messages)

    provider = CountingProvider(text_response())
    result = await AgentRunner(
        runtime=with_memory(build_runtime(tmp_path, provider=provider), service),
        store=lifecycle_store,
    ).start(AgentRunRequest(input="首次输入"))
    session = lifecycle_store.load_session("default")
    if too_large:
        assert result.run.stop_reason is RunStopReason.FAILED
        assert result.error.source == "context"
        assert session.context_window is None and not session.messages
        assert not provider.requests
    else:
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
        assert session.context_window.mode == "navigation"
        assert "超长事实" not in session.context_window.memory_overview
        assert (
            provider.requests[0].messages[0].text.endswith(session.context_window.memory_overview)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["input", "model"])
async def test_cancellation_preserves_only_committed_window_and_input(
    tmp_path: Path, lifecycle_store: LifecycleStore, monkeypatch: pytest.MonkeyPatch, cancel_at: str
) -> None:
    service = OverviewService(tmp_path / "memory.db")
    started = asyncio.Event()
    release = asyncio.Event()
    original = service.aload_overviews

    async def load(namespaces: Sequence[str]) -> tuple[MemoryOverviewDocument, ...]:
        started.set()
        if cancel_at == "input":
            await release.wait()
        return await original(namespaces)

    monkeypatch.setattr(service, "aload_overviews", load)
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=with_memory(build_runtime(tmp_path, provider=provider), service),
        store=lifecycle_store,
    )
    task = asyncio.create_task(runner.start(AgentRunRequest(input="取消", run_id="cancel-window")))
    await asyncio.wait_for((started if cancel_at == "input" else provider.started).wait(), 2)
    cancelled = await runner.cancel("cancel-window", settlement_timeout=2)
    assert await task == cancelled
    assert cancelled.run.stop_reason is RunStopReason.CANCELLED
    session = lifecycle_store.load_session("default")
    assert (session.context_window is None) is (cancel_at == "input")
    assert len(session.messages) == (0 if cancel_at == "input" else 2)
