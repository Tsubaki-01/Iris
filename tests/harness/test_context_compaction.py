"""通过真实 Runner 和 store 验证压缩、恢复与子 run 的核心链路。"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import iris.store.sqlite as sqlite_module
from iris.agents import CompactionConfig
from iris.context import ContextSection, ContextSlot
from iris.harness import AgentRunner, SessionHistory
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import (
    AgentRunRequest,
    LifecycleStore,
    RunEventKind,
    RunStopReason,
    SessionContextWindow,
)
from iris.memory import MemoryService
from iris.message import (
    LLMRequest,
    LLMResponse,
    ModelResponseCompleted,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from iris.runtime import AgentRuntime
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolRegistry

from .fakes import (
    FrozenClock,
    RecordingPublisher,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)
from .test_runner_subagent import ChildProviders, _write_configs


class CompactionProvider(StaticProvider):
    """摘要请求单独计量和响应；主请求保留真实消息大小变化。"""

    def __init__(self, *responses: LLMResponse, summary_prefix: str = "工作摘要") -> None:
        super().__init__(*responses)
        self.summary_prefix = summary_prefix
        self.summary_requests: list[LLMRequest] = []
        self.block_main = False
        self.main_started = asyncio.Event()
        self.release = asyncio.Event()

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """以字符数驱动主请求阈值，摘要分批由独立 runtime 测试覆盖。"""
        if request.provider_options.get("num_retries") == 0:
            return 100
        size = 0
        for message in request.messages:
            size += 4
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    size += len(block.text)
                elif isinstance(block, ToolResultBlock):
                    size += len(block.content)
                elif isinstance(block, ToolUseBlock):
                    size += len(block.id) + len(block.name)
        return size + (20 if request.tools else 0)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """分离摘要账本，并允许在已提交摘要之后模拟主请求中断。"""
        if request.provider_options.get("num_retries") == 0:
            self.summary_requests.append(request)
            return text_response(f"{self.summary_prefix}{len(self.summary_requests)}").model_copy(
                update={"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}
            )
        self.requests.append(request)
        self.main_started.set()
        if self.block_main:
            await self.release.wait()
        return self.responses.pop(0)

    async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """为 live publisher 用例提供最小合法主请求 stream。"""
        assert request.provider_options.get("num_retries") != 0
        response = await self.complete(request)
        scope = ModelStreamScope(
            model_stream_id=response.id, provider=response.provider, model=response.model, attempt=1
        )
        now = FrozenClock().now()
        yield ModelResponseStarted(
            scope=scope, sequence=1, occurred_at=now, response_id=response.id
        )
        yield ModelResponseCompleted(
            scope=scope,
            sequence=2,
            occurred_at=now,
            response=response,
            semantic_output_emitted=False,
        )


@pytest.fixture(params=["memory", "sqlite"])
def compaction_store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """关键历史与恢复流程使用两个真实 store。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "compaction.db")
    return InMemoryLifecycleStore()


def _runtime(
    tmp_path: Path, provider: CompactionProvider, *, registry: ToolRegistry | None = None
) -> AgentRuntime:
    runtime = build_runtime(tmp_path, provider=provider, registry=registry)
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={"compaction": CompactionConfig(input_budget_tokens=1000)}
    )
    runtime.environment.context_input = runtime.environment.context_input.model_copy(
        update={
            "before_current_input": ContextSection(
                slots=[ContextSlot(name="environment", content="本轮有效环境")]
            )
        }
    )
    return runtime


async def _seed_history(
    tmp_path: Path, store: LifecycleStore, *, memory_service: MemoryService | None = None
) -> None:
    runtime = build_runtime(tmp_path, provider=StaticProvider(text_response("旧" * 900)))
    runtime.environment.memory_service = memory_service
    await AgentRunner(
        runtime=runtime,
        store=store,
    ).start(AgentRunRequest(input="已有历史", run_id="seed"))


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_failure", [False, True])
async def test_disabled_memory_compaction_commits_empty_window_or_preserves_failed_summary(
    tmp_path: Path, compaction_store: LifecycleStore, summary_failure: bool
) -> None:
    """关闭后的真实压缩不靠旧概览触发；成功提交空窗口，摘要失败保留旧窗口与用量。"""
    from .test_runner_memory import OverviewService

    service = OverviewService(tmp_path / "memory.db", "关闭前的长期记忆")
    await _seed_history(tmp_path, compaction_store, memory_service=service)
    before = compaction_store.load_session("default")
    assert "关闭前的长期记忆" in before.context_window.memory_overview

    class SummaryProvider(CompactionProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            response = await super().complete(request)
            if summary_failure and request.provider_options.get("num_retries") == 0:
                return response.model_copy(update={"content": []})
            return response

    provider = SummaryProvider(text_response("关闭后压缩完成"))
    result = await AgentRunner(
        runtime=_runtime(tmp_path, provider), store=compaction_store
    ).start(AgentRunRequest(input="继续处理普通历史", run_id="disabled-compaction"))
    after = compaction_store.load_session("default")
    assert len(provider.summary_requests) == 1
    assert service.reads == [("project",)]
    assert after.messages[: len(before.messages)] == before.messages
    assert result.run.usage.compaction.total_tokens == 11
    if summary_failure:
        assert result.run.stop_reason is RunStopReason.FAILED
        assert result.error.code == "CONTEXT_COMPACTION_FAILED"
        assert after.compaction == before.compaction
        assert after.context_window == before.context_window
        assert provider.requests == []
        retry_provider = StaticProvider(text_response())
        retry = await AgentRunner(
            runtime=build_runtime(tmp_path, provider=retry_provider), store=compaction_store
        ).start(AgentRunRequest(input="失败后继续聊天"))
        assert retry.run.stop_reason is RunStopReason.COMPLETED, retry.error
        assert "关闭前的长期记忆" not in retry_provider.requests[0].messages[0].text
        assert compaction_store.load_session("default").context_window == before.context_window
    else:
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
        assert after.compaction is not None
        assert after.context_window == SessionContextWindow()
        assert "关闭前的长期记忆" not in provider.requests[0].messages[0].text
        assert "# Memory overview" not in provider.requests[0].messages[0].text
        assert after.revision == before.revision + 3
        checkpoint = compaction_store.load_checkpoint("disabled-compaction")
        assert checkpoint.session_revision == after.revision


@pytest.mark.asyncio
async def test_long_session_and_current_run_tool_steps_keep_raw_history_and_anchors(
    tmp_path: Path, compaction_store: LifecycleStore
) -> None:
    """旧 session 与同 run 新工具输出均可压缩，锚点及两套用量保持独立。"""
    await _seed_history(tmp_path, compaction_store)
    effects: list[int] = []

    def read_document(part: int) -> str:
        effects.append(part)
        return str(part) * 900

    registry = ToolRegistry()
    registry.register_function(read_document, description="读取长资料")
    provider = CompactionProvider(
        tool_response(ToolUseBlock(id="read-1", name="read_document", input={"part": 1})),
        tool_response(ToolUseBlock(id="read-2", name="read_document", input={"part": 2})),
        text_response("完成原始任务"),
    )
    result = await AgentRunner(
        runtime=_runtime(tmp_path, provider, registry=registry), store=compaction_store
    ).start(AgentRunRequest(input="比较两份资料", run_id="long-loop"))

    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert effects == [1, 2]
    assert len(provider.requests) == len(provider.summary_requests) == 3
    for request in provider.requests:
        assert sum(message.text == "比较两份资料" for message in request.messages) == 1
        assert sum("本轮有效环境" in message.text for message in request.messages) == 1
        assert sum(message.text.startswith("<summary>") for message in request.messages) == 1
        assert provider.estimate_input_tokens(request) <= 800
    session = compaction_store.load_session("default")
    assert session.compaction is not None and session.compaction.covered_message_count > 2
    assert sum(message.text == "比较两份资料" for message in session.messages) == 1
    assert sum("本轮有效环境" in message.text for message in session.messages) == 1
    assert sum(message.text == "旧" * 900 for message in session.messages) == 1
    assert [block.content for message in session.messages for block in message.tool_results] == [
        "1" * 900,
        "2" * 900,
    ]
    assert not any(message.text.startswith("<summary>") for message in session.messages)
    assert result.run.usage.model_steps_reserved == result.run.usage.model_steps_committed == 3
    assert result.run.usage.total_tokens == 21
    assert result.run.usage.compaction.total_tokens == 33
    assert len(compaction_store.list_tool_calls("long-loop")) == 2


@pytest.mark.asyncio
async def test_recover_after_projection_commit_reuses_summary_reservation_and_fork_snapshot(
    tmp_path: Path, compaction_store: LifecycleStore
) -> None:
    """step0 主响应前崩溃复用摘要，后续原会话更新不会改变旧 run 的 fork。"""
    from .test_runner_memory import OverviewService, overview

    await _seed_history(tmp_path, compaction_store)
    provider = CompactionProvider()
    provider.block_main = True
    service = OverviewService(tmp_path / "memory.db", "压缩时采用的概览")
    runtime = _runtime(tmp_path, provider)
    runtime.environment.memory_service = service
    memory = runtime.environment.agent_config.memory
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={
            "memory": memory.model_copy(
                update={"overview": memory.overview.model_copy(update={"system_budget_ratio": 0.5})}
            )
        }
    )
    running = asyncio.create_task(
        AgentRunner(runtime=runtime, store=compaction_store).start(
            AgentRunRequest(input="继续原任务", run_id="crashed")
        )
    )
    try:
        await asyncio.wait_for(provider.main_started.wait(), timeout=2)
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    before = compaction_store.load_session("default")
    crashed = compaction_store.load_run("crashed")
    assert before.compaction is not None
    assert "压缩时采用的概览" in before.context_window.memory_overview
    assert len(before.messages) == 4
    assert before.messages[-2].metadata["context_kind"] == "before_current_input"
    assert before.messages[-1].text == "继续原任务"
    assert crashed is not None and crashed.current_activation_id is not None
    assert crashed.usage.model_steps_reserved == 1
    assert crashed.usage.model_steps_committed == 0
    service.documents = (overview("恢复时不应重新采用", revision=2),)
    resumed_provider = CompactionProvider(text_response("新" * 900))
    resumed_runtime = _runtime(tmp_path, resumed_provider)
    resumed_runtime.environment.memory_service = service
    recovered = await AgentRunner(
        runtime=resumed_runtime, store=compaction_store
    ).recover("crashed", expected_activation_id=crashed.current_activation_id)

    assert recovered.run.stop_reason is RunStopReason.COMPLETED, recovered.error
    assert resumed_provider.summary_requests == []
    assert len(resumed_provider.requests) == 1
    assert before.compaction.summary in resumed_provider.requests[0].messages[1].text
    assert resumed_provider.requests[0].messages[0].text.endswith(
        before.context_window.memory_overview
    )
    assert service.reads == [("project",)]
    restored = compaction_store.load_session("default")
    assert restored.compaction == before.compaction
    assert restored.context_window == before.context_window
    assert sum(message.text == "继续原任务" for message in restored.messages) == 1
    assert sum("本轮有效环境" in message.text for message in restored.messages) == 1
    assert recovered.run.usage.compaction.total_tokens == 11
    assert (
        recovered.run.usage.model_steps_reserved == recovered.run.usage.model_steps_committed == 1
    )
    assert [event.kind for event in compaction_store.list_events("crashed")].count(
        RunEventKind.MODEL_STEP_RESERVED
    ) == 1

    later_provider = CompactionProvider(text_response("后续完成"), summary_prefix="后续摘要")
    later = await AgentRunner(
        runtime=_runtime(tmp_path, later_provider), store=compaction_store
    ).start(AgentRunRequest(input="新任务", run_id="later"))
    assert later.run.stop_reason is RunStopReason.COMPLETED, later.error
    assert compaction_store.load_session("default").compaction != before.compaction
    branch = SessionHistory(compaction_store).fork("crashed")
    assert branch.compaction == before.compaction
    assert branch.messages == restored.messages
    assert branch.context_window is None


@pytest.mark.asyncio
async def test_sqlite_projection_write_failure_preserves_old_summary_and_never_sends_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """关闭后的压缩提交失败保留旧摘要和非空窗口，不发送候选或抹掉已记录用量。"""
    from .test_runner_memory import OverviewService

    store = SQLiteStore(tmp_path / "fault.db")
    service = OverviewService(tmp_path / "memory.db", "写入失败后仍保存的旧概览")
    await _seed_history(tmp_path, store, memory_service=service)
    initial_runtime = _runtime(tmp_path, CompactionProvider(text_response("新" * 900)))
    initial_runtime.environment.memory_service = service
    memory = initial_runtime.environment.agent_config.memory
    initial_runtime.environment.agent_config = initial_runtime.environment.agent_config.model_copy(
        update={
            "memory": memory.model_copy(
                update={"overview": memory.overview.model_copy(update={"system_budget_ratio": 0.5})}
            )
        }
    )
    await AgentRunner(
        runtime=initial_runtime, store=store
    ).start(AgentRunRequest(input="第一次压缩", run_id="first"))
    before = store.load_session("default")
    assert before.compaction is not None
    assert "写入失败后仍保存的旧概览" in before.context_window.memory_overview
    original_execute = sqlite_module._execute

    def fail_projection_update(
        connection: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()
    ) -> sqlite3.Cursor:
        if "UPDATE sessions SET revision = ?, compaction_json" in sql:
            raise sqlite3.OperationalError("injected compaction write failure")
        return original_execute(connection, sql, params)

    monkeypatch.setattr(sqlite_module, "_execute", fail_projection_update)
    provider = CompactionProvider(text_response("不应发送"))
    publisher = RecordingPublisher()
    result = await AgentRunner(
        runtime=_runtime(tmp_path, provider), store=store, live_publisher=publisher
    ).start(AgentRunRequest(input="再次压缩", run_id="write-failed"))

    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None and result.error.source == "persistence"
    assert result.error.code == "RUN_PERSISTENCE_ERROR"
    assert provider.requests == []
    assert len(provider.summary_requests) == 1
    after = store.load_session("default")
    assert after.compaction == before.compaction
    assert after.context_window == before.context_window
    assert service.reads == [("project",), ("project",)]
    assert after.messages[:-2] == before.messages
    assert after.messages[-2].metadata["context_kind"] == "before_current_input"
    assert after.messages[-1].text == "再次压缩"
    assert store.load_run("write-failed").usage.compaction.total_tokens == 11
    assert [
        fact.kind for fact in publisher.facts if fact.kind.startswith("context.compaction")
    ] == ["context.compaction.started", "context.compaction.failed"]
    assert RunEventKind.CONTEXT_COMPACTED not in {
        event.kind for event in store.list_events("write-failed")
    }


@pytest.mark.asyncio
async def test_compaction_adopts_overview_outside_raw_history(
    tmp_path: Path, compaction_store: LifecycleStore
) -> None:
    """概览在压缩提交后进入system，原始历史继续只含普通消息。"""
    from .test_runner_memory import OverviewService

    await _seed_history(tmp_path, compaction_store)
    provider = CompactionProvider(text_response())
    runtime = _runtime(tmp_path, provider)
    runtime.environment.memory_service = OverviewService(tmp_path / "memory.db", "新概览事实")
    memory = runtime.environment.agent_config.memory
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={
            "memory": memory.model_copy(
                update={"overview": memory.overview.model_copy(update={"system_budget_ratio": 0.5})}
            )
        }
    )
    result = await AgentRunner(runtime=runtime, store=compaction_store).start(
        AgentRunRequest(input="保留原始问题", run_id="memory-compaction")
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert len(provider.summary_requests) == len(provider.requests) == 1
    request = provider.requests[0]
    session = compaction_store.load_session("default")
    assert session.compaction is not None
    assert session.context_window is not None
    assert "新概览事实" in session.context_window.memory_overview
    assert request.messages[0].text.endswith(session.context_window.memory_overview)
    assert not any(message.metadata.get("context_kind") == "memory" for message in session.messages)
    assert all("新概览事实" not in message.text for message in session.messages)
    assert sum("本轮有效环境" in message.text for message in request.messages) == 1
    assert sum(message.text == "保留原始问题" for message in request.messages) == 1


@pytest.mark.asyncio
async def test_child_waiting_resume_compacts_answer_without_repeating_tool_or_parent_usage(
    tmp_path: Path,
) -> None:
    """代理问答恢复后只推进原 child，child 的摘要用量不复制给 parent。"""
    config_path = _write_configs(tmp_path)
    child_path = tmp_path / "child.yaml"
    with child_path.open("a", encoding="utf-8") as config_file:
        config_file.write("compaction:\n  input_budget_tokens: 1000\n")
    parent = CompactionProvider(
        tool_response(ToolUseBlock(id="delegate", name="subagent", input={"prompt": "子任务"})),
        text_response("主任务完成"),
    )
    child = CompactionProvider(
        tool_response(
            ToolUseBlock(id="ask", name="ask_question", input={"question": "提供资料？"})
        ),
        text_response("子任务完成"),
    )
    runner = AgentRunner.from_config_path(
        config_path,
        provider=parent,
        store=SQLiteStore(tmp_path / "child-compaction.db"),
        child_provider_factory=ChildProviders(child),
    )
    waiting = await runner.start(AgentRunRequest(input="研究资料", run_id="parent"))
    assert waiting.pending_interaction is not None, waiting.error
    assert child.summary_requests == []
    result = await runner.resume(
        "parent",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="资料" * 450),
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    child_id = runner.store.load_subagent_link("parent", "delegate").child_run_id
    child_run = runner.store.load_run(child_id)
    assert child_run.stop_reason is RunStopReason.COMPLETED
    assert len(child.requests) == 2 and len(child.summary_requests) == 1
    assert len(parent.requests) == 2 and parent.summary_requests == []
    assert child_run.usage.total_tokens == result.run.usage.total_tokens == 13
    assert child_run.usage.compaction.total_tokens == 11
    assert result.run.usage.compaction.total_tokens == 0
    assert len(runner.store.list_tool_calls(child_id)) == 1
    assert len(runner.store.list_tool_calls("parent")) == 1
    assert sum(message.text == "子任务" for message in child.requests[-1].messages) == 1
