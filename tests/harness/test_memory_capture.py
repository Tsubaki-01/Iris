"""自动来源只捕获本 run 新原文，终态恢复和真实证据保留可追溯。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from iris.exceptions import IrisConfigError, IrisRunPersistenceError
from iris.harness import AgentRunner, SessionManager
from iris.harness._memory_capture import capture_episode, capture_source
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import (
    AgentRunRequest,
    ClaimToolCall,
    CreateRun,
    FinishRun,
    LifecycleStore,
    RunCommit,
    RunPhase,
    RunStopReason,
)
from iris.lifecycle.history import RunMessageSlice
from iris.memory import (
    MemoryItemPatch,
    MemoryObserveInput,
    MemorySearchQuery,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.memory.generation_models import (
    DreamPlan,
    GenerationResult,
    MemoryCaptureSource,
    MemoryGenerationConfig,
)
from iris.memory.mirror import FileMemoryMirror
from iris.message import LLMRequest, LLMResponse, Msg, ToolUseBlock
from iris.runtime.environment import RuntimeExecutionScope
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response


def _runner(
    tmp_path: Path,
    provider: StaticProvider,
    *,
    store: LifecycleStore | None = None,
    service: MemoryService | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[AgentRunner, MemoryService]:
    """绑定完整自动依赖，长 idle 让本测试只验证本地捕获。"""
    if service is None:
        generation_provider = StaticProvider()
        service = MemoryService(
            SQLiteMemoryStore(tmp_path / "memory.db"),
            mirror=FileMemoryMirror(tmp_path / "mirror", workspace_root=tmp_path),
            generation_provider=generation_provider,
            generation_model="generation",
            generation_config=MemoryGenerationConfig(idle_seconds=100),
            overview_provider=generation_provider,
            overview_model="overview",
        )
    runtime = build_runtime(tmp_path, provider=provider, registry=registry)
    config = runtime.environment.agent_config
    runtime.environment.agent_config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={
                    "enabled": True,
                    "generation": MemoryGenerationConfig(enabled=True, idle_seconds=100),
                }
            )
        }
    )
    runtime.environment.memory_service = service
    return AgentRunner(runtime=runtime, store=store or InMemoryLifecycleStore()), service


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_no_mcp_registers_before_main_model_and_captures_terminal(
    tmp_path: Path,
    fails: bool,
) -> None:
    """MCP already-prepared 不跳过记忆初始化，失败 run 同样保留已提交输入。"""

    class InspectProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            sources = service.store.list_capture_sources(runner.store.source_id, "project")
            assert len(sources) == 1
            assert sources[0].initial_message_count == 0
            if fails:
                raise RuntimeError("主模型失败")
            return text_response()

    runner, service = _runner(tmp_path, InspectProvider())
    result = await runner.start(AgentRunRequest(input="只用 uv", run_id="capture-run"))
    assert result.run.stop_reason is (RunStopReason.FAILED if fails else RunStopReason.COMPLETED)
    episodes = service.store.list_pending_episodes("project")
    assert len(episodes) == 1
    assert episodes[0].episode.records[0].text == "只用 uv"
    assert episodes[0].episode.metadata["outcome"] == result.run.stop_reason.value
    assert service.store.list_capture_sources(runner.store.source_id, "project") == []
    assert result.run.usage.total_tokens == (0 if fails else 5)
    await runner.aclose()


@pytest.mark.asyncio
async def test_waiting_capture_stays_open_and_cancel_seals_without_duplicate(
    tmp_path: Path,
) -> None:
    """WAITING 不是终态；取消封口且原文水位不回退或重复采集。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "written", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    runner, service = _runner(
        tmp_path,
        StaticProvider(tool_response(ToolUseBlock(id="write", name="write"))),
        registry=registry,
    )
    waiting = await runner.start(AgentRunRequest(input="记住这次尝试", run_id="waiting"))
    assert waiting.pending_interaction is not None
    source = service.store.list_capture_sources(runner.store.source_id, "project")[0]
    assert source.terminal_message_count is None
    first_ids = [p.episode.id for p in service.store.list_pending_episodes("project")]
    cancelled = await runner.cancel("waiting")
    assert cancelled.run.stop_reason is RunStopReason.CANCELLED
    assert service.store.list_capture_sources(runner.store.source_id, "project") == []
    episodes = service.store.list_pending_episodes("project")
    assert [p.episode.id for p in episodes[: len(first_ids)]] == first_ids
    record_ids = [record.id for progress in episodes for record in progress.episode.records]
    assert len(record_ids) == len(set(record_ids))
    assert episodes[-1].episode.metadata["outcome"] == "cancelled"
    await runner.aclose()


@pytest.mark.asyncio
async def test_resume_keeps_one_source_and_captures_only_suffix(tmp_path: Path) -> None:
    """已捕获 waiting 前缀不因同 run resume 被重复创建。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "written", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    runner, service = _runner(
        tmp_path,
        StaticProvider(tool_response(ToolUseBlock(id="write", name="write")), text_response()),
        registry=registry,
    )
    waiting = await runner.start(AgentRunRequest(input="执行", run_id="resume"))
    result = await runner.resume(
        "resume",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    episodes = service.store.list_pending_episodes("project")
    assert len(episodes) == 2
    ids = [record.id for progress in episodes for record in progress.episode.records]
    assert len(ids) == len(set(ids))
    assert service.store.list_capture_sources(runner.store.source_id, "project") == []
    await runner.aclose()


def test_capture_excludes_bci_and_memory_readback_but_keeps_write_targets() -> None:
    """消息块内容进入证据，reasoning/BCI/已存记忆正文不进入新事实。"""
    source = MemoryCaptureSource(
        lifecycle_source_id="store",
        run_id="run",
        session_id="session",
        namespace="project",
        initial_message_count=4,
        captured_until=4,
    )
    messages = (
        Msg.user("生成环境", metadata={"context_kind": "before_current_input"}),
        Msg.system("系统指令"),
        Msg.user("请改用 uv", metadata={"reasoning": "不会保存"}),
        Msg.assistant(
            content=[ToolUseBlock(id="fetch", name="memory_fetch", input={"item_id": "existing"})]
        ),
        Msg.tool_result(
            tool_use_id="fetch",
            name="memory_fetch",
            content=json.dumps({"item": {"id": "existing", "text": "旧记忆正文"}}),
        ),
        Msg.tool_result(
            tool_use_id="search",
            name="memory_search",
            content=json.dumps({"items": [{"item_id": "hit", "snippet": "搜索片段"}]}),
        ),
        Msg.tool_result(
            tool_use_id="remember",
            name="memory_remember",
            content=json.dumps({"item": {"id": "new", "text": "项目使用 uv"}}),
        ),
    )
    updated, episode = capture_episode(
        source,
        RunMessageSlice(
            source_id="store",
            run_id="run",
            session_id="session",
            initial_message_count=4,
            start_message_count=4,
            end_message_count=11,
            terminal_message_count=11,
            outcome=RunStopReason.COMPLETED,
            messages=messages,
        ),
    )
    assert updated.captured_until == updated.terminal_message_count == 11
    assert episode.source_id == source.run_id
    assert "run_id" not in episode.metadata
    records = episode.records
    assert records[0].text == "请改用 uv"
    assert records[0].id == "store:run:6:0"
    assert all("reasoning" not in record.metadata for record in records)
    assert [record.text for record in records[1:4]] == ["", "", ""]
    assert records[2].metadata["memory_item_ids"] == ["existing"]
    assert records[3].metadata["memory_item_ids"] == ["hit"]
    assert records[4].metadata["memory_item_ids"] == ["new"]


def test_child_does_not_install_automatic_memory_maintenance(tmp_path: Path) -> None:
    """Child 范围明确禁用自动生成，独立的显式读写服务继续可用。"""
    runtime = build_runtime(tmp_path, provider=StaticProvider())
    runtime.environment.execution_scope = RuntimeExecutionScope.CHILD
    config = runtime.environment.agent_config
    runtime.environment.agent_config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={
                    "enabled": True,
                    "generation": MemoryGenerationConfig(enabled=True),
                }
            )
        }
    )
    runtime.environment.memory_service = MemoryService(SQLiteMemoryStore(tmp_path / "child.db"))
    runner = AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())
    assert runner._memory_maintenance is None
    assert runtime.environment.memory_capture_port is None


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["finalize", "unknown"])
async def test_recovery_terminal_without_activation_seals_memory_source(
    tmp_path: Path,
    disposition: str,
) -> None:
    """FINALIZE/OUTCOME_UNKNOWN 绕过 runtime 的恢复仍封口来源。"""

    class CrashStore(InMemoryLifecycleStore):
        failed = False

        def finish_run(self, command: FinishRun) -> RunCommit:
            if disposition == "finalize" and not self.failed:
                self.failed = True
                raise IrisRunPersistenceError("finish crash")
            return super().finish_run(command)

        def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
            result = super().claim_tool_call(command)
            if disposition == "unknown":
                raise IrisRunPersistenceError("claim crash")
            return result

    store = CrashStore()
    registry = ToolRegistry()
    registry.register_function(lambda: "effect", name="effect", description="执行")
    provider = StaticProvider(
        text_response()
        if disposition == "finalize"
        else tool_response(ToolUseBlock(id="effect", name="effect"))
    )
    first, service = _runner(tmp_path, provider, store=store, registry=registry)
    with pytest.raises(IrisRunPersistenceError):
        await first.start(AgentRunRequest(input="捕获恢复", run_id="crash"))
    crashed = store.load_run("crash")
    await first.aclose()
    sources = service.store.list_capture_sources(store.source_id, "project")
    assert len(sources) == 1
    resumed, _ = _runner(
        tmp_path, StaticProvider(), store=store, service=service, registry=registry
    )
    result = await resumed.recover("crash", expected_activation_id=crashed.current_activation_id)
    assert result.run.stop_reason is (
        RunStopReason.COMPLETED if disposition == "finalize" else RunStopReason.OUTCOME_UNKNOWN
    )
    assert service.store.list_capture_sources(store.source_id, "project") == []
    await resumed.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_changed", [False, True])
async def test_prepare_reopens_only_changed_budget_blocked_input(
    tmp_path: Path,
    budget_changed: bool,
) -> None:
    """只有 blocked 输入时也在装配准备重评预算变化，原预算保持 blocked。"""
    runner, service = _runner(tmp_path, StaticProvider())
    item = service.remember(MemoryWriteInput(text="项目使用 uv", reason="项目约定"))
    snapshot = service.store.read_dream_snapshot("project")
    budget = service.generation_config.dream_input_budget_tokens
    assert service.store.block_dream(
        snapshot,
        reason="完整材料超限",
        budget=100 if budget_changed else budget,
        dependency_item_ids=[item.id],
    )
    await runner.aprepare()
    state = service.generation_state("project")
    assert state.pending_changes == int(budget_changed)
    assert state.blocked_changes == int(not budget_changed)
    await runner.aclose()


@pytest.mark.asyncio
async def test_long_active_run_blocks_generation_past_idle(tmp_path: Path) -> None:
    """真实 runner 的完整调用范围覆盖长期 provider 等待，不能仅依据最近输入时间。"""
    provider = BlockingProvider()
    runner, service = _runner(tmp_path, provider)
    service.generation_config = service.generation_config.model_copy(update={"idle_seconds": 0.01})
    service.observe(MemoryObserveInput(text="已有待处理经历"))
    running = asyncio.create_task(runner.start(AgentRunRequest(input="等待主任务")))
    await asyncio.wait_for(provider.started.wait(), 1)
    await asyncio.sleep(0.04)
    assert service.generation_provider.requests == []
    provider.release.set()
    await running
    await runner.aclose()


@pytest.mark.asyncio
async def test_capture_failure_does_not_change_run_result_and_close_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地捕获失败保留来源水位与独立错误，后续边界补捕获。"""
    runner, service = _runner(tmp_path, StaticProvider(text_response()))
    commit = service.store.commit_capture

    def fail_capture(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("capture IO unavailable")

    monkeypatch.setattr(service.store, "commit_capture", fail_capture)
    result = await runner.start(AgentRunRequest(input="主任务正常", run_id="capture-failure"))
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert service.generation_state("project").latest_results[0].stage == "capture"
    assert service.generation_state("project").latest_results[0].status == "failed"
    assert (
        service.store.list_capture_sources(runner.store.source_id, "project")[0].captured_until == 0
    )
    monkeypatch.setattr(service.store, "commit_capture", commit)
    await runner.aclose()
    assert service.store.list_capture_sources(runner.store.source_id, "project") == []
    assert len(service.store.list_pending_episodes("project")) == 1


@pytest.mark.asyncio
async def test_prepare_recovers_terminal_capture_without_later_session_messages(
    tmp_path: Path,
) -> None:
    """重启补读已登记终态来源，只读该 run 截点，后续会话消息不串入。"""
    store = InMemoryLifecycleStore()
    restarted, service = _runner(tmp_path, StaticProvider(), store=store)

    class RegisterProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            run = store.load_run("old")
            service.store.register_source(
                capture_source(run, source_id=store.source_id, namespace="project")
            )
            return text_response("旧结果")

    first = AgentRunner(runtime=build_runtime(tmp_path, provider=RegisterProvider()), store=store)
    await first.start(AgentRunRequest(input="旧输入", run_id="old"))
    later = AgentRunner(
        runtime=build_runtime(tmp_path, provider=StaticProvider(text_response("后续结果"))),
        store=store,
    )
    await later.start(AgentRunRequest(input="后续输入", run_id="later"))
    await restarted.aprepare()
    records = service.store.list_pending_episodes("project")[0].episode.records
    assert [record.text for record in records] == ["旧输入", "旧结果"]
    assert service.store.list_capture_sources(store.source_id, "project") == []
    await restarted.aclose()


@pytest.mark.asyncio
async def test_external_service_revision_during_overview_schedules_republication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共享库的独立服务没有本地通知，陈旧概览仍留下一次发布工作。"""
    runner, service = _runner(tmp_path, StaticProvider())
    service.generation_config = service.generation_config.model_copy(update={"idle_seconds": 0.01})
    item = service.remember(MemoryWriteInput(text="旧事实", reason="初始化"))

    def consume_explicit(target: MemoryService) -> None:
        snapshot = target.store.read_dream_snapshot("project")
        assert target.store.commit_dream(
            snapshot,
            DreamPlan(),
            result=GenerationResult(namespace="project", stage="dream", status="completed"),
        )

    consume_explicit(service)
    peer = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=service.mirror)

    class UpdatingOverviewProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                peer.update(item.id, "project", MemoryItemPatch(text="新事实"), reason="外部修正")
                consume_explicit(peer)
            return text_response(
                json.dumps({"core_facts": "当前事实", "knowledge_scope": "项目事实"})
            )

    provider = UpdatingOverviewProvider()
    service.overview_provider = provider
    refreshed = asyncio.Event()
    refresh = service.refresh_overview

    async def observe_refresh(namespace: str) -> object:
        result = await refresh(namespace)
        if result.source_revision == 2:
            refreshed.set()
        return result

    monkeypatch.setattr(service, "refresh_overview", observe_refresh)
    await runner.aprepare()
    await asyncio.wait_for(refreshed.wait(), 1)
    await asyncio.sleep(0.03)
    assert len(provider.requests) == 2
    assert service.generation_provider.requests == []
    assert service.generation_state("project").overview_revision == 2
    await runner.aclose()


@pytest.mark.asyncio
async def test_sqlite_restart_captures_terminal_tail_once(tmp_path: Path) -> None:
    """终态已落 SQLite 但尚未 capture 时，重开来源库和记忆库仍补采一次。"""
    lifecycle_path = tmp_path / "lifecycle.db"
    first_store = SQLiteStore(lifecycle_path)
    memory_store = SQLiteMemoryStore(tmp_path / "memory.db")

    class RegisteredProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            memory_store.register_source(
                capture_source(
                    first_store.load_run("interrupted-capture"),
                    source_id=first_store.source_id,
                    namespace="project",
                )
            )
            return text_response("已完成但未捕获")

    original = AgentRunner(
        runtime=build_runtime(tmp_path, provider=RegisteredProvider()),
        store=first_store,
    )
    result = await original.start(
        AgentRunRequest(input="新增持久事实", run_id="interrupted-capture")
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert memory_store.list_pending_episodes("project") == []
    await original.aclose()

    reopened_store = SQLiteStore(lifecycle_path)
    assert reopened_store.source_id == first_store.source_id
    restarted, memory = _runner(tmp_path, StaticProvider(), store=reopened_store)
    await restarted.aprepare()
    first_episode = memory.store.list_pending_episodes("project")[0].episode
    assert [record.text for record in first_episode.records] == ["新增持久事实", "已完成但未捕获"]
    assert first_episode.metadata["outcome"] == "completed"
    assert memory.store.list_capture_sources(reopened_store.source_id, "project") == []
    await restarted.aprepare()
    await restarted.aclose()

    again, memory_again = _runner(tmp_path, StaticProvider(), store=SQLiteStore(lifecycle_path))
    await again.aprepare()
    assert [entry.episode.id for entry in memory_again.store.list_pending_episodes("project")] == [
        first_episode.id
    ]
    await again.aclose()


@pytest.mark.parametrize(
    "missing",
    [
        "generation_provider",
        "generation_model",
        "overview_provider",
        "overview_model",
        "mirror",
    ],
)
def test_automatic_generation_rejects_incomplete_injected_dependencies(
    tmp_path: Path,
    missing: str,
) -> None:
    """装配一次确认完整流水线依赖，避免构造注定无法发布的后台任务。"""
    configured, service = _runner(tmp_path, StaticProvider())
    setattr(service, missing, None)
    with pytest.raises(IrisConfigError, match="自动记忆生成"):
        AgentRunner(runtime=configured.runtime, store=InMemoryLifecycleStore())


@pytest.mark.asyncio
async def test_disabled_memory_ignores_incomplete_injected_generation_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """memory总开关关闭时不校验、准备或关闭宿主注入的生成依赖。"""
    configured, service = _runner(tmp_path, StaticProvider())
    service.generation_provider = None
    service.generation_model = None
    service.overview_provider = None
    service.overview_model = None
    service.mirror = None

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("关闭memory后不应操作注入service")

    for method in ("run_async_io", "add_change_listener", "wait_pending_io"):
        monkeypatch.setattr(service, method, forbidden)
    config = configured.runtime.environment.agent_config
    config = config.model_copy(
        update={"memory": config.memory.model_copy(update={"enabled": False})}
    )
    runner = AgentRunner.from_config(
        config,
        provider=StaticProvider(text_response()),
        memory_service=service,
    )
    assert runner.runtime.environment.memory_service is None
    assert runner._memory_maintenance is None
    await runner.aprepare()
    await runner.start(AgentRunRequest(input="正常聊天"))
    await runner.aclose()


@pytest.mark.asyncio
async def test_root_run_idle_generation_publishes_memory_without_refreshing_old_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实SQLite流水线从run原文产观察/知识/概览，旧窗口稳定，新会话采用发布物。"""
    dream_started = asyncio.Event()
    allow_dream = asyncio.Event()

    class GenerationProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            source = json.loads(request.messages[1].text)
            if request.model == "overview":
                payload = {"core_facts": "本项目使用 uv", "knowledge_scope": "项目依赖管理约定"}
            elif "records" in source:
                evidence = next(
                    record["ref"] for record in source["records"] if "uv" in record["text"]
                )
                payload = {
                    "observations": [
                        {
                            "text": "本项目使用 uv",
                            "applicability": "本项目",
                            "reason": "用户明确指定",
                            "category": "reference",
                            "kind": "fact",
                            "evidence": [evidence],
                        }
                    ]
                }
            else:
                dream_started.set()
                await allow_dream.wait()
                observation = source["observations"][0]
                payload = {
                    "operations": [
                        {
                            "action": "add",
                            "new_key": "project-uv",
                            "text": "本项目使用 uv",
                            "category": "reference",
                            "kind": "fact",
                            "reason": "正式项目约定",
                            "evidence": observation["evidence"],
                        }
                    ],
                    "resolutions": [
                        {
                            "observation_id": observation["id"],
                            "target_id": "project-uv",
                            "reason": "归入项目约定",
                        }
                    ],
                }
            return text_response(json.dumps(payload, ensure_ascii=False))

    background_provider = GenerationProvider()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror", workspace_root=tmp_path),
        generation_provider=background_provider,
        generation_model="generation",
        generation_config=MemoryGenerationConfig(idle_seconds=0.01),
        overview_provider=background_provider,
        overview_model="overview",
    )
    main_provider = StaticProvider(
        text_response("收到"), text_response("继续"), text_response("新会话")
    )
    runner, _ = _runner(
        tmp_path,
        main_provider,
        service=service,
        store=SQLiteStore(tmp_path / "lifecycle.db"),
    )
    published = asyncio.Event()
    refresh = service.refresh_overview

    async def observe_publish(namespace: str) -> object:
        result = await refresh(namespace)
        published.set()
        return result

    monkeypatch.setattr(service, "refresh_overview", observe_publish)
    result = await runner.start(AgentRunRequest(input="本项目使用 uv 管理依赖", run_id="root"))
    original_window = runner.store.load_session("default").context_window
    assert result.run.usage.total_tokens == 5
    await asyncio.wait_for(dream_started.wait(), 2)
    assert service.generation_state("project").pending_observations == 1
    assert service.search(MemorySearchQuery(query="uv"), ["project"]).items == ()
    allow_dream.set()
    await asyncio.wait_for(published.wait(), 2)
    assert len(background_provider.requests) == 3
    assert (
        service.search(MemorySearchQuery(query="uv"), ["project"]).items[0].snippet
        == "本项目使用 uv"
    )
    assert service.generation_state("project").overview_revision == 1
    assert runner.store.load_session("default").context_window == original_window
    assert result.run.usage.total_tokens == 5
    assert runner.store.load_run("root").usage.total_tokens == 5
    await runner.start(AgentRunRequest(input="继续", run_id="next"))
    assert runner.store.load_session("default").context_window == original_window
    assert "本项目使用 uv" not in main_provider.requests[1].messages[0].text
    await runner.start(AgentRunRequest(input="新会话", session_id="fresh", run_id="fresh"))
    assert "本项目使用 uv" in main_provider.requests[2].messages[0].text
    assert runner.store.load_session("fresh").context_window.sources[0].source_revision == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_waiting_activation_cleanup_finishes_before_idle_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """durable WAITING 先于live清理出现时，后台模型仍等待原调用真正退出。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "written", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    runner, service = _runner(
        tmp_path,
        StaticProvider(tool_response(ToolUseBlock(id="write", name="write"))),
        registry=registry,
    )
    service.generation_config = service.generation_config.model_copy(update={"idle_seconds": 0.01})
    service.observe(MemoryObserveInput(text="已有材料"))
    generation_started = asyncio.Event()

    class FlushProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            generation_started.set()
            return text_response('{"observations": []}')

    generation_provider = FlushProvider()
    service.generation_provider = generation_provider
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    settle = runner._settle_live_resources

    async def gated_cleanup(active: object) -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        await settle(active)

    monkeypatch.setattr(runner, "_settle_live_resources", gated_cleanup)
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="执行", run_id="waiting-cleanup"))
    )
    await asyncio.wait_for(cleanup_started.wait(), 1)
    assert runner.store.load_run("waiting-cleanup").phase is RunPhase.WAITING
    assert runner._active
    await asyncio.sleep(0.04)
    assert generation_provider.requests == []
    release_cleanup.set()
    await running
    assert not runner._active
    await asyncio.wait_for(generation_started.wait(), 1)
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_seconds", [0, 0.01])
async def test_managed_follow_up_has_priority_over_idle_generation(
    tmp_path: Path,
    idle_seconds: float,
) -> None:
    """排队follow-up走同一foreground入口，前一run结束不会抢先启动维护。"""
    started = (asyncio.Event(), asyncio.Event())
    releases = (asyncio.Event(), asyncio.Event())
    timeline: list[str] = []

    class MainProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            index = len(self.requests)
            self.requests.append(request)
            timeline.append(f"main-{index + 1}")
            started[index].set()
            await releases[index].wait()
            return text_response()

    runner, service = _runner(tmp_path, MainProvider())
    service.generation_config = service.generation_config.model_copy(
        update={"idle_seconds": idle_seconds}
    )
    service.observe(MemoryObserveInput(text="已有材料"))
    generation_started = asyncio.Event()

    class FlushProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            timeline.append(f"flush-foreground-{runner._memory_maintenance._foreground}")
            generation_started.set()
            return text_response('{"observations": []}')

    generation_provider = FlushProvider()
    service.generation_provider = generation_provider
    manager = SessionManager(runner, "managed")
    first = await manager.submit("先完成第一步")
    await asyncio.wait_for(started[0].wait(), 1)
    follow_up = await manager.submit("然后完成第二步", mode="follow_up")
    releases[0].set()
    await asyncio.wait_for(started[1].wait(), 1)
    assert runner.store.load_result(first.run_id).run.stop_reason is RunStopReason.COMPLETED
    assert runner.store.load_run(follow_up.run_id).phase is RunPhase.ACTIVE
    await asyncio.sleep(0.04)
    assert generation_provider.requests == [], timeline
    releases[1].set()
    await asyncio.wait_for(generation_started.wait(), 1)
    assert runner.store.load_result(follow_up.run_id).run.stop_reason is RunStopReason.COMPLETED
    await manager.close()
    await runner.aclose()


@pytest.mark.asyncio
async def test_follow_up_admission_failure_releases_memory_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """后继create失败后释放交接凭证，已捕获材料仍能进入空闲维护。"""
    main = BlockingProvider()
    runner, service = _runner(tmp_path, main)
    service.generation_config = service.generation_config.model_copy(update={"idle_seconds": 0})
    create = runner.store.create_run

    def fail_follow_up(command: CreateRun) -> RunCommit:
        if command.request.input == "失败后继":
            raise IrisRunPersistenceError("follow-up admission failed")
        return create(command)

    monkeypatch.setattr(runner.store, "create_run", fail_follow_up)
    generated = asyncio.Event()

    class FlushProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            generated.set()
            return text_response('{"observations": []}')

    service.generation_provider = FlushProvider()
    manager = SessionManager(runner, "managed-failure")
    await manager.submit("第一步")
    await asyncio.wait_for(main.started.wait(), 1)
    follow_up = await manager.submit("失败后继", mode="follow_up")
    main.release.set()
    await asyncio.wait_for(generated.wait(), 1)
    assert runner.store.load_run(follow_up.run_id) is None
    assert manager._memory_handoffs == set()
    assert not runner._memory_maintenance.foreground_active
    await manager.close()
    await runner.aclose()


@pytest.mark.asyncio
async def test_manager_close_releases_ready_follow_up_memory_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """terminal事件已预留后继但旧activation还在清理时，close释放未派发凭证。"""
    main = BlockingProvider()
    runner, _ = _runner(tmp_path, main)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    settle = runner._settle_live_resources

    async def gated_cleanup(active: object) -> None:
        cleanup_started.set()
        await cleanup_release.wait()
        await settle(active)

    monkeypatch.setattr(runner, "_settle_live_resources", gated_cleanup)
    manager = SessionManager(runner, "managed-close")
    await manager.submit("第一步")
    await asyncio.wait_for(main.started.wait(), 1)
    follow_up = await manager.submit("不再启动的后继", mode="follow_up")
    main.release.set()
    await asyncio.wait_for(cleanup_started.wait(), 1)
    assert manager._memory_handoffs == {follow_up.run_id}
    closing = asyncio.create_task(manager.close(cancel_run=True))
    await asyncio.sleep(0)
    assert manager._memory_handoffs == set()
    assert runner.store.load_run(follow_up.run_id) is None
    cleanup_release.set()
    await asyncio.wait_for(closing, 1)
    assert not runner._memory_maintenance.foreground_active
    await runner.aclose()
