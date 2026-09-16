"""SessionHistory 的历史分支与既有运行入口集成测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.exceptions import IrisRunStateError
from iris.harness import AgentRunner, SessionHistory, SessionManager
from iris.lifecycle import (
    AgentRunRequest,
    ForkPointCursor,
    ForkPointPage,
    ForkSession,
    LifecycleStore,
    RunEvent,
    RunEventKind,
    RunHistorySnapshot,
    RunPhase,
    RunStopReason,
    SessionSnapshot,
)
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response


@pytest.fixture(params=["memory", "sqlite"])
def history_store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """使用两种真实 store 验证同一 SDK 契约。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "history.db")
    return InMemoryLifecycleStore()


@pytest.mark.asyncio
async def test_fork_creates_history_then_starts_independently(
    tmp_path: Path, history_store: LifecycleStore
) -> None:
    """旧 run 前缀独立复制，fork 本身不请求 provider 或启动 run。"""
    provider = StaticProvider(
        text_response("first"), text_response("later"), text_response("branch")
    )
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=history_store)
    await runner.start(AgentRunRequest(input="one", session_id="main", run_id="r1"))
    await runner.start(AgentRunRequest(input="two", session_id="main", run_id="r2"))
    history = SessionHistory(history_store)
    source_before = history_store.load_session("main")
    before_calls = len(provider.requests)
    branch = history.fork("r1")
    sibling = history.fork("r1")
    assert branch.session_id != sibling.session_id
    assert branch.session_id != "main"
    assert len(provider.requests) == before_calls
    assert branch.messages == source_before.messages[:2]
    assert branch.revision == 0
    assert history.list_fork_points(branch.session_id).items == ()
    assert history_store.load_session_lane(branch.session_id) is None
    preview = history.get_at_run("r1")
    assert preview.messages == tuple(branch.messages)
    assert history.list_fork_points("main").items[0] == preview.point
    result = await runner.start(AgentRunRequest(input="alternate", session_id=branch.session_id))
    assert result.run.run_id not in {"r1", "r2"}
    assert result.run.usage.model_steps_reserved == 1
    assert result.run.usage.model_steps_committed == 1
    assert [message.text for message in provider.requests[-1].messages][-3:] == [
        "one",
        "first",
        "alternate",
    ]
    assert history_store.load_session("main") == source_before
    continued = history_store.load_session(branch.session_id)
    assert continued.forked_from_run_id == "r1"
    assert continued.revision == 1


@pytest.mark.asyncio
async def test_fork_does_not_interrupt_blocked_source_run(
    tmp_path: Path, history_store: LifecycleStore
) -> None:
    """来源后续轮次停在 provider 时，旧截点分支立即可用且不影响其运行。"""
    first_runner = AgentRunner(runtime=build_runtime(tmp_path), store=history_store)
    await first_runner.start(AgentRunRequest(input="one", session_id="main", run_id="r1"))
    provider = BlockingProvider(text_response("later"))
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=history_store)
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="two", session_id="main", run_id="r2"))
    )
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        before = history_store.load_run("r2")
        branch = SessionHistory(history_store).fork("r1")
        assert len(branch.messages) == 2
        assert not running.done()
        assert not provider.release.is_set()
        assert history_store.load_run("r2") == before
        assert history_store.load_session_lane("main") == "r2"
        assert history_store.load_session_lane(branch.session_id) is None
        assert len(provider.requests) == 1
    finally:
        provider.release.set()
        result = await running
    assert result.run.stop_reason is RunStopReason.COMPLETED


@pytest.mark.asyncio
async def test_branch_uses_new_runner_environment_and_fresh_execution_state(
    tmp_path: Path, history_store: LifecycleStore
) -> None:
    """继承历史由新 runner 重新组装 system，执行事实从 fresh run 开始。"""
    original = AgentRunner(
        runtime=build_runtime(tmp_path, system_text="original system"), store=history_store
    )
    await original.start(AgentRunRequest(input="one", session_id="main", run_id="r1"))
    source = history_store.load_run("r1")
    source_checkpoint = history_store.load_checkpoint("r1")
    branch = SessionHistory(history_store).fork("r1")
    provider = BlockingProvider(text_response("branch answer"))
    changed = AgentRunner(
        runtime=build_runtime(
            tmp_path, system_text="new system", agent_name="branch-agent", provider=provider
        ),
        store=history_store,
    )
    running = asyncio.create_task(
        changed.start(AgentRunRequest(input="alternate", session_id=branch.session_id))
    )
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        run_id = history_store.load_session_lane(branch.session_id)
        current = history_store.load_run(run_id)
        checkpoint = history_store.load_checkpoint(run_id)
        assert current.run_id != "r1"
        assert current.phase is RunPhase.ACTIVE
        assert current.agent_id == "branch-agent"
        assert current.usage.model_steps_reserved == 1
        assert current.usage.model_steps_committed == 0
        assert current.usage.tool_calls_committed == 0
        assert checkpoint.sequence == 1
        assert checkpoint.session_revision == 0
        assert checkpoint.activation_id != source_checkpoint.activation_id
        assert checkpoint.engine_cursor["position"] == "before_model"
        assert checkpoint.engine_cursor["step_index"] == 0
        messages = provider.requests[0].messages
        assert messages[0].role.value == "system"
        assert "new system" in messages[0].text
        assert "original system" not in messages[0].text
        assert [message.text for message in messages][-3:] == ["one", "完成", "alternate"]
    finally:
        provider.release.set()
        result = await running
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert history_store.load_run("r1") == source


@pytest.mark.asyncio
async def test_new_session_manager_submits_directly_on_branch(
    tmp_path: Path, history_store: LifecycleStore
) -> None:
    """新 manager 可直接管理分支，通过既有终态事件结束本轮。"""
    provider = StaticProvider(text_response("first"), text_response("managed branch"))
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=history_store)
    await runner.start(AgentRunRequest(input="one", session_id="main", run_id="r1"))
    branch = SessionHistory(history_store).fork("r1")
    manager = SessionManager(runner, branch.session_id)
    events = manager.events()
    try:
        receipt = await manager.submit("continue")
        async with asyncio.timeout(2):
            async for event in events:
                if (
                    isinstance(event, RunEvent)
                    and event.run_id == receipt.run_id
                    and event.kind is RunEventKind.RUN_TERMINAL
                ):
                    break
            else:
                pytest.fail("分支运行未发布终态事件")
        result = history_store.load_result(receipt.run_id)
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert result.assistant_message.text == "managed branch"
        assert history_store.load_session(branch.session_id).forked_from_run_id == "r1"
        assert [message.text for message in provider.requests[-1].messages][-3:] == [
            "one",
            "first",
            "continue",
        ]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_sdk_delegates_exact_arguments_and_domain_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDK 直接传递来源、游标和容量，返回 store 结果并原样传播领域错误。"""
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    await runner.start(AgentRunRequest(input="one", session_id="main", run_id="r1"))
    page = store.list_fork_points("main")
    preview = store.load_session_at_run("r1")
    branch = SessionSnapshot(session_id="store-result", messages=list(preview.messages))
    cursor = ForkPointCursor(created_at=page.items[0].created_at, run_id="r1")
    calls: list[object] = []
    error = IrisRunStateError("store rejected limit")

    def list_points(
        session_id: str, *, after: ForkPointCursor | None = None, limit: int = 50
    ) -> ForkPointPage:
        """记录分页参数并模拟 store 的唯一容量检查。"""
        calls.append((session_id, after, limit))
        if limit == 0:
            raise error
        return page

    def get_at_run(source_run_id: str) -> RunHistorySnapshot:
        """返回实际读取的历史结果供委托身份断言。"""
        calls.append(source_run_id)
        return preview

    def fork_session(command: ForkSession) -> SessionSnapshot:
        """记录完整 fork command 并返回 store 负责构造的快照。"""
        calls.append(command)
        return branch

    monkeypatch.setattr(store, "list_fork_points", list_points)
    monkeypatch.setattr(store, "load_session_at_run", get_at_run)
    monkeypatch.setattr(store, "fork_session", fork_session)
    history = SessionHistory(store)
    assert history.list_fork_points("exact session", after=cursor, limit=7) is page
    assert calls[-1] == ("exact session", cursor, 7)
    assert calls[-1][1] is cursor
    assert history.get_at_run("exact source") is preview
    assert calls[-1] == "exact source"
    before = datetime.now(UTC)
    assert history.fork("exact source") is branch
    command = calls[-1]
    assert isinstance(command, ForkSession)
    assert command.source_run_id == "exact source"
    assert command.target_session_id.startswith("session_")
    assert before <= command.now <= datetime.now(UTC)
    with pytest.raises(IrisRunStateError) as captured:
        history.list_fork_points("exact session", after=cursor, limit=0)
    assert captured.value is error
    assert calls[-1] == ("exact session", cursor, 0)
