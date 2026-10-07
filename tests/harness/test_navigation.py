"""两种生命周期后端的会话与运行分页发现。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.exceptions import IrisRunStateError
from iris.harness import AgentRunner, AgentRunRequest, SessionHistory
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import (
    BlockingProvider,
    FrozenClock,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)
from .test_runner_subagent import ChildProviders, _write_configs


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_session_and_run_pages_include_active_and_empty_forks(
    tmp_path: Path, backend: str
) -> None:
    """通用分页不借用 fork-only 过滤，时间相同也有稳定 ID 次序。"""
    store = SQLiteStore(tmp_path / "pages.db") if backend == "sqlite" else InMemoryLifecycleStore()
    clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=StaticProvider(*[text_response("完成")] * 3)),
        store=store,
        clock=clock,
    )
    for run_id, session_id in [("r1", "s1"), ("r2", "s1"), ("r3", "s2")]:
        await runner.start(AgentRunRequest(input=run_id, run_id=run_id, session_id=session_id))
    fork = SessionHistory(store).fork("r1")
    first = runner.list_sessions(limit=1)
    assert first.items[0].session_id == "s2"
    second = runner.list_sessions(after=first.next_cursor, limit=1)
    assert second.items[0].latest_run_id == "r2"
    last = runner.list_sessions(after=second.next_cursor, limit=1)
    assert last.items[0].session_id == fork.session_id
    assert last.items[0].latest_run_at is None
    assert last.next_cursor is None
    page = runner.list_runs("s1", limit=1)
    assert [run.run_id for run in page.items] == ["r1"]
    assert [run.run_id for run in runner.list_runs("s1", after=page.next_cursor).items] == ["r2"]
    assert runner.get_session_lane("s1") is None
    assert runner.read_session_messages("s1", start=1, limit=1).items[0][0] == 1
    provider = BlockingProvider()
    active_runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    active = asyncio.create_task(
        active_runner.start(AgentRunRequest(input="等待", run_id="r4", session_id="s1"))
    )
    await provider.started.wait()
    assert runner.get_session_lane("s1").run_id == "r4"
    assert runner.list_runs("s1").items[-1].run_id == "r4"
    assert runner.list_sessions().items[0].current_run_id == "r4"
    for read in (
        runner.list_sessions,
        lambda **kw: runner.list_runs("s1", **kw),
        lambda **kw: runner.list_child_runs("r4", **kw),
    ):
        with pytest.raises(IrisRunStateError):
            read(limit=0)
    provider.release.set()
    await active


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_child_pages_preserve_selector_and_do_not_mix_root_sessions(
    tmp_path: Path, backend: str
) -> None:
    """默认 selector 固定于 admission；child 独立会话不进入根侧栏。"""
    store = SQLiteStore(tmp_path / "child.db") if backend == "sqlite" else InMemoryLifecycleStore()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        store=store,
        provider=StaticProvider(
            *[
                tool_response(ToolUseBlock(id=f"c{i}", name="subagent", input={"prompt": "任务"}))
                for i in range(2)
            ],
            text_response("父完成"),
        ),
        child_provider_factory=ChildProviders(StaticProvider(text_response("子完成"))),
    )
    await runner.start(AgentRunRequest(input="委派", run_id="parent", session_id="root"))
    first = runner.list_child_runs("parent", limit=1)
    second = runner.list_child_runs("parent", after=first.next_cursor, limit=1)
    assert first.next_cursor is not None and second.next_cursor is None
    assert {first.items[0].parent_tool_call_id, second.items[0].parent_tool_call_id} == {"c0", "c1"}
    assert all(item.agent_selector == "researcher" for item in (*first.items, *second.items))
    assert first.items[0].run.session_id != second.items[0].run.session_id
    assert [session.session_id for session in runner.list_sessions().items] == ["root"]
    if backend == "sqlite":
        assert SQLiteStore(store.path).list_child_runs("parent", limit=1) == first
    other = InMemoryLifecycleStore()
    assert not other.list_runs("root").items
    assert not other.list_child_runs("parent").items
    await runner.aclose()
