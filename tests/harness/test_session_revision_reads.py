"""Runner 的 admission 和 checkpoint 检查只读取所需的 session revision。"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.exceptions import IrisRunRecoveryError
from iris.harness import AgentRunner
from iris.harness._commit_port import StoreRuntimeCommitPort
from iris.harness._events import _RunEventCollector
from iris.lifecycle import AgentRunRequest, LifecycleStore, SessionSnapshot
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import build_runtime


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """在两种存储上验证相同调用契约。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "revision.db")
    return InMemoryLifecycleStore()


@pytest.mark.asyncio
async def test_start_facts_and_port_reuse_revision_without_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: LifecycleStore
) -> None:
    """已有历史的 start 构造只取 revision，port 初始化复用已读取 checkpoint。"""
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    await runner.start(AgentRunRequest(input="已有一轮历史"))
    revision = store.load_session("default").revision

    def reject_history(session_id: str) -> SessionSnapshot:
        pytest.fail("admission and port construction must not load full history")

    monkeypatch.setattr(store, "load_session", reject_history)
    command, cursor = runner._build_start_facts(AgentRunRequest(input="下一轮", run_id="next"))
    assert command.initial_checkpoint.session_revision == revision
    admitted = store.create_run(command)
    port = StoreRuntimeCommitPort(
        store=store,
        run=admitted.run,
        activation_id=command.start_activation_id,
        cursor=cursor,
        clock=runner._now,
        event_collector=_RunEventCollector(),
        workspace_root=tmp_path,
    )
    assert port._session_revision == revision


def test_recovery_revision_check_uses_current_store_without_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: LifecycleStore
) -> None:
    """独立读取当前 revision，不能用被验证 checkpoint 自证一致。"""
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    command, cursor = runner._build_start_facts(AgentRunRequest(input="恢复", run_id="recover"))
    admitted = store.create_run(command)
    assert admitted.checkpoint is not None

    def reject_history(session_id: str) -> SessionSnapshot:
        pytest.fail("recovery validation must not load full history")

    monkeypatch.setattr(store, "load_session", reject_history)
    assert runner._validate_recovery_checkpoint(admitted.run, admitted.checkpoint) == cursor
    stale = admitted.checkpoint.model_copy(update={"session_revision": 1})
    with pytest.raises(IrisRunRecoveryError, match="durable run/session"):
        runner._validate_recovery_checkpoint(admitted.run, stale)
