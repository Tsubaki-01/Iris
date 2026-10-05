"""Todo 查询使用实际 runtime 工作区，不接管会话或执行状态。"""

import asyncio
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisRunStateError, IrisTodoError
from iris.harness import AgentRunner, AgentRunOptions, AgentRunRequest, RunLimits
from iris.harness._context_access import ContextAccess
from iris.runtime import RuntimeFactory
from iris.store import InMemoryLifecycleStore

from .fakes import BlockingProvider, StaticProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("construction", ["runner", "runtime"])
async def test_get_todo_uses_resolved_environment_without_session(
    tmp_path: Path, construction: str
) -> None:
    """两条公开装配路径共享读取行为，查询本身不创建文件或 session。"""
    config = AgentConfig.model_validate(
        {
            "name": "todo-agent",
            "model": "openai/test",
            "system": "system",
            "todo": {"enabled": True},
            "permissions": {"workspace": "workspace"},
        }
    )
    store = InMemoryLifecycleStore()
    provider = StaticProvider()
    config_path = tmp_path / "config" / "agent.yaml"
    if construction == "runner":
        runner = AgentRunner.from_config(
            config, config_path=config_path, store=store, provider=provider
        )
    else:
        runtime = RuntimeFactory.from_config(
            config,
            config_path=config_path,
            provider=provider,
            context_access=ContextAccess(store),
        )
        runner = AgentRunner(runtime=runtime, store=store)
    try:
        snapshot = await runner.get_todo(" work ")
        assert snapshot.path == tmp_path / "config/workspace/.iris/todos/776f726b.md"
        assert snapshot.items == () and snapshot.error is None
        assert not snapshot.path.parent.exists()
        assert store.load_session_revision("work") == 0
        assert store.load_session_lane("work") is None
        assert not provider.requests

        snapshot.path.parent.mkdir(parents=True)
        snapshot.path.write_text("- [ ] 首次计划\n", encoding="utf-8")
        assert (await runner.get_todo("work")).items[0].content == "首次计划"
        snapshot.path.write_text("- [x] 人工修改\n", encoding="utf-8")
        updated = await runner.get_todo("work")
        assert updated.items[0].content == "人工修改"
        assert updated.items[0].status == "completed"
        assert store.load_session_revision("work") == 0
        assert not provider.requests
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_todo_query_does_not_change_active_run(tmp_path: Path) -> None:
    """查询活跃会话不等待执行结束、不改变 lane 或历史版本。"""
    provider = BlockingProvider()
    config = AgentConfig.model_validate(
        {
            "name": "todo-agent",
            "model": "openai/test",
            "system": "system",
            "todo": {"enabled": True},
            "permissions": {"workspace": str(tmp_path)},
        }
    )
    runner = AgentRunner.from_config(config, provider=provider)
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="执行", session_id="work"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=1)),
        )
    )
    try:
        await provider.started.wait()
        before = runner.get_session("work")
        lane = runner.store.load_session_lane("work")
        snapshot = await runner.get_todo("work")
        assert snapshot.error is None
        assert runner.get_session("work") == before
        assert runner.store.load_session_lane("work") == lane
        assert not task.done()
    finally:
        provider.release.set()
        await task
        await runner.aclose()


@pytest.mark.asyncio
async def test_todo_query_reports_disabled_and_empty_identity(tmp_path: Path) -> None:
    """关闭开关和空身份明确失败，不偷偷启用或创建 Todo 文件。"""
    config = AgentConfig.model_validate(
        {
            "name": "agent",
            "model": "openai/test",
            "system": "system",
            "permissions": {"workspace": str(tmp_path)},
        }
    )
    runner = AgentRunner.from_config(config, provider=StaticProvider())
    try:
        with pytest.raises(IrisRunStateError, match="session_id"):
            await runner.get_todo("  ")
        with pytest.raises(IrisTodoError, match="未启用"):
            await runner.get_todo("work")
        assert not (tmp_path / ".iris" / "todos").exists()
    finally:
        await runner.aclose()
