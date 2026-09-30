"""Goal 单轮执行复用真正 runtime、提交与共享准备的契约。"""

import asyncio
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisConfigError, IrisRunStateError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RuntimeExecutionOptions
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore

from .fakes import StaticProvider, build_runtime, text_response, tool_response


def goal_config(tmp_path: Path) -> AgentConfig:
    """使用真实配置边界开启 Goal。"""
    return AgentConfig.model_validate(
        {
            "name": "goal-test",
            "model": "deepseek/deepseek-chat",
            "system": "完成目标",
            "goal": {"enabled": True},
            "permissions": {"workspace": str(tmp_path)},
        }
    )


@pytest.mark.asyncio
async def test_goal_run_commits_source_and_report_without_automatic_next_run(
    tmp_path: Path,
) -> None:
    store = InMemoryLifecycleStore()

    class ReportProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                goal = store.get_current_goal("s")
                return tool_response(
                    ToolUseBlock(
                        id="report",
                        name="report_goal",
                        input={
                            "goal_id": goal.goal_id,
                            "revision": goal.revision,
                            "decision": "complete",
                            "reason": "已验证结果",
                        },
                    )
                )
            return text_response()

    provider = ReportProvider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    service = runner._goal_service
    goal = service.create("s", "给出结果")
    started = asyncio.Event()
    result = await runner._start_goal_managed(
        goal.ref, run_id="goal-run", activation_started=started
    )
    assert started.is_set()
    assert result.run.run_id == "goal-run"
    assert store.get_goal(goal.goal_id).rounds_started == 1
    assert store.get_goal(goal.goal_id).status == "active"
    messages = store.load_session("s").messages
    assert messages[0].sender == "context"
    assert messages[0].metadata["context_kind"] == "goal_continuation"
    assert store._sessions["s"].read_state.last_ordinary_user_index is None
    assert any("给出结果" in message.text for message in provider.requests[0].messages)
    settled = service.settle_run("goal-run", now=runner._now())
    assert settled.goal.status == "completed"
    await runner.aclose()


@pytest.mark.asyncio
async def test_goal_enabled_ordinary_start_stays_unbound(tmp_path: Path) -> None:
    runner = AgentRunner.from_config(
        goal_config(tmp_path), provider=StaticProvider(text_response())
    )
    goal = runner._goal_service.create("s", "目标与普通聊天独立")
    result = await runner.start(AgentRunRequest(input="普通问题", session_id="s"))
    assert runner._goal_service.store.get_goal_run(result.run.run_id) is None
    assert runner._goal_service.get(goal.goal_id).rounds_started == 0
    assert runner.store.load_session("s").messages[0].sender != "context"
    await runner.aclose()


@pytest.mark.parametrize(
    "options",
    [
        AgentRunOptions(runtime=RuntimeExecutionOptions(include_tools=False)),
        AgentRunOptions(
            runtime=RuntimeExecutionOptions(request_options={"tool_choice": "required"})
        ),
    ],
)
def test_goal_create_rejects_options_that_prevent_reporting(
    tmp_path: Path, options: AgentRunOptions
) -> None:
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=StaticProvider())
    with pytest.raises(IrisConfigError):
        runner._goal_service.create("s", "目标", run_options=options)
    assert runner._goal_service.get_current("s") is None


@pytest.mark.asyncio
async def test_shared_prepare_survives_cancelled_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    runner._prepared = False
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def prepare(environment: object) -> None:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()

    monkeypatch.setattr(type(runner.runtime.environment), "aprepare", prepare)
    first = asyncio.create_task(runner.aprepare())
    await entered.wait()
    second = asyncio.create_task(runner.aprepare())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not runner._closed
    release.set()
    await second
    assert calls == 1
    assert runner._prepared
    await runner.aclose()


@pytest.mark.asyncio
async def test_close_waits_for_shared_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    runner._prepared = False
    entered, release = asyncio.Event(), asyncio.Event()
    closed = False

    async def prepare(environment: object) -> None:
        entered.set()
        await release.wait()

    async def close(environment: object) -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(type(runner.runtime.environment), "aprepare", prepare)
    monkeypatch.setattr(type(runner.runtime.environment), "aclose", close)
    preparing = asyncio.create_task(runner.aprepare())
    await entered.wait()
    closing = asyncio.create_task(runner.aclose())
    await asyncio.sleep(0)
    assert not closed
    release.set()
    with pytest.raises(IrisRunStateError):
        await preparing
    await closing
    assert closed
