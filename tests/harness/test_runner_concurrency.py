"""AgentRunner session lane 与 activation 并发隔离测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisRunConflictError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import LLMRequest, LLMResponse
from iris.store import InMemoryLifecycleStore

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response


@pytest.mark.asyncio
async def test_same_session_concurrent_start_has_single_lane_owner(tmp_path: Path) -> None:
    """同 session 的第二个 run 必须在 provider effect 外被拒绝。"""
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
    )
    first = asyncio.create_task(
        runner.start(AgentRunRequest(input="first", session_id="shared", run_id="run-first"))
    )
    await provider.started.wait()

    with pytest.raises(IrisRunConflictError, match="session lane"):
        await runner.start(
            AgentRunRequest(input="second", session_id="shared", run_id="run-second")
        )

    assert len(provider.requests) == 1
    assert set(runner._active) == {"run-first"}
    provider.release.set()
    assert (await first).run.stop_reason is RunStopReason.COMPLETED


@pytest.mark.asyncio
async def test_different_sessions_can_own_live_activations_concurrently(
    tmp_path: Path,
) -> None:
    """process-local active map 不能把不同 session 全局串行化。"""

    class ConcurrentProvider:
        def __init__(self) -> None:
            self.requests: list[LLMRequest] = []
            self.both_started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if len(self.requests) == 2:
                self.both_started.set()
            await self.release.wait()
            return text_response()

    provider = ConcurrentProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
    )
    first = asyncio.create_task(
        runner.start(AgentRunRequest(input="one", session_id="session-1", run_id="run-1"))
    )
    second = asyncio.create_task(
        runner.start(AgentRunRequest(input="two", session_id="session-2", run_id="run-2"))
    )
    await asyncio.wait_for(provider.both_started.wait(), timeout=2)

    assert set(runner._active) == {"run-1", "run-2"}
    assert not first.done()
    assert not second.done()
    provider.release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.run.stop_reason is RunStopReason.COMPLETED
    assert second_result.run.stop_reason is RunStopReason.COMPLETED
    assert runner._active == {}


@pytest.mark.asyncio
async def test_terminal_finish_releases_lane_for_next_run(tmp_path: Path) -> None:
    """terminal fact 与 lane release 必须同事务完成。"""
    provider = StaticProvider(text_response("one"), text_response("two"))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
    )

    first = await runner.start(AgentRunRequest(input="one", session_id="shared", run_id="run-one"))
    second = await runner.start(AgentRunRequest(input="two", session_id="shared", run_id="run-two"))

    assert first.run.stop_reason is RunStopReason.COMPLETED
    assert second.run.stop_reason is RunStopReason.COMPLETED
    assert len(provider.requests) == 2
