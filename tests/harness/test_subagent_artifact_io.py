"""WAITING child 结果落盘期间重新观察 parent 控制事实。"""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import ToolUseBlock
from iris.store import SQLiteStore
from iris.tools import ToolArtifact
from iris.tools.artifacts import ToolArtifactStore

from .fakes import FrozenClock, StaticProvider, text_response, tool_response
from .test_runner_subagent import ChildProviders, _parent_provider, _write_configs


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
async def test_waiting_child_artifact_rechecks_parent_after_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    """新 await 期间取消或到期后不以旧 revision finalize 已失效的 parent。"""
    clock = FrozenClock()
    provider = _parent_provider()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=provider,
        store=SQLiteStore(tmp_path / "lifecycle.db"),
        clock=clock,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                ),
                text_response("Long child answer " * 3000),
            )
        ),
    )
    waiting = await runner.start(
        AgentRunRequest(input="Start", run_id="parent"),
        options=AgentRunOptions(limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=10))),
    )
    entered = threading.Event()
    release = threading.Event()
    persist = ToolArtifactStore._persist_text

    def blocked_persist(self: ToolArtifactStore, *args: object, **kwargs: object) -> ToolArtifact:
        entered.set()
        assert release.wait(3)
        return persist(self, *args, **kwargs)

    monkeypatch.setattr(ToolArtifactStore, "_persist_text", blocked_persist)
    continuation = asyncio.create_task(
        runner.resume(
            "parent",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="Continue"),
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert runner.get_run("parent").phase is RunPhase.WAITING
        if interruption == "cancel":
            runner.request_cancel("parent")
        else:
            clock.advance(seconds=11)
    finally:
        release.set()
    result = await continuation
    assert result.run.stop_reason is (
        RunStopReason.CANCELLED if interruption == "cancel" else RunStopReason.DEADLINE_EXCEEDED
    )
    assert len(provider.requests) == 1
    assert runner.store.load_result("parent") == result
    await runner.aclose()
