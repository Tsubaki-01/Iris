"""AgentRunner durable resume 集成测试。"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisRunRecoveryError
from iris.harness import AgentRunner
from iris.hitl import (
    PermissionInteractionResponse,
    QuestionInteractionResponse,
)
from iris.lifecycle import AgentRunRequest, RunEvent, RunEventKind, RunPhase, RunStopReason
from iris.message import LLMResponse, TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import AskQuestionTool, ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response


@pytest.mark.asyncio
async def test_managed_resume_signals_after_begin_and_relays_live_events(
    tmp_path: Path,
) -> None:
    """Managed resume 在 resolve/begin/register 后早于 resumed provider 返回。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    store = InMemoryLifecycleStore()
    waiting = await AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write-1", name="write", input={"value": "x"}))
            ),
        ),
        store=store,
    ).start(AgentRunRequest(input="写入", run_id="run-managed-resume"))
    assert waiting.pending_interaction is not None
    before_sequence = waiting.run.last_event_sequence
    provider = BlockingProvider(text_response("完成"))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=store,
    )
    activation_started = asyncio.Event()
    relayed: list[RunEvent] = []
    running = asyncio.create_task(
        runner._resume_managed(
            "run-managed-resume",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
            durable_event_callback=relayed.append,
            activation_started=activation_started,
        )
    )

    try:
        await asyncio.wait_for(activation_started.wait(), timeout=1)
        active = store.load_run("run-managed-resume")
        assert active is not None and active.phase is RunPhase.ACTIVE
        assert "run-managed-resume" in runner._active
        assert not running.done()
        await asyncio.wait_for(provider.started.wait(), timeout=1)
    finally:
        provider.release.set()
    result = await running

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert relayed == store.list_events("run-managed-resume", before_sequence)
    assert RunEventKind.INTERACTION_RESOLVED in {event.kind for event in relayed}
    assert RunEventKind.ACTIVATION_STARTED in {event.kind for event in relayed}


@pytest.mark.asyncio
async def test_resume_exposes_same_batch_question_gates_in_original_order(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    batch = LLMResponse(
        provider="fake",
        id="response-batch",
        model="fake-model",
        content=[
            TextBlock(text="需要回答两个问题。"),
            ToolUseBlock(id="first", name="ask_question", input={"question": "一？"}),
            ToolUseBlock(id="second", name="ask_question", input={"question": "二？"}),
        ],
        finish_reason="tool_calls",
    )
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=StaticProvider(batch)),
        store=store,
    )
    first = await runner.start(AgentRunRequest(input="提问", run_id="run-questions"))
    assert first.pending_interaction is not None
    assert first.pending_interaction.tool_call_id == "first"

    second = await AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=StaticProvider()),
        store=SQLiteStore(tmp_path / "lifecycle.db"),
    ).resume(
        "run-questions",
        interaction_id=first.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="答案一"),
    )

    assert second.run.phase is RunPhase.WAITING
    assert second.pending_interaction is not None
    assert second.pending_interaction.tool_call_id == "second"
    final = await AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(text_response("收到")),
        ),
        store=SQLiteStore(tmp_path / "lifecycle.db"),
    ).resume(
        "run-questions",
        interaction_id=second.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="答案二"),
    )
    assert final.run.stop_reason is RunStopReason.COMPLETED


@pytest.mark.asyncio
async def test_resume_fails_closed_when_environment_fingerprint_changes(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    path = tmp_path / "lifecycle.db"
    waiting = await AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(id="question", name="ask_question", input={"question": "继续？"})
                )
            ),
        ),
        store=SQLiteStore(path),
    ).start(AgentRunRequest(input="提问", run_id="run-drift"))
    assert waiting.pending_interaction is not None

    changed = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            system_text="改变后的系统提示",
            provider=StaticProvider(text_response()),
        ),
        store=SQLiteStore(path),
    )
    with pytest.raises(IrisRunRecoveryError):
        await changed.resume(
            "run-drift",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("environment_fingerprint", "drifted-checkpoint"),
        ("cursor_json", "{}"),
        ("session_revision", "99"),
        ("model_steps_reserved", "2"),
        ("sequence", "99"),
    ],
)
async def test_resume_rejects_corrupt_checkpoint_before_consuming_response(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    path = tmp_path / "lifecycle.db"
    store = SQLiteStore(path)
    waiting = await AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(id="question", name="ask_question", input={"question": "继续？"})
                )
            ),
        ),
        store=store,
    ).start(AgentRunRequest(input="提问", run_id="run-checkpoint-drift"))
    assert waiting.pending_interaction is not None
    before_run = store.load_run("run-checkpoint-drift")
    before_interaction = store.load_interaction(waiting.pending_interaction.interaction_id)

    statement = {
        "environment_fingerprint": (
            "UPDATE run_checkpoints SET environment_fingerprint = ? WHERE run_id = ?"
        ),
        "cursor_json": "UPDATE run_checkpoints SET cursor_json = ? WHERE run_id = ?",
        "session_revision": ("UPDATE run_checkpoints SET session_revision = ? WHERE run_id = ?"),
        "model_steps_reserved": (
            "UPDATE run_checkpoints SET model_steps_reserved = ? WHERE run_id = ?"
        ),
        "sequence": "UPDATE run_checkpoints SET sequence = ? WHERE run_id = ?",
    }[column]
    with sqlite3.connect(path) as connection:
        connection.execute(statement, (value, "run-checkpoint-drift"))

    provider = StaticProvider(text_response())
    resumed_store = SQLiteStore(path)
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=resumed_store,
    )
    with pytest.raises(IrisRunRecoveryError):
        await runner.resume(
            "run-checkpoint-drift",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )

    assert resumed_store.load_run("run-checkpoint-drift") == before_run
    assert resumed_store.load_interaction(waiting.pending_interaction.interaction_id) == (
        before_interaction
    )
    assert provider.requests == []
