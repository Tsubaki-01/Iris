"""原生 Responses 信息穿过真实持久化、HITL 与恢复边界。"""

import asyncio
from pathlib import Path

import pytest
from fakes import history_snapshot

from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunStopReason, SessionCompaction
from iris.message import LLMResponse, Msg
from iris.providers.responses import ResponsesMapper
from iris.runtime.compaction import project_history
from iris.store import SQLiteStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response


def _native_call() -> LLMResponse:
    return ResponsesMapper().parse_response(
        {
            "id": "response-native",
            "model": "native-model",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "opaque-reasoning",
                },
                {
                    "type": "message",
                    "id": "message-1",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "需要写入。"}],
                },
                {
                    "type": "function_call",
                    "id": "item-write",
                    "call_id": "call-write",
                    "name": "write",
                    "arguments": '{"value":"saved"}',
                },
            ],
        },
        provider="openai",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_responses_replay_survives_sqlite_resume_and_recovery(
    tmp_path: Path, recover: bool
) -> None:
    writes: list[str] = []

    def write(value: str) -> str:
        writes.append(value)
        return value

    registry = ToolRegistry()
    registry.register_function(write, description="写入", capabilities={ToolCapability.WRITE})
    database = tmp_path / "responses.db"
    waiting = await AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=StaticProvider(_native_call())),
        store=SQLiteStore(database),
    ).start(AgentRunRequest(input="写入记录", run_id="native-run"))
    assert waiting.pending_interaction is not None
    provider = StaticProvider(text_response("done"))
    reopened = SQLiteStore(database)
    if recover:
        blocked = BlockingProvider()
        resuming = asyncio.create_task(
            AgentRunner(
                runtime=build_runtime(tmp_path, registry=registry, provider=blocked),
                store=reopened,
            ).resume(
                "native-run",
                interaction_id=waiting.pending_interaction.interaction_id,
                response=PermissionInteractionResponse(decision="approve"),
            )
        )
        await asyncio.wait_for(blocked.started.wait(), timeout=2)
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resuming
        reopened = SQLiteStore(database)
        run = reopened.load_run("native-run")
        result = await AgentRunner(
            runtime=build_runtime(tmp_path, registry=registry, provider=provider),
            store=reopened,
        ).recover("native-run", expected_activation_id=run.current_activation_id)
    else:
        result = await AgentRunner(
            runtime=build_runtime(tmp_path, registry=registry, provider=provider),
            store=reopened,
        ).resume(
            "native-run",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert writes == ["saved"]
    wire = ResponsesMapper().format_messages(provider.requests[0].messages)
    native = [item for item in wire if item.get("id")]
    assert [item["id"] for item in native] == ["reasoning-1", "message-1", "item-write"]
    assert native[0]["encrypted_content"] == "opaque-reasoning"
    assert native[1]["phase"] == "commentary"
    assert native[2]["call_id"] == "call-write"
    output = next(item for item in wire if item["type"] == "function_call_output")
    assert output["call_id"] == "call-write"
    assert "saved" in str(output["output"])


def test_responses_compaction_does_not_replay_covered_native_items() -> None:
    native = _native_call().to_msg()
    history = [
        Msg.user("旧任务"),
        native,
        Msg.tool_result(tool_use_id="call-write", content="saved", name="write"),
        Msg.assistant("已完成"),
        Msg.user("新任务"),
    ]
    snapshot = history_snapshot(history, initial_count=4)
    projected = project_history(
        snapshot, SessionCompaction(summary="已完成旧任务", covered_message_count=4)
    )
    wire = ResponsesMapper().format_messages(projected)
    assert not any(item.get("id") == "reasoning-1" for item in wire)
    assert not any(item.get("call_id") == "call-write" for item in wire)
    assert native.metadata["responses"]["items"][0]["item"]["id"] == "reasoning-1"
