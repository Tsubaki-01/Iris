from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider

import iris.runtime.runtime as runtime_module
from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisSessionError
from iris.hitl import (
    InMemoryInteractionStore,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
)
from iris.message import LLMResponse, Role, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import (
    BoundedLoopOptions,
    RuntimeOptions,
    RuntimeStatus,
    RuntimeTurnResult,
    ToolErrorPolicy,
)
from iris.session import InMemorySessionStore
from iris.tools import (
    AskQuestionTool,
    DefaultPermissionPolicy,
    ToolCapability,
    ToolExecutor,
    ToolRegistry,
)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="hitl-resume-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
    )


def _tool_response(call: ToolUseBlock) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要工具。"), call],
        finish_reason="tool_calls",
    )


def _runtime_with_remaining_echo(
    executed: list[str],
) -> tuple[AgentRuntime, InMemoryInteractionStore]:
    def echo() -> str:
        executed.append("echo")
        return "echo"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(echo, description="回显")
    interaction_store = InMemoryInteractionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                LLMResponse(
                    provider="fake",
                    content=[
                        ToolUseBlock(
                            id="question",
                            name="ask_question",
                            input={"question": "继续？"},
                        ),
                        ToolUseBlock(id="echo", name="echo", input={}),
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ),
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    return runtime, interaction_store


class FailingRunMetadataStore(InMemorySessionStore):
    """按需模拟 resume metadata 写入失败。"""

    def __init__(self) -> None:
        super().__init__()
        self.fail_run_metadata = False

    def save_run_metadata(self, session_id: str, metadata: dict[str, object]) -> None:
        if self.fail_run_metadata:
            raise IrisSessionError("模拟 run metadata 写入失败", session_id=session_id)
        super().save_run_metadata(session_id, metadata)


_TOOL_RESULT_EVENT_KEYS = {
    "event_id",
    "type",
    "tool_call_id",
    "tool_name",
    "status",
    "error",
    "artifact",
    "run_id",
    "step_index",
    "agent_id",
    "metadata",
}


def _assert_tool_result_event_schema(event: dict[str, object]) -> None:
    assert set(event) == _TOOL_RESULT_EVENT_KEYS
    assert event["type"] == "tool_result"
    assert event["agent_id"] == "hitl-resume-agent"
    assert event["artifact"] is None


@pytest.mark.asyncio
async def test_resume_question_returns_answer_without_another_provider_call() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    session_store = InMemorySessionStore()
    interaction_store = InMemoryInteractionStore()
    provider = FakeProvider(
        [
            _tool_response(
                ToolUseBlock(
                    id="call_question",
                    name="ask_question",
                    input={"question": "选哪个？"},
                )
            )
        ]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=session_store,
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )

    waiting = await runtime.run_turn(
        "请询问我",
        options=RuntimeOptions(
            session_id="session-1",
            run_id="run-1",
            metadata={"caller": "cli"},
        ),
    )
    assert waiting.pending_interaction is not None
    waiting_metadata = session_store.load_run_metadata("session-1")

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="选生产"),
    )

    assert resumed.status is RuntimeStatus.OK
    assert [result.model_content for result in resumed.tool_results] == ["选生产"]
    assert len(provider.requests) == 1
    assert session_store.load_messages("session-1")[-1]["content"][0]["content"] == "选生产"
    question_event = session_store.load_tool_events("session-1")[0]
    _assert_tool_result_event_schema(question_event)
    assert question_event["event_id"] == "tool_result:run-1:call_question"
    metadata = session_store.load_run_metadata("session-1")
    latest_run = metadata["latest_run"]
    assert latest_run["status"] == "ok"
    assert latest_run["provider"] == "openai"
    assert latest_run["model"] == "gpt-4o-mini"
    assert latest_run["metadata"] == {"caller": "cli"}
    assert latest_run["steps"] == 1
    assert latest_run["tool_count"] == 1
    assert latest_run["message_count"] == 3
    assert "waiting_human" not in latest_run
    assert "interaction_id" not in latest_run
    assert len(metadata["runs"]) == len(waiting_metadata["runs"]) + 1


@pytest.mark.asyncio
async def test_resume_approved_permission_executes_only_matching_call_once() -> None:
    calls: list[str] = []

    def write_note(value: str) -> str:
        calls.append(value)
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="call_write", name="write_note", input={"value": "ok"}))]
    )
    session_store = InMemorySessionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=session_store,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        ),
        workspace_root=Path.cwd(),
    )

    waiting = await runtime.run_turn("写入", options=RuntimeOptions(run_id="run-2"))
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        PermissionInteractionResponse(decision="approve"),
    )

    assert resumed.status is RuntimeStatus.OK
    assert calls == ["ok"]
    assert len(provider.requests) == 1
    permission_event = session_store.load_tool_events("default")[0]
    _assert_tool_result_event_schema(permission_event)
    assert permission_event["event_id"] == "tool_result:run-2:call_write"


@pytest.mark.asyncio
async def test_resume_rejected_permission_does_not_execute_tool() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("called")
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note, description="写入笔记", capabilities={ToolCapability.WRITE}
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [_tool_response(ToolUseBlock(id="call_write", name="write_note", input={}))]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry, permission_policy=DefaultPermissionPolicy(write_mode="confirm")
        ),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("写入")
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id, PermissionInteractionResponse(decision="reject")
    )

    assert resumed.status is RuntimeStatus.OK
    assert resumed.tool_results[0].error is not None
    assert resumed.tool_results[0].error.code == "USER_REJECTED"
    assert calls == []


@pytest.mark.asyncio
async def test_resume_pauses_again_for_second_gate_in_same_batch() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    session_store = InMemorySessionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                LLMResponse(
                    provider="fake",
                    content=[
                        ToolUseBlock(id="first", name="ask_question", input={"question": "一？"}),
                        ToolUseBlock(id="second", name="ask_question", input={"question": "二？"}),
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ),
        session_store=session_store,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id, QuestionInteractionResponse(answer="一")
    )

    assert resumed.status is RuntimeStatus.WAITING_HUMAN
    assert resumed.pending_interaction is not None
    assert resumed.pending_interaction.request.tool_call.tool_call_id == "second"
    latest_run = session_store.load_run_metadata("default")["latest_run"]
    assert latest_run["status"] == "waiting_human"
    assert latest_run["waiting_human"] is True
    assert latest_run["interaction_id"] == resumed.pending_interaction.interaction_id


@pytest.mark.asyncio
async def test_resume_supports_question_then_permission_in_the_same_batch() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("write")
        return "written"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    policy = DefaultPermissionPolicy(write_mode="confirm")
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                LLMResponse(
                    provider="fake",
                    content=[
                        ToolUseBlock(
                            id="question",
                            name="ask_question",
                            input={"question": "一？"},
                        ),
                        ToolUseBlock(id="write", name="write_note", input={}),
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry, permission_policy=policy),
        workspace_root=Path.cwd(),
    )

    first = await runtime.run_turn("开始")
    assert first.pending_interaction is not None
    assert isinstance(first.pending_interaction.request.prompt, QuestionPrompt)

    second = await runtime.resume(
        first.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    assert second.pending_interaction is not None
    assert isinstance(second.pending_interaction.request.prompt, PermissionPrompt)
    assert second.pending_interaction.request.tool_call.tool_call_id == "write"

    completed = await runtime.resume(
        second.pending_interaction.interaction_id,
        PermissionInteractionResponse(decision="approve"),
    )

    assert completed.status is RuntimeStatus.OK
    assert calls == ["write"]


@pytest.mark.asyncio
async def test_resume_permission_fails_closed_when_policy_changes_to_deny() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("write")
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    policy = DefaultPermissionPolicy(write_mode="confirm")
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [_tool_response(ToolUseBlock(id="write", name="write_note", input={}))]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry, permission_policy=policy),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    policy.write_mode = "deny"

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        PermissionInteractionResponse(decision="approve"),
    )

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.code == "HITL_CHECKPOINT_INVALID"
    assert calls == []


@pytest.mark.asyncio
async def test_resume_rejects_tampered_tool_call_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(
                        id="question",
                        name="ask_question",
                        input={"question": "继续？"},
                    )
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    resolved = runtime.interaction_service.resolve(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    tool_call = resolved.request.tool_call.model_copy(update={"fingerprint": "b" * 64})
    tampered = resolved.model_copy(
        update={"request": resolved.request.model_copy(update={"tool_call": tool_call})}
    )
    monkeypatch.setattr(runtime.interaction_service, "get", lambda _: tampered)

    resumed = await runtime.resume(resolved.interaction_id)

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.code == "HITL_CHECKPOINT_INVALID"


@pytest.mark.asyncio
async def test_resume_rejects_v1_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(
                        id="question",
                        name="ask_question",
                        input={"question": "继续？"},
                    )
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    resolved = runtime.interaction_service.resolve(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    checkpoint = dict(resolved.checkpoint)
    checkpoint["checkpoint_version"] = 1
    legacy = resolved.model_copy(update={"checkpoint": checkpoint})
    monkeypatch.setattr(runtime.interaction_service, "get", lambda _: legacy)

    resumed = await runtime.resume(resolved.interaction_id)

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.code == "HITL_CHECKPOINT_INVALID"


@pytest.mark.asyncio
async def test_resume_claimed_interaction_without_result_fails_closed() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    session_store = InMemorySessionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="call", name="ask_question", input={"question": "继续？"})
                )
            ]
        ),
        session_store=session_store,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    resolved = runtime.interaction_service.resolve(
        interaction_id, QuestionInteractionResponse(answer="是")
    )
    assert resolved.status is InteractionStatus.RESOLVED
    assert runtime.load_resumable_interaction("default") == resolved
    claimed = runtime.interaction_service.claim(
        interaction_id, waiting.pending_interaction.checkpoint
    )
    assert claimed.resume_phase is InteractionResumePhase.CLAIMED
    assert runtime.load_resumable_interaction("default") == claimed

    resumed = await runtime.resume(interaction_id)

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.code == "HITL_EXECUTION_OUTCOME_UNKNOWN"
    latest_run = session_store.load_run_metadata("default")["latest_run"]
    assert latest_run["status"] == "error"
    assert latest_run["error"]["code"] == "HITL_EXECUTION_OUTCOME_UNKNOWN"
    assert "waiting_human" not in latest_run
    assert "interaction_id" not in latest_run


@pytest.mark.asyncio
async def test_resolved_interaction_can_be_loaded_and_resumed_without_response() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="call", name="ask_question", input={"question": "继续？"})
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    resolved = runtime.interaction_service.resolve(
        interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert runtime.load_resumable_interaction("default") == resolved
    resumed = await runtime.resume(interaction_id)

    assert resumed.status is RuntimeStatus.OK
    assert [result.model_content for result in resumed.tool_results] == ["继续"]


@pytest.mark.asyncio
async def test_result_ready_interaction_can_be_loaded_and_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    interaction_store = InMemoryInteractionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="call", name="ask_question", input={"question": "继续？"})
                )
            ]
        ),
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    commit_ready = runtime_module.commit_ready_interaction
    synchronize = runtime_module.synchronize_resume_metadata

    async def fail_before_result_commit(**_: object) -> RuntimeTurnResult:
        raise RuntimeError("模拟 result_ready 后崩溃")

    monkeypatch.setattr(runtime_module, "commit_ready_interaction", fail_before_result_commit)
    monkeypatch.setattr(
        runtime_module,
        "synchronize_resume_metadata",
        lambda *, session_store, result: result,
    )
    interrupted = await runtime.resume(
        interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    assert interrupted.status is RuntimeStatus.ERROR
    result_ready = interaction_store.load_interaction(interaction_id)
    assert result_ready is not None
    assert result_ready.resume_phase is InteractionResumePhase.RESULT_READY
    assert runtime.load_resumable_interaction("default") == result_ready

    monkeypatch.setattr(runtime_module, "commit_ready_interaction", commit_ready)
    monkeypatch.setattr(runtime_module, "synchronize_resume_metadata", synchronize)
    recovered = await runtime.resume(interaction_id)

    assert recovered.status is RuntimeStatus.OK
    ready_event = runtime.session_store.load_tool_events("default")[0]
    _assert_tool_result_event_schema(ready_event)
    assert str(ready_event["event_id"]).endswith(":call")


@pytest.mark.asyncio
async def test_result_committed_marker_yields_different_orphan_pending() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    interaction_store = InMemoryInteractionStore()
    session_store = InMemorySessionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                LLMResponse(
                    provider="fake",
                    content=[
                        ToolUseBlock(id="first", name="ask_question", input={"question": "一？"}),
                        ToolUseBlock(id="second", name="ask_question", input={"question": "二？"}),
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ),
        session_store=session_store,
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    first_waiting = await runtime.run_turn("开始")
    assert first_waiting.pending_interaction is not None
    first_id = first_waiting.pending_interaction.interaction_id
    second_waiting = await runtime.resume(
        first_id,
        QuestionInteractionResponse(answer="一"),
    )
    assert second_waiting.pending_interaction is not None
    first = interaction_store.load_interaction(first_id)
    assert first is not None
    assert first.resume_phase is InteractionResumePhase.RESULT_COMMITTED
    metadata = session_store.load_run_metadata("default")
    latest_run = dict(metadata["latest_run"])
    latest_run["interaction_id"] = first_id
    metadata["latest_run"] = latest_run
    session_store.save_run_metadata("default", metadata)

    assert runtime.load_resumable_interaction("default") == second_waiting.pending_interaction


@pytest.mark.asyncio
async def test_resume_result_committed_is_idempotent() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    store = InMemorySessionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="call", name="ask_question", input={"question": "继续？"})
                )
            ]
        ),
        session_store=store,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    await runtime.resume(interaction_id, QuestionInteractionResponse(answer="是"))

    retried = await runtime.resume(interaction_id)

    assert retried.status is RuntimeStatus.OK
    assert len(store.load_tool_events("default")) == 1


@pytest.mark.asyncio
async def test_resume_executes_calls_before_gate_in_original_order() -> None:
    executed: list[str] = []

    def read_probe() -> str:
        executed.append("read")
        return "read"

    registry = ToolRegistry()
    registry.register_function(
        read_probe,
        description="记录只读调用",
        capabilities={ToolCapability.READ},
    )
    registry.register(AskQuestionTool())
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                LLMResponse(
                    provider="fake",
                    content=[
                        ToolUseBlock(id="read", name="read_probe", input={}),
                        ToolUseBlock(
                            id="question",
                            name="ask_question",
                            input={"question": "继续？"},
                        ),
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert resumed.status is RuntimeStatus.OK
    assert executed == ["read"]
    assert [result.tool_use_id for result in resumed.tool_results] == ["read", "question"]


@pytest.mark.asyncio
async def test_resumed_loop_persists_tool_result_for_next_provider_request() -> None:
    def echo(value: str) -> str:
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(echo, description="回显")
    provider = FakeProvider(
        [
            _tool_response(
                ToolUseBlock(id="question", name="ask_question", input={"question": "继续？"})
            ),
            _tool_response(ToolUseBlock(id="echo", name="echo", input={"value": "Iris"})),
            LLMResponse(
                provider="fake",
                content=[TextBlock(text="完成")],
                finish_reason="stop",
            ),
        ]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_loop("开始")
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert resumed.status is RuntimeStatus.OK
    assert len(provider.requests) == 3
    tool_messages = [
        message for message in provider.requests[2].messages if message.role is Role.USER
    ]
    assert tool_messages[-1].tool_results[0].content == "echo:Iris"


@pytest.mark.asyncio
async def test_result_committed_resume_continues_remaining_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    def echo() -> str:
        executed.append("echo")
        return "echo"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(echo, description="回显")
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                LLMResponse(
                    provider="fake",
                    content=[
                        ToolUseBlock(
                            id="question",
                            name="ask_question",
                            input={"question": "继续？"},
                        ),
                        ToolUseBlock(id="echo", name="echo", input={}),
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    resume_batch = runtime._resume_batch

    async def simulate_crash(**_: object) -> object:
        raise RuntimeError("提交结果后崩溃")

    monkeypatch.setattr(runtime, "_resume_batch", simulate_crash)
    interrupted = await runtime.resume(
        interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    assert interrupted.status is RuntimeStatus.ERROR
    assert executed == []
    monkeypatch.setattr(runtime, "_resume_batch", resume_batch)

    recovered = await runtime.resume(interaction_id)

    assert recovered.status is RuntimeStatus.OK
    assert executed == ["echo"]
    message_count = len(runtime.session_store.load_messages("default"))

    retried = await runtime.resume(interaction_id)

    assert retried.status is RuntimeStatus.OK
    assert executed == ["echo"]
    assert len(runtime.session_store.load_messages("default")) == message_count


@pytest.mark.asyncio
async def test_resume_fails_closed_after_tool_before_result_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []
    runtime, interaction_store = _runtime_with_remaining_echo(executed)
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id

    def fail_after_tool(**_: object) -> object:
        raise RuntimeError("模拟工具执行后、结果写入前崩溃")

    append_result = runtime_module.append_resumed_result
    monkeypatch.setattr(runtime_module, "append_resumed_result", fail_after_tool)
    interrupted = await runtime.resume(
        interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert interrupted.status is RuntimeStatus.ERROR
    assert executed == ["echo"]
    stored = interaction_store.load_interaction(interaction_id)
    assert stored is not None
    assert stored.resume_phase is InteractionResumePhase.RESULT_COMMITTED
    assert stored.checkpoint["continuation_claim"] == {
        "kind": "tool",
        "next_tool_index": 1,
        "tool_call_id": "echo",
    }

    monkeypatch.setattr(runtime_module, "append_resumed_result", append_result)
    retried = await runtime.resume(interaction_id)

    assert retried.status is RuntimeStatus.ERROR
    assert retried.error is not None
    assert retried.error.code == "HITL_EXECUTION_OUTCOME_UNKNOWN"
    assert executed == ["echo"]


@pytest.mark.asyncio
async def test_resume_fails_closed_after_claim_before_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []
    runtime, interaction_store = _runtime_with_remaining_echo(executed)
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    execute_prepared = runtime.tool_executor.execute_prepared

    async def fail_before_tool(*_: object, **__: object) -> object:
        raise RuntimeError("模拟 claim 后、工具执行前崩溃")

    monkeypatch.setattr(runtime.tool_executor, "execute_prepared", fail_before_tool)
    interrupted = await runtime.resume(
        interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert interrupted.status is RuntimeStatus.ERROR
    assert executed == []
    stored = interaction_store.load_interaction(interaction_id)
    assert stored is not None
    assert stored.checkpoint["continuation_claim"]["tool_call_id"] == "echo"

    monkeypatch.setattr(runtime.tool_executor, "execute_prepared", execute_prepared)
    retried = await runtime.resume(interaction_id)

    assert retried.status is RuntimeStatus.ERROR
    assert retried.error is not None
    assert retried.error.code == "HITL_EXECUTION_OUTCOME_UNKNOWN"
    assert executed == []


@pytest.mark.asyncio
async def test_resume_fails_closed_after_result_append_before_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []
    runtime, interaction_store = _runtime_with_remaining_echo(executed)
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id

    def fail_before_cursor(**_: object) -> object:
        raise RuntimeError("模拟结果写入后、游标提交前崩溃")

    complete = runtime._complete_resumed_continuation
    monkeypatch.setattr(runtime, "_complete_resumed_continuation", fail_before_cursor)
    interrupted = await runtime.resume(
        interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert interrupted.status is RuntimeStatus.ERROR
    assert executed == ["echo"]
    stored = interaction_store.load_interaction(interaction_id)
    assert stored is not None
    assert stored.checkpoint["continuation_claim"]["tool_call_id"] == "echo"
    message_count = len(runtime.session_store.load_messages("default"))
    event_count = len(runtime.session_store.load_tool_events("default"))

    monkeypatch.setattr(runtime, "_complete_resumed_continuation", complete)
    retried = await runtime.resume(interaction_id)

    assert retried.status is RuntimeStatus.ERROR
    assert retried.error is not None
    assert retried.error.code == "HITL_EXECUTION_OUTCOME_UNKNOWN"
    assert executed == ["echo"]
    assert len(runtime.session_store.load_messages("default")) == message_count
    assert len(runtime.session_store.load_tool_events("default")) == event_count


@pytest.mark.asyncio
async def test_resumed_loop_claim_prevents_provider_and_tool_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    executed: list[str] = []

    def echo() -> str:
        executed.append("echo")
        return "echo"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(echo, description="回显")
    interaction_store = InMemoryInteractionStore()
    provider = FakeProvider(
        [
            _tool_response(
                ToolUseBlock(id="question", name="ask_question", input={"question": "继续？"})
            ),
            _tool_response(ToolUseBlock(id="echo", name="echo", input={})),
        ]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_loop("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    execute_once = runtime.tool_bridge.execute_once

    async def crash_after_loop_tool(**kwargs: object) -> object:
        await execute_once(**kwargs)
        raise SimulatedProcessCrash

    monkeypatch.setattr(runtime.tool_bridge, "execute_once", crash_after_loop_tool)
    with pytest.raises(SimulatedProcessCrash):
        await runtime.resume(
            interaction_id,
            QuestionInteractionResponse(answer="继续"),
        )

    stored = interaction_store.load_interaction(interaction_id)
    assert stored is not None
    assert stored.checkpoint["continuation_claim"] == {
        "kind": "loop",
        "next_tool_index": 1,
        "tool_call_id": None,
    }
    assert executed == ["echo"]
    assert len(provider.requests) == 2

    monkeypatch.setattr(runtime.tool_bridge, "execute_once", execute_once)
    retried = await runtime.resume(interaction_id)

    assert retried.status is RuntimeStatus.ERROR
    assert retried.error is not None
    assert retried.error.code == "HITL_EXECUTION_OUTCOME_UNKNOWN"
    assert executed == ["echo"]
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_resumed_loop_clears_claim_after_followup_gate() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    interaction_store = InMemoryInteractionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="first", name="ask_question", input={"question": "一？"})
                ),
                _tool_response(
                    ToolUseBlock(id="second", name="ask_question", input={"question": "二？"})
                ),
            ]
        ),
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_loop("开始")
    assert waiting.pending_interaction is not None
    first_id = waiting.pending_interaction.interaction_id

    followup = await runtime.resume(
        first_id,
        QuestionInteractionResponse(answer="一"),
    )

    assert followup.status is RuntimeStatus.WAITING_HUMAN
    assert followup.pending_interaction is not None
    assert followup.pending_interaction.request.tool_call.tool_call_id == "second"
    first = interaction_store.load_interaction(first_id)
    assert first is not None
    assert first.checkpoint["continuation_claim"] is None
    assert first.checkpoint["continuation_complete"] is True


@pytest.mark.asyncio
async def test_followup_gate_remains_discoverable_before_old_claim_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    interaction_store = InMemoryInteractionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="first", name="ask_question", input={"question": "一？"})
                ),
                _tool_response(
                    ToolUseBlock(id="second", name="ask_question", input={"question": "二？"})
                ),
            ]
        ),
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_loop("开始")
    assert waiting.pending_interaction is not None
    first_id = waiting.pending_interaction.interaction_id
    complete = runtime._complete_resumed_continuation

    def crash_before_claim_clear(**_: object) -> object:
        raise SimulatedProcessCrash

    monkeypatch.setattr(
        runtime,
        "_complete_resumed_continuation",
        crash_before_claim_clear,
    )
    with pytest.raises(SimulatedProcessCrash):
        await runtime.resume(
            first_id,
            QuestionInteractionResponse(answer="一"),
        )

    pending = runtime.interaction_service.list_pending("default")
    assert len(pending) == 1
    monkeypatch.setattr(runtime, "_complete_resumed_continuation", complete)
    assert runtime.load_resumable_interaction("default") == pending[0]


@pytest.mark.asyncio
async def test_resumed_loop_honors_stop_tool_error_policy() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    provider = FakeProvider(
        [
            _tool_response(
                ToolUseBlock(id="question", name="ask_question", input={"question": "继续？"})
            ),
            _tool_response(ToolUseBlock(id="missing", name="missing", input={})),
        ]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_loop(
        "开始",
        options=RuntimeOptions(
            loop=BoundedLoopOptions(tool_error_policy=ToolErrorPolicy.STOP),
        ),
    )
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.code == "TOOL_NOT_ALLOWED"
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_resume_rejects_question_checkpoint_from_another_workspace() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    interaction_store = InMemoryInteractionStore()
    first_workspace = Path.cwd()
    second_workspace = Path.cwd() / "src"
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(
                        id="question",
                        name="ask_question",
                        input={"question": "继续？"},
                    )
                )
            ]
        ),
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=first_workspace,
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    restarted = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider([]),
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=second_workspace,
    )

    resumed = await restarted.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.code == "HITL_CHECKPOINT_INVALID"


@pytest.mark.asyncio
async def test_resumed_loop_saves_terminal_max_steps_metadata() -> None:
    def read_probe() -> str:
        return "read"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(
        read_probe,
        description="读取探针",
        capabilities={ToolCapability.READ},
    )
    session_store = InMemorySessionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(
                        id="question",
                        name="ask_question",
                        input={"question": "继续？"},
                    )
                ),
                _tool_response(ToolUseBlock(id="read", name="read_probe", input={})),
            ]
        ),
        session_store=session_store,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_loop(
        "开始",
        options=RuntimeOptions(
            session_id="session-max",
            run_id="run-max",
            loop=BoundedLoopOptions(max_steps=2),
        ),
    )
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert resumed.status is RuntimeStatus.MAX_STEPS
    latest_run = session_store.load_run_metadata("session-max")["latest_run"]
    assert latest_run["status"] == "max_steps"
    assert latest_run["steps"] == 2
    assert latest_run["tool_count"] == 2
    assert latest_run["error"]["code"] == "MAX_STEPS_REACHED"
    assert "waiting_human" not in latest_run
    assert "interaction_id" not in latest_run


@pytest.mark.asyncio
async def test_resume_metadata_save_failure_returns_session_error() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    session_store = FailingRunMetadataStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(
                        id="question",
                        name="ask_question",
                        input={"question": "继续？"},
                    )
                )
            ]
        ),
        session_store=session_store,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn(
        "开始",
        options=RuntimeOptions(session_id="session-fail", run_id="run-fail"),
    )
    assert waiting.pending_interaction is not None
    session_store.fail_run_metadata = True

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.source == "session"
    assert resumed.error.code == "SESSION_ERROR"
