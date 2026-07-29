from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    MutableCancellationSignal,
    build_runtime,
    resume_activation,
    start_activation,
)

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisRunConflictError, IrisRunPersistenceError
from iris.lifecycle import RuntimeExecutionOptions, ToolErrorPolicy
from iris.memory import (
    MemoryCategory,
    MemoryItem,
    MemoryItemKind,
    MemoryLevel,
    MemoryScope,
    MemorySearchResult,
)
from iris.message import LLMResponse, Msg, Role, TextBlock, ToolUseBlock
from iris.runtime import (
    AgentRuntime,
    ModelStepReservation,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeApprovedToolCall,
    RuntimeCursor,
    RuntimeToolCall,
    ToolCallClaim,
)
from iris.tools import (
    AskQuestionTool,
    DefaultPermissionPolicy,
    ReadFileState,
    ToolCapability,
    ToolErrorInfo,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    register_file_tools,
)


def _agent_config(*, writes: str = "allow") -> AgentConfig:
    return AgentConfig(
        name="phase-two-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
        permissions={"workspace": ".", "writes": writes},
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="遵守用户指令")]
        )
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id=f"response-{text}",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=4,
        output_tokens=2,
        total_tokens=6,
    )


def _tool_response(call: ToolUseBlock) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id=f"response-{call.id}",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要调用工具。"), call],
        finish_reason="tool_calls",
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
    )


def _runtime(
    *,
    provider: FakeProvider,
    tmp_path: Path,
    registry: ToolRegistry | None = None,
    writes: str = "allow",
) -> AgentRuntime:
    resolved_registry = registry or ToolRegistry()
    return build_runtime(
        agent_config=_agent_config(writes=writes),
        context_input=_context_input(),
        provider=provider,
        tool_registry=resolved_registry,
        tool_view=resolved_registry.view(),
        tool_executor=ToolExecutor(resolved_registry),
        workspace_root=tmp_path,
    )


def _approved_projection(waiting: RuntimeActivationResult) -> RuntimeApprovedToolCall:
    interaction = waiting.suspension
    assert interaction is not None
    call = interaction.request.tool_call
    return RuntimeApprovedToolCall(
        interaction_id=interaction.interaction_id,
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        fingerprint=call.fingerprint,
    )


def _interaction_result(
    waiting: RuntimeActivationResult,
    *,
    answer: str | None = None,
) -> ToolResult:
    interaction = waiting.suspension
    assert interaction is not None
    call = interaction.request.tool_call
    if answer is not None:
        return ToolResult(
            tool_use_id=call.tool_call_id,
            tool_name=call.tool_name,
            content=[TextBlock(text=answer)],
        )
    return ToolResult(
        tool_use_id=call.tool_call_id,
        tool_name=call.tool_name,
        is_error=True,
        error=ToolErrorInfo(code="USER_REJECTED", message="用户拒绝了工具调用"),
    )


@pytest.mark.asyncio
async def test_execute_start_commits_no_tool_completion(tmp_path: Path) -> None:
    provider = FakeProvider([_text_response("最终回答")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation(input="当前问题")
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert result.assistant_message is not None
    assert result.assistant_message.text == "最终回答"
    assert result.cursor.position == "outcome_ready"
    assert len(commits.model_commits) == 1
    assert [message.role for message in commits.model_commits[0].message_delta] == [
        Role.USER,
        Role.ASSISTANT,
    ]
    assert [message.role for message in provider.requests[0].messages] == [
        Role.SYSTEM,
        Role.USER,
    ]
    assert provider.requests[0].messages[-1].text == "当前问题"


@pytest.mark.asyncio
async def test_execute_runs_multi_step_tool_loop_through_required_commits(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def echo(value: str) -> str:
        effects.append(value)
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider(
        [
            _tool_response(ToolUseBlock(id="call-1", name="echo", input={"value": "Iris"})),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["Iris"]
    assert len(provider.requests) == 2
    assert len(commits.model_commits) == 2
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].claim is not None
    assert commits.events.index("claim_tool_call") < commits.events.index("commit_tool_result")
    assert provider.requests[1].messages[-1].tool_results[0].content == "echo:Iris"


@pytest.mark.asyncio
async def test_execute_injects_explicit_memory_only_on_first_model_step(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    def echo(value: str) -> str:
        return f"echo:{value}"

    registry.register_function(echo, description="回显")
    provider = FakeProvider(
        [
            _tool_response(ToolUseBlock(id="call-1", name="echo", input={"value": "Iris"})),
            _text_response("完成"),
        ]
    )
    memory = MemorySearchResult(
        item=MemoryItem(
            id="memory-1",
            scope=MemoryScope(workspace_id="workspace", agent_id="agent"),
            text="用户喜欢简洁回答",
            category=MemoryCategory.USER,
            kind=MemoryItemKind.PREFERENCE,
            level=MemoryLevel.SEMANTIC,
        ),
        score=0.98,
        matched_text="用户喜欢简洁回答",
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation(
        options=RuntimeExecutionOptions(memory_results=[memory.model_dump(mode="json")])
    )
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert any("用户喜欢简洁回答" in message.text for message in provider.requests[0].messages)
    assert all(message.sender != "context" for message in provider.requests[1].messages)


@pytest.mark.asyncio
async def test_execute_denied_reservation_skips_provider(tmp_path: Path) -> None:
    provider = FakeProvider([_text_response("不应调用")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, max_model_steps=0)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.BUDGET_EXHAUSTED
    assert provider.requests == []
    assert commits.model_commits == []


@pytest.mark.asyncio
async def test_execute_suspends_before_model_step_commit(tmp_path: Path) -> None:
    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    call = ToolUseBlock(id="write-1", name="write", input={"value": "x"})
    provider = FakeProvider([_tool_response(call)])
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.SUSPENDED
    assert result.suspension is not None
    assert result.cursor.position == "tool_batch"
    assert commits.model_commits == []
    assert len(commits.suspensions) == 1
    assert "suspend" in commits.events
    assert "claim_tool_call" not in commits.events


@pytest.mark.asyncio
async def test_execute_resume_approval_uses_normal_claim_result_path(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def write(value: str) -> str:
        effects.append(value)
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [
            _tool_response(ToolUseBlock(id="write-1", name="write", input={"value": "x"})),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    start = start_activation()
    commits = FakeRuntimeCommitPort(start)
    waiting = await runtime.execute(
        start,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )
    resumed = resume_activation(
        waiting.cursor,
        interaction_projection=_approved_projection(waiting),
    )
    commits.activation = resumed

    result = await runtime.execute(
        resumed,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["x"]
    assert len(commits.claims) == 1
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].claim is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["resume", "recover"])
async def test_execute_exact_approval_after_ordinary_prefix(
    tmp_path: Path,
    kind: str,
) -> None:
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    def write() -> str:
        effects.append("write")
        return "write"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[
                    ToolUseBlock(id="echo-1", name="echo", input={}),
                    ToolUseBlock(id="write-1", name="write", input={}),
                ],
                finish_reason="tool_calls",
            ),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    start = start_activation()
    commits = FakeRuntimeCommitPort(start)
    waiting = await runtime.execute(
        start,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )
    projection = _approved_projection(waiting)
    resumed = resume_activation(
        waiting.cursor,
        kind=kind,
        interaction_projection=projection,
    )
    commits.activation = resumed

    result = await runtime.execute(
        resumed,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["echo", "write"]
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == [
        "echo-1",
        "write-1",
    ]
    assert commits.tool_commits[1].tool_call.interaction_id == projection.interaction_id


@pytest.mark.asyncio
async def test_execute_rejection_after_ordinary_prefix_commits_without_effect(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    def write() -> str:
        effects.append("write")
        return "write"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[
                    ToolUseBlock(id="echo-1", name="echo", input={}),
                    ToolUseBlock(id="write-1", name="write", input={}),
                ],
                finish_reason="tool_calls",
            ),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    start = start_activation()
    commits = FakeRuntimeCommitPort(start)
    waiting = await runtime.execute(
        start,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )
    resumed = resume_activation(
        waiting.cursor,
        interaction_projection=_interaction_result(waiting),
    )
    commits.activation = resumed

    result = await runtime.execute(
        resumed,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["echo"]
    assert commits.tool_commits[-1].result.error is not None
    assert commits.tool_commits[-1].result.error.code == "USER_REJECTED"
    assert commits.tool_commits[-1].claim is None


@pytest.mark.asyncio
async def test_execute_question_after_ordinary_prefix_commits_answer(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    registry.register(AskQuestionTool())
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[
                    ToolUseBlock(id="echo-1", name="echo", input={}),
                    ToolUseBlock(
                        id="question-1",
                        name="ask_question",
                        input={"question": "继续？"},
                    ),
                ],
                finish_reason="tool_calls",
            ),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    start = start_activation()
    commits = FakeRuntimeCommitPort(start)
    waiting = await runtime.execute(
        start,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )
    resumed = resume_activation(
        waiting.cursor,
        interaction_projection=_interaction_result(waiting, answer="继续"),
    )
    commits.activation = resumed

    result = await runtime.execute(
        resumed,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["echo"]
    assert commits.tool_commits[-1].result.model_content == "继续"
    assert commits.tool_commits[-1].claim is None


@pytest.mark.asyncio
async def test_execute_rejects_mismatched_interaction_projection_before_effect(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    def write() -> str:
        effects.append("write")
        return "write"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[
                    ToolUseBlock(id="echo-1", name="echo", input={}),
                    ToolUseBlock(id="write-1", name="write", input={}),
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    start = start_activation()
    commits = FakeRuntimeCommitPort(start)
    waiting = await runtime.execute(
        start,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )
    projection = _approved_projection(waiting).model_copy(
        update={"tool_call_id": "other-call"}
    )
    resumed = resume_activation(
        waiting.cursor,
        interaction_projection=projection,
    )
    commits.activation = resumed

    with pytest.raises(IrisRunConflictError, match="interaction projection"):
        await runtime.execute(
            resumed,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )

    assert effects == []


@pytest.mark.asyncio
async def test_execute_cancellation_before_reservation_has_no_provider_effect(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([_text_response("不应调用")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(requested=True),
    )

    assert result.outcome is RuntimeActivationOutcome.CANCELLED
    assert provider.requests == []
    assert "reserve_model_step" not in commits.events


@pytest.mark.asyncio
async def test_execute_cancellation_after_reservation_skips_provider(
    tmp_path: Path,
) -> None:
    signal = MutableCancellationSignal()

    class ReservationCancellingPort(FakeRuntimeCommitPort):
        def reserve_model_step(self, cursor: RuntimeCursor) -> ModelStepReservation:
            reservation = super().reserve_model_step(cursor)
            signal.requested = True
            return reservation

    provider = FakeProvider([_text_response("不应调用")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = ReservationCancellingPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=signal,
    )

    assert result.outcome is RuntimeActivationOutcome.CANCELLED
    assert provider.requests == []
    assert "reserve_model_step" in commits.events


@pytest.mark.asyncio
async def test_execute_rejects_wrong_claim_version_before_tool_effect(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    class WrongVersionPort(FakeRuntimeCommitPort):
        def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
            claim = super().claim_tool_call(call)
            return claim.model_copy(update={"tool_version": claim.tool_version + 1})

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = WrongVersionPort(activation)

    with pytest.raises(IrisRunConflictError, match="claim identity"):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )

    assert effects == []


@pytest.mark.asyncio
async def test_execute_required_commit_failure_propagates(tmp_path: Path) -> None:
    provider = FakeProvider([_text_response("已生成但未提交")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, fail_at="commit_model_step")

    with pytest.raises(IrisRunPersistenceError, match="模拟 required commit 失败"):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )


def test_execute_source_does_not_call_old_persistence_writers() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            AgentRuntime.execute,
            AgentRuntime._execute_model_step,
            AgentRuntime._suspend_existing_batch,
        )
    )

    for forbidden in (
        "save_messages",
        "save_run_metadata",
        "append_tool_event",
        "interaction_service",
    ):
        assert forbidden not in source


@pytest.mark.asyncio
async def test_execute_injects_explicit_memory_snapshot(tmp_path: Path) -> None:
    memory = MemorySearchResult(
        item=MemoryItem(
            id="memory-1",
            scope=MemoryScope(workspace_id="workspace", agent_id="agent"),
            text="用户喜欢简洁回答",
            category=MemoryCategory.USER,
            kind=MemoryItemKind.PREFERENCE,
            level=MemoryLevel.SEMANTIC,
        ),
        score=0.98,
        matched_text="用户喜欢简洁回答",
    )
    provider = FakeProvider([_text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation(
        options=RuntimeExecutionOptions(
            memory_results=[memory.model_dump(mode="json")],
            memory_max_chars=100,
        )
    )
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    memory_message = provider.requests[0].messages[1]
    assert memory_message.sender == "context"
    assert "用户喜欢简洁回答" in memory_message.text
    assert provider.requests[0].messages[2].text == "当前问题"


@pytest.mark.asyncio
async def test_execute_restores_read_state_before_resumed_tool(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello old\n", encoding="utf-8")
    read_state = ReadFileState()
    read_state.update(path)
    assistant = Msg.assistant(
        [
            ToolUseBlock(
                id="edit-1",
                name="edit_file",
                input={
                    "file_path": "notes.txt",
                    "old_string": "old",
                    "new_string": "new",
                },
            )
        ]
    )
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        tool_calls=tuple(assistant.tool_calls),
        assistant_message=assistant,
        read_state=read_state.model_dump(mode="json"),
    )
    activation = resume_activation(cursor)
    registry = register_file_tools()
    provider = FakeProvider([_text_response("完成")])
    runtime = build_runtime(
        agent_config=_agent_config(writes="allow"),
        context_input=_context_input(),
        provider=provider,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(write_mode="allow"),
        ),
        workspace_root=tmp_path,
    )
    commits = FakeRuntimeCommitPort(activation, messages=[assistant])

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert path.read_text(encoding="utf-8") == "hello new\n"
    assert commits.tool_commits[0].result.is_error is False


@pytest.mark.asyncio
async def test_execute_preserves_full_tool_order_for_preflight_errors(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[
                    ToolUseBlock(id="missing-1", name="missing", input={}),
                    ToolUseBlock(id="echo-1", name="echo", input={}),
                ],
                finish_reason="tool_calls",
            ),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == [
        "missing-1",
        "echo-1",
    ]
    assert commits.tool_commits[0].claim is None
    assert commits.tool_commits[1].claim is not None
    assert effects == ["echo"]


@pytest.mark.asyncio
async def test_execute_stop_policy_fails_only_after_error_result_commit(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="missing-1", name="missing", input={}))]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation(
        options=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
    )
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "TOOL_NOT_ALLOWED"
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].claim is None


@pytest.mark.asyncio
async def test_execute_resume_cursor_continues_after_projected_question_or_reject(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(echo, description="回显")
    assistant = Msg.assistant(
        [
            ToolUseBlock(
                id="question-1",
                name="ask_question",
                input={"question": "继续？"},
            ),
            ToolUseBlock(id="echo-1", name="echo", input={}),
        ]
    )
    projected = ToolResult(
        tool_use_id="question-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )
    projected_message = Msg.tool_result(
        tool_use_id=projected.tool_use_id,
        content=projected.model_content,
        name=projected.tool_name,
    )
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        next_tool_index=1,
        tool_calls=tuple(assistant.tool_calls),
        tool_results=(projected,),
        assistant_message=assistant,
    )
    activation = resume_activation(cursor)
    provider = FakeProvider([_text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    commits = FakeRuntimeCommitPort(
        activation,
        messages=[assistant, projected_message],
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["echo"]
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == ["echo-1"]


@pytest.mark.asyncio
async def test_projected_question_does_not_approve_following_permission_gate(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def write() -> str:
        effects.append("write")
        return "write"

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    assistant = Msg.assistant(
        [
            ToolUseBlock(
                id="question-1",
                name="ask_question",
                input={"question": "继续？"},
            ),
            ToolUseBlock(id="write-1", name="write", input={}),
        ]
    )
    projected = ToolResult(
        tool_use_id="question-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        next_tool_index=1,
        tool_calls=tuple(assistant.tool_calls),
        tool_results=(projected,),
        assistant_message=assistant,
    )
    activation = resume_activation(cursor)
    runtime = _runtime(
        provider=FakeProvider([_text_response("不应调用")]),
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    commits = FakeRuntimeCommitPort(activation, messages=[assistant])

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.SUSPENDED
    assert result.suspension is not None
    assert result.suspension.request.tool_call.tool_call_id == "write-1"
    assert effects == []
    assert commits.claims == {}


@pytest.mark.asyncio
async def test_projected_reject_does_not_approve_following_permission_gate(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def write() -> str:
        effects.append("write")
        return "write"

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    assistant = Msg.assistant(
        [
            ToolUseBlock(id="write-1", name="write", input={}),
            ToolUseBlock(id="write-2", name="write", input={}),
        ]
    )
    rejected = ToolResult(
        tool_use_id="write-1",
        tool_name="write",
        is_error=True,
        error=ToolErrorInfo(code="USER_REJECTED", message="用户拒绝了工具调用"),
    )
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        next_tool_index=1,
        tool_calls=tuple(assistant.tool_calls),
        tool_results=(rejected,),
        assistant_message=assistant,
    )
    activation = resume_activation(cursor)
    runtime = _runtime(
        provider=FakeProvider([_text_response("不应调用")]),
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    commits = FakeRuntimeCommitPort(activation, messages=[assistant])

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.SUSPENDED
    assert result.suspension is not None
    assert result.suspension.request.tool_call.tool_call_id == "write-2"
    assert effects == []
    assert commits.claims == {}


@pytest.mark.asyncio
async def test_execute_same_batch_next_gate_suspends_then_continues(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def write(value: str) -> str:
        effects.append(value)
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[
                    ToolUseBlock(id="write-1", name="write", input={"value": "one"}),
                    ToolUseBlock(id="write-2", name="write", input={"value": "two"}),
                ],
                finish_reason="tool_calls",
            ),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    start = start_activation()
    commits = FakeRuntimeCommitPort(start)
    first_wait = await runtime.execute(
        start,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )
    first_resume = resume_activation(
        first_wait.cursor,
        activation_id="activation-2",
        interaction_projection=_approved_projection(first_wait),
    )
    commits.activation = first_resume

    second_wait = await runtime.execute(
        first_resume,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert second_wait.outcome is RuntimeActivationOutcome.SUSPENDED
    assert effects == ["one"]
    assert second_wait.cursor.next_tool_index == 1
    second_resume = resume_activation(
        second_wait.cursor,
        activation_id="activation-3",
        interaction_projection=_approved_projection(second_wait),
    )
    commits.activation = second_resume
    result = await runtime.execute(
        second_resume,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["one", "two"]
    assert len(commits.tool_commits) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "cancellation_requested",
        "remaining_deadline_seconds",
        "load_session",
        "reserve_model_step",
        "commit_model_step",
    ],
)
async def test_execute_model_required_port_failures_propagate(
    tmp_path: Path,
    failure: str,
) -> None:
    provider = FakeProvider([_text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, fail_at=failure)

    with pytest.raises(IrisRunPersistenceError):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["claim_tool_call", "commit_tool_result"])
@pytest.mark.parametrize("approved", [False, True], ids=["ordinary", "approved"])
async def test_execute_tool_required_commit_failures_propagate(
    tmp_path: Path,
    failure: str,
    approved: bool,
) -> None:
    effects: list[str] = []

    def effect() -> str:
        effects.append("effect")
        return "effect"

    registry = ToolRegistry()
    registry.register_function(
        effect,
        description="执行 effect",
        capabilities={ToolCapability.WRITE} if approved else set(),
    )
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="effect-1", name="effect", input={}))]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm" if approved else "allow",
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(
        activation,
        fail_at=None if approved else failure,
    )
    if approved:
        waiting = await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
        activation = resume_activation(
            waiting.cursor,
            interaction_projection=_approved_projection(waiting),
        )
        commits.activation = activation
        commits.fail_at = failure

    with pytest.raises(IrisRunPersistenceError):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )

    assert effects == ([] if failure == "claim_tool_call" else ["effect"])


@pytest.mark.asyncio
async def test_execute_suspend_required_commit_failure_propagates(tmp_path: Path) -> None:
    def write() -> str:
        return "write"

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="write-1", name="write", input={}))]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm",
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, fail_at="suspend")

    with pytest.raises(IrisRunPersistenceError):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True], ids=["ordinary", "approved"])
async def test_execute_claim_then_cancellation_is_outcome_unknown(
    tmp_path: Path,
    approved: bool,
) -> None:
    effects: list[str] = []
    signal = MutableCancellationSignal()

    def echo() -> str:
        effects.append("echo")
        return "echo"

    class ClaimCancellingPort(FakeRuntimeCommitPort):
        def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
            claim = super().claim_tool_call(call)
            signal.requested = True
            return claim

    registry = ToolRegistry()
    registry.register_function(
        echo,
        description="回显",
        capabilities={ToolCapability.WRITE} if approved else set(),
    )
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm" if approved else "allow",
    )
    activation = start_activation()
    commits = ClaimCancellingPort(activation)
    if approved:
        waiting = await runtime.execute(
            activation,
            commits=commits,
            cancellation=signal,
        )
        activation = resume_activation(
            waiting.cursor,
            interaction_projection=_approved_projection(waiting),
        )
        commits.activation = activation

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=signal,
    )

    assert result.outcome is RuntimeActivationOutcome.OUTCOME_UNKNOWN
    assert effects == []
    assert len(commits.claims) == 1
    assert commits.tool_commits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True], ids=["ordinary", "approved"])
async def test_execute_recover_does_not_replay_claimed_tool_without_result(
    tmp_path: Path,
    approved: bool,
) -> None:
    effects: list[str] = []
    signal = MutableCancellationSignal()

    def effect() -> str:
        effects.append("effect")
        return "effect"

    class ClaimCancellingPort(FakeRuntimeCommitPort):
        def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
            claim = super().claim_tool_call(call)
            signal.requested = True
            return claim

    registry = ToolRegistry()
    registry.register_function(
        effect,
        description="执行 effect",
        capabilities={ToolCapability.WRITE} if approved else set(),
    )
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="effect-1", name="effect", input={}))]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        writes="confirm" if approved else "allow",
    )
    activation = start_activation()
    commits = ClaimCancellingPort(activation)
    projection: RuntimeApprovedToolCall | None = None
    if approved:
        waiting = await runtime.execute(
            activation,
            commits=commits,
            cancellation=signal,
        )
        projection = _approved_projection(waiting)
        activation = resume_activation(
            waiting.cursor,
            interaction_projection=projection,
        )
        commits.activation = activation

    unknown = await runtime.execute(
        activation,
        commits=commits,
        cancellation=signal,
    )
    assert unknown.outcome is RuntimeActivationOutcome.OUTCOME_UNKNOWN
    signal.requested = False
    recovery = resume_activation(
        unknown.cursor,
        activation_id="activation-3",
        kind="recover",
        interaction_projection=projection,
    )
    commits.activation = recovery

    with pytest.raises(IrisRunConflictError):
        await runtime.execute(
            recovery,
            commits=commits,
            cancellation=signal,
        )

    assert effects == []
    assert len(commits.claims) == 1


@pytest.mark.asyncio
async def test_execute_commits_explicit_tool_result_before_latched_cancellation(
    tmp_path: Path,
) -> None:
    signal = MutableCancellationSignal()

    def echo() -> str:
        signal.requested = True
        return "echo"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=signal,
    )

    assert result.outcome is RuntimeActivationOutcome.CANCELLED
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].result.model_content == "echo"


@pytest.mark.asyncio
async def test_execute_expired_deadline_skips_provider_and_reservation(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([_text_response("不应调用")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, remaining_deadline_seconds=0)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.DEADLINE_EXCEEDED
    assert provider.requests == []
    assert "reserve_model_step" not in commits.events


@pytest.mark.asyncio
async def test_execute_provider_timeout_without_run_deadline_is_provider_failure(
    tmp_path: Path,
) -> None:
    class TimeoutProvider:
        async def complete(self, request: object) -> LLMResponse:
            del request
            raise TimeoutError("provider operation timeout")

    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=TimeoutProvider(),
        workspace_root=tmp_path,
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "PROVIDER_TIMEOUT"
    assert result.error.source == "provider"


@pytest.mark.asyncio
async def test_provider_timeout_with_remaining_deadline_is_not_run_deadline(
    tmp_path: Path,
) -> None:
    class TimeoutProvider:
        async def complete(self, request: object) -> LLMResponse:
            del request
            raise TimeoutError("provider operation timeout")

    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=TimeoutProvider(),
        workspace_root=tmp_path,
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, remaining_deadline_seconds=10)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "PROVIDER_TIMEOUT"


@pytest.mark.asyncio
async def test_provider_operation_is_capped_by_exhausted_run_deadline(
    tmp_path: Path,
) -> None:
    class SlowProvider:
        async def complete(self, request: object) -> LLMResponse:
            del request
            await asyncio.sleep(1)
            return _text_response("不应完成")

    class ExpiringPort(FakeRuntimeCommitPort):
        deadline_reads = 0

        def remaining_deadline_seconds(self) -> float:
            self._record("remaining_deadline_seconds")
            self.deadline_reads += 1
            return 0.01 if self.deadline_reads == 1 else 0.0

    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=SlowProvider(),
        workspace_root=tmp_path,
    )
    activation = start_activation()
    commits = ExpiringPort(activation, remaining_deadline_seconds=0.01)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.DEADLINE_EXCEEDED
