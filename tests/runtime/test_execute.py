from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    FakeRuntimeSteeringPort,
    MutableCancellationSignal,
    build_runtime,
    resume_activation,
    start_activation,
)
from pydantic import BaseModel, ValidationError

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisRunConflictError,
    IrisRunPersistenceError,
)
from iris.lifecycle import CheckpointResumability, RuntimeExecutionOptions, ToolErrorPolicy
from iris.memory import (
    MemoryCategory,
    MemoryIOExecutionMode,
    MemoryItem,
    MemoryItemKind,
    MemoryLevel,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.message import LLMResponse, Msg, Role, TextBlock, ToolUseBlock
from iris.runtime import (
    AgentRuntime,
    ModelStepReservation,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeApprovedToolCall,
    RuntimeCursor,
    RuntimeModelStepCommit,
    RuntimeToolCall,
    RuntimeToolResultCommit,
    SteeringInput,
    ToolCallClaim,
)
from iris.tools import (
    AskQuestionTool,
    BaseTool,
    CallableExecutionMode,
    CallableTool,
    DefaultPermissionPolicy,
    PermissionPolicy,
    ReadFileState,
    ToolCapability,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
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
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
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


def _tool_batch_response(calls: list[ToolUseBlock]) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-tool-batch",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要调用工具。"), *calls],
        finish_reason="tool_calls",
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
    )


class _CountingSerialTool(BaseTool):
    """记录 runtime tool batch 对每条输入的校验次数。"""

    definition = ToolDefinition(
        name="counting_serial",
        description="记录串行工具校验次数",
        input_schema={"type": "object"},
        capabilities={ToolCapability.WRITE},
    )

    def __init__(self) -> None:
        self.validation_calls: dict[str, int] = {}

    def validate_input(self, params: dict[str, Any]) -> dict[str, Any]:
        value = str(params["value"])
        self.validation_calls[value] = self.validation_calls.get(value, 0) + 1
        return {"value": value}

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        assert isinstance(params, dict)
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=context.tool_name,
            content=[TextBlock(text=str(params["value"]))],
        )


def test_steering_input_requires_user_message() -> None:
    """Steering input 只能承载待提交的 user message。"""
    with pytest.raises(ValidationError, match="user message"):
        SteeringInput(
            submission_id="submission-1",
            message=Msg.assistant("不能作为 steer 输入"),
        )
    with pytest.raises(ValidationError, match="submission_id"):
        SteeringInput(submission_id=" ", message=Msg.user("新的指令"))

    input_value = SteeringInput(
        submission_id="submission-1",
        message=Msg.user("新的指令"),
    )
    with pytest.raises(ValidationError, match="frozen"):
        input_value.submission_id = "submission-2"


def _runtime(
    *,
    provider: FakeProvider,
    tmp_path: Path,
    registry: ToolRegistry | None = None,
    middleware: list[ToolMiddleware] | None = None,
    permission_policy: PermissionPolicy | None = None,
    memory_service: MemoryService | None = None,
    writes: str = "allow",
) -> AgentRuntime:
    resolved_registry = registry or ToolRegistry()
    return build_runtime(
        agent_config=_agent_config(writes=writes),
        context_input=_context_input(),
        provider=provider,
        tool_registry=resolved_registry,
        tool_view=resolved_registry.view(),
        tool_executor=ToolExecutor(
            resolved_registry,
            permission_policy=permission_policy,
            middleware=middleware,
        ),
        workspace_root=tmp_path,
        memory_service=memory_service,
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
async def test_execute_no_tool_steer_commits_input_before_next_model_step(
    tmp_path: Path,
) -> None:
    """漏掉原子 steer delta 会让下一次 provider 看不到新输入。"""
    provider = FakeProvider([_text_response("先回答"), _text_response("最终回答")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation(input="当前问题")
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort(
        [
            SteeringInput(
                submission_id="submission-1",
                message=Msg.user(
                    "调整方向",
                    metadata={"submission_id": "submission-1", "mode": "steer"},
                ),
            )
        ]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 2
    assert [message.role for message in commits.model_commits[0].message_delta] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert commits.model_commits[0].cursor_after == RuntimeCursor(
        position="before_model",
        step_index=1,
    )
    assert commits.model_commits[0].resumability is CheckpointResumability.SAFE
    assert provider.requests[1].messages[-1].text == "调整方向"
    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("acknowledge", "submission-1", None),
        ("claim", activation.run_id, activation.activation_id),
    ]


@pytest.mark.asyncio
async def test_execute_empty_steering_port_keeps_no_tool_completion(
    tmp_path: Path,
) -> None:
    """空 steering port 只观察边界，不改变既有完成语义。"""
    provider = FakeProvider([_text_response("最终回答")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation(input="当前问题")
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort()

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert result.cursor.position == "outcome_ready"
    assert [message.role for message in commits.model_commits[0].message_delta] == [
        Role.USER,
        Role.ASSISTANT,
    ]
    assert commits.model_commits[0].resumability is CheckpointResumability.OUTCOME_READY
    assert steering.events == [("claim", activation.run_id, activation.activation_id)]


@pytest.mark.asyncio
async def test_execute_consumes_one_steer_per_no_tool_boundary(
    tmp_path: Path,
) -> None:
    """连续安全边界按 FIFO 每次只提交一条 steer。"""
    provider = FakeProvider(
        [_text_response("第一轮"), _text_response("第二轮"), _text_response("最终轮")]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    messages = [Msg.user("方向一"), Msg.user("方向二")]
    steering = FakeRuntimeSteeringPort(
        [
            SteeringInput(submission_id=f"submission-{index}", message=message)
            for index, message in enumerate(messages, start=1)
        ]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(commits.model_commits) == 3
    assert commits.model_commits[0].message_delta[-1] == messages[0]
    assert commits.model_commits[1].message_delta[-1] == messages[1]
    assert [event[0] for event in steering.events] == [
        "claim",
        "acknowledge",
        "claim",
        "acknowledge",
        "claim",
    ]


@pytest.mark.asyncio
async def test_execute_has_no_await_between_claim_return_commit_and_ack(
    tmp_path: Path,
) -> None:
    """Claim 返回后，commit 与 ack 必须在 event-loop 下一次调度前完成。"""
    order: list[str] = []

    class RecordingSteeringPort(FakeRuntimeSteeringPort):
        async def claim(
            self,
            run_id: str,
            activation_id: str,
        ) -> SteeringInput | None:
            claimed = await super().claim(run_id, activation_id)
            if claimed is not None:
                order.append("claim-return")
                asyncio.get_running_loop().call_soon(order.append, "interleaved")
            return claimed

        def acknowledge(self, submission_id: str) -> None:
            order.append("acknowledge")
            super().acknowledge(submission_id)

    class RecordingCommitPort(FakeRuntimeCommitPort):
        def commit_model_step(self, commit: RuntimeModelStepCommit) -> RuntimeCursor:
            order.append("commit")
            return super().commit_model_step(commit)

    provider = FakeProvider([_text_response("第一轮"), _text_response("最终轮")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = RecordingCommitPort(activation)
    steering = RecordingSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )
    await asyncio.sleep(0)

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert order[:3] == ["claim-return", "commit", "acknowledge"]
    assert order.index("interleaved") > order.index("acknowledge")


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_fails", [False, True])
async def test_execute_final_tool_settlement_has_no_await_after_claim(
    tmp_path: Path,
    commit_fails: bool,
) -> None:
    """Final-tool claim 后的 commit 与 ack/fail 不允许 task interleave。"""
    order: list[str] = []

    class RecordingSteeringPort(FakeRuntimeSteeringPort):
        async def claim(
            self,
            run_id: str,
            activation_id: str,
        ) -> SteeringInput | None:
            claimed = await super().claim(run_id, activation_id)
            if claimed is not None:
                order.append("claim-return")
                asyncio.get_running_loop().call_soon(order.append, "interleaved")
            return claimed

        def acknowledge(self, submission_id: str) -> None:
            order.append("acknowledge")
            super().acknowledge(submission_id)

        def fail(self, submission_id: str, reason: str) -> None:
            order.append("fail")
            super().fail(submission_id, reason)

    class RecordingCommitPort(FakeRuntimeCommitPort):
        def commit_tool_result(self, commit: RuntimeToolResultCommit) -> RuntimeCursor:
            order.append("commit")
            return super().commit_tool_result(commit)

    def echo() -> str:
        return "echo"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider(
        [
            _tool_response(ToolUseBlock(id="echo-1", name="echo", input={})),
            _text_response("完成"),
        ]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = RecordingCommitPort(
        activation,
        fail_at="commit_tool_result" if commit_fails else None,
    )
    steering = RecordingSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    if commit_fails:
        with pytest.raises(IrisRunPersistenceError, match="required commit"):
            await runtime.execute(
                activation,
                commits=commits,
                cancellation=MutableCancellationSignal(),
                steering=steering,
            )
        settlement = "fail"
    else:
        result = await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            steering=steering,
        )
        assert result.outcome is RuntimeActivationOutcome.COMPLETED
        settlement = "acknowledge"
    await asyncio.sleep(0)

    assert order[:3] == ["claim-return", "commit", settlement]
    assert order.index("interleaved") > order.index(settlement)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancellation_requested", "deadline", "expected_outcome"),
    [
        (True, None, RuntimeActivationOutcome.CANCELLED),
        (False, 0.0, RuntimeActivationOutcome.DEADLINE_EXCEEDED),
    ],
)
async def test_execute_cancel_or_deadline_never_claims_steer(
    tmp_path: Path,
    cancellation_requested: bool,
    deadline: float | None,
    expected_outcome: RuntimeActivationOutcome,
) -> None:
    """Activation 已取消或过期时不进入 steering safe boundary。"""
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    class BoundaryStateProvider(FakeProvider):
        async def complete(self, request: Any) -> Any:
            response = await super().complete(request)
            commits.cancel_requested = cancellation_requested
            commits.deadline = deadline
            return response

    provider = BoundaryStateProvider([_text_response("边界回答")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is expected_outcome
    assert len(provider.requests) == 1
    assert commits.model_commits == []
    assert steering.events == []


@pytest.mark.asyncio
async def test_execute_outcome_ready_cursor_never_claims_steer(
    tmp_path: Path,
) -> None:
    """已经 durable terminal 的 cursor 直接兑现结果，不再观察 steering。"""
    assistant = Msg.assistant("已完成")
    activation = start_activation().model_copy(
        update={
            "cursor": RuntimeCursor(
                position="outcome_ready",
                step_index=0,
                assistant_message=assistant,
            )
        }
    )
    provider = FakeProvider([])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    commits = FakeRuntimeCommitPort(activation, messages=[assistant])
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert result.assistant_message == assistant
    assert provider.requests == []
    assert steering.events == []


@pytest.mark.asyncio
async def test_execute_steers_only_after_final_ordered_tool_result(
    tmp_path: Path,
) -> None:
    """同一批次仅在最终有序工具结果提交时领取 steer。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    calls = [
        ToolUseBlock(id=f"write-{index}", name="write", input={"value": str(index)})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        permission_policy=DefaultPermissionPolicy(write_mode="allow"),
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    steered_message = Msg.user("调整工具结果后的方向")
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=steered_message)]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(commits.tool_commits) == 2
    assert len(commits.tool_commits[0].message_delta) == 1
    assert commits.tool_commits[1].message_delta[1] == steered_message
    assert provider.requests[1].messages[-1] == steered_message
    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("acknowledge", "submission-1", None),
        ("claim", activation.run_id, activation.activation_id),
    ]


@pytest.mark.asyncio
async def test_execute_validates_serial_tool_batch_once_per_call(tmp_path: Path) -> None:
    """同一 activation 的串行工具批次复用首次 preflight 计划。"""
    tool = _CountingSerialTool()
    registry = ToolRegistry()
    registry.register(tool)
    calls = [
        ToolUseBlock(
            id=f"counting-{index}",
            name=tool.name,
            input={"value": str(index)},
        )
        for index in range(5)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        permission_policy=DefaultPermissionPolicy(write_mode="allow"),
    )
    activation = start_activation()

    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation),
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert tool.validation_calls == {str(index): 1 for index in range(5)}


@pytest.mark.asyncio
async def test_execute_return_to_model_error_can_steer_after_final_result(
    tmp_path: Path,
) -> None:
    """RETURN_TO_MODEL 的 final error 是可领取 steer 的安全边界。"""
    call = ToolUseBlock(id="missing-1", name="missing", input={})
    provider = FakeProvider([_tool_response(call), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    steered_message = Msg.user("根据错误调整方向")
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=steered_message)]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert commits.tool_commits[0].result.is_error is True
    assert commits.tool_commits[0].message_delta[1] == steered_message
    assert provider.requests[1].messages[-1] == steered_message
    assert steering.events[1] == ("acknowledge", "submission-1", None)


@pytest.mark.asyncio
async def test_execute_model_commit_failure_fails_claimed_steer(
    tmp_path: Path,
) -> None:
    """No-tool required commit 失败时 fail exact submission 并保留原异常。"""
    provider = FakeProvider([_text_response("回答")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, fail_at="commit_model_step")
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    with pytest.raises(IrisRunPersistenceError, match="required commit"):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            steering=steering,
        )

    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("fail", "submission-1", "commit_failed"),
    ]


@pytest.mark.asyncio
async def test_execute_tool_commit_failure_fails_claimed_steer(
    tmp_path: Path,
) -> None:
    """Final-tool required commit 失败时不把 steer 误报为 delivered。"""

    def echo() -> str:
        return "echo"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider([_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, fail_at="commit_tool_result")
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    with pytest.raises(IrisRunPersistenceError, match="required commit"):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            steering=steering,
        )

    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("fail", "submission-1", "commit_failed"),
    ]


@pytest.mark.asyncio
async def test_execute_model_cursor_mismatch_fails_claimed_steer(
    tmp_path: Path,
) -> None:
    """Model commit 返回意外 cursor 时不得 acknowledge submission。"""

    class UnexpectedCursorCommitPort(FakeRuntimeCommitPort):
        def commit_model_step(self, commit: RuntimeModelStepCommit) -> RuntimeCursor:
            super().commit_model_step(commit)
            return RuntimeCursor(
                position="before_model",
                step_index=commit.cursor_after.step_index + 1,
            )

    provider = FakeProvider([_text_response("回答")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = UnexpectedCursorCommitPort(activation)
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    with pytest.raises(IrisRunConflictError, match="意外 cursor"):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            steering=steering,
        )

    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("fail", "submission-1", "commit_failed"),
    ]


@pytest.mark.asyncio
async def test_execute_tool_cursor_mismatch_fails_claimed_steer(
    tmp_path: Path,
) -> None:
    """Final-tool commit 返回意外 cursor 时不得 acknowledge submission。"""

    class UnexpectedCursorCommitPort(FakeRuntimeCommitPort):
        def commit_tool_result(self, commit: RuntimeToolResultCommit) -> RuntimeCursor:
            super().commit_tool_result(commit)
            return RuntimeCursor(
                position="before_model",
                step_index=commit.cursor_after.step_index + 1,
            )

    def echo() -> str:
        return "echo"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider([_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = UnexpectedCursorCommitPort(activation)
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    with pytest.raises(IrisRunConflictError, match="意外 cursor"):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            steering=steering,
        )

    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("fail", "submission-1", "commit_failed"),
    ]


@pytest.mark.asyncio
async def test_execute_isolates_acknowledge_callback_failure(
    tmp_path: Path,
) -> None:
    """Ack callback 违反 non-throwing contract 不覆盖 durable commit。"""
    provider = FakeProvider([_text_response("第一轮"), _text_response("最终轮")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))],
        callback_error_at="acknowledge",
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert commits.messages[-2].text == "新方向"
    assert steering.events[1] == ("acknowledge", "submission-1", None)


@pytest.mark.asyncio
async def test_execute_isolates_fail_callback_failure(
    tmp_path: Path,
) -> None:
    """Fail callback 异常不能覆盖原始 required commit 异常。"""
    provider = FakeProvider([_text_response("回答")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation, fail_at="commit_model_step")
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))],
        callback_error_at="fail",
    )

    with pytest.raises(IrisRunPersistenceError, match="required commit"):
        await runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
            steering=steering,
        )

    assert steering.events[-1] == ("fail", "submission-1", "commit_failed")


@pytest.mark.asyncio
async def test_execute_parallel_tools_steer_on_final_ordered_commit(
    tmp_path: Path,
) -> None:
    """并发完成的工具仍只在最终有序提交中携带 steer。"""

    async def read_value(index: int) -> str:
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    steered_message = Msg.user("并发结果后的新方向")
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=steered_message)]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == [
        "read-1",
        "read-2",
    ]
    assert [len(commit.message_delta) for commit in commits.tool_commits] == [1, 2]
    assert commits.tool_commits[1].message_delta[1] == steered_message
    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("acknowledge", "submission-1", None),
        ("claim", activation.run_id, activation.activation_id),
    ]


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
    steering = FakeRuntimeSteeringPort()

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["Iris"]
    assert len(provider.requests) == 2
    assert len(commits.model_commits) == 2
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].claim is not None
    assert commits.events.index("claim_tool_call") < commits.events.index("commit_tool_result")
    assert provider.requests[1].messages[-1].tool_results[0].content == "echo:Iris"
    assert steering.events == [
        ("claim", activation.run_id, activation.activation_id),
        ("claim", activation.run_id, activation.activation_id),
    ]


@pytest.mark.asyncio
async def test_execute_overlaps_two_async_safe_tool_calls(tmp_path: Path) -> None:
    entered: list[int] = []
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def read_value(index: int) -> str:
        assert f"read-{index}" in commits.claims
        entered.append(index)
        if len(entered) == 2:
            both_entered.set()
        await release.wait()
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )
    try:
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        assert entered == [1, 2]
    finally:
        release.set()
        await execution


@pytest.mark.asyncio
async def test_execute_commits_reverse_completed_calls_in_original_order(
    tmp_path: Path,
) -> None:
    entered: list[int] = []
    completion_order: list[int] = []
    both_entered = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    second_finished = asyncio.Event()

    async def read_value(index: int) -> str:
        entered.append(index)
        if len(entered) == 2:
            both_entered.set()
        if index == 1:
            await release_first.wait()
        else:
            await release_second.wait()
        completion_order.append(index)
        if index == 2:
            second_finished.set()
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    try:
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        release_second.set()
        await asyncio.wait_for(second_finished.wait(), timeout=1)
        assert commits.tool_commits == []
        release_first.set()
        result = await execution
    finally:
        release_first.set()
        release_second.set()
        if not execution.done():
            await execution

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert completion_order == [2, 1]
    assert [commit.tool_call.tool_call_id for commit in commits.tool_commits] == [
        "read-1",
        "read-2",
    ]
    assert [message.tool_results[0].tool_use_id for message in commits.messages[-3:-1]] == [
        "read-1",
        "read-2",
    ]
    assert [
        message.tool_results[0].tool_use_id for message in provider.requests[1].messages[-2:]
    ] == ["read-1", "read-2"]


@pytest.mark.asyncio
async def test_execute_limits_parallel_window_to_eight_calls(tmp_path: Path) -> None:
    active = 0
    peak = 0
    entered: list[int] = []
    first_window_entered = asyncio.Event()
    release_first_window = asyncio.Event()
    ninth_started_after_commits: int | None = None

    async def read_value(index: int) -> str:
        nonlocal active, ninth_started_after_commits, peak
        active += 1
        peak = max(peak, active)
        entered.append(index)
        if len(entered) == 8:
            first_window_entered.set()
        if index == 9:
            ninth_started_after_commits = len(commits.tool_commits)
        else:
            await release_first_window.wait()
        active -= 1
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in range(1, 10)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    try:
        await asyncio.wait_for(first_window_entered.wait(), timeout=1)
        assert entered == list(range(1, 9))
        release_first_window.set()
        result = await execution
    finally:
        release_first_window.set()
        if not execution.done():
            await execution

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert peak == 8
    assert ninth_started_after_commits == 8


@pytest.mark.asyncio
async def test_execute_keeps_unsafe_call_as_serial_barrier(tmp_path: Path) -> None:
    entered: list[str] = []
    first_pair_entered = asyncio.Event()
    unsafe_entered = asyncio.Event()
    second_pair_entered = asyncio.Event()
    release_first_pair = asyncio.Event()
    release_unsafe = asyncio.Event()
    release_second_pair = asyncio.Event()
    unsafe_started_after_commits: int | None = None
    second_pair_started_after_commits: list[int] = []

    async def safe_read(name: str) -> str:
        entered.append(name)
        if name in {"safe-1", "safe-2"}:
            if {"safe-1", "safe-2"}.issubset(entered):
                first_pair_entered.set()
            await release_first_pair.wait()
        else:
            second_pair_started_after_commits.append(len(commits.tool_commits))
            if {"safe-3", "safe-4"}.issubset(entered):
                second_pair_entered.set()
            await release_second_pair.wait()
        return name

    async def unsafe_read() -> str:
        nonlocal unsafe_started_after_commits
        entered.append("unsafe")
        unsafe_started_after_commits = len(commits.tool_commits)
        unsafe_entered.set()
        await release_unsafe.wait()
        return "unsafe"

    class SerialReadTool(CallableTool):
        def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
            del params
            return False

    registry = ToolRegistry()
    registry.register_function(safe_read, description="安全读取")
    registry.register(SerialReadTool(unsafe_read, description="串行读取"))
    calls = [
        ToolUseBlock(id="safe-1", name="safe_read", input={"name": "safe-1"}),
        ToolUseBlock(id="safe-2", name="safe_read", input={"name": "safe-2"}),
        ToolUseBlock(id="unsafe", name="unsafe_read", input={}),
        ToolUseBlock(id="safe-3", name="safe_read", input={"name": "safe-3"}),
        ToolUseBlock(id="safe-4", name="safe_read", input={"name": "safe-4"}),
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    try:
        await asyncio.wait_for(first_pair_entered.wait(), timeout=1)
        assert entered == ["safe-1", "safe-2"]
        release_first_pair.set()
        await asyncio.wait_for(unsafe_entered.wait(), timeout=1)
        assert entered == ["safe-1", "safe-2", "unsafe"]
        release_unsafe.set()
        await asyncio.wait_for(second_pair_entered.wait(), timeout=1)
        assert entered == ["safe-1", "safe-2", "unsafe", "safe-3", "safe-4"]
        release_second_pair.set()
        result = await execution
    finally:
        release_first_pair.set()
        release_unsafe.set()
        release_second_pair.set()
        if not execution.done():
            await execution

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert unsafe_started_after_commits == 2
    assert second_pair_started_after_commits == [3, 3]


@pytest.mark.asyncio
async def test_execute_return_to_model_error_does_not_cancel_sibling(
    tmp_path: Path,
) -> None:
    entered: list[int] = []
    finished: list[int] = []
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def read_value(index: int) -> str:
        entered.append(index)
        if len(entered) == 2:
            both_entered.set()
        await release.wait()
        if index == 1:
            raise RuntimeError("读取失败")
        finished.append(index)
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    try:
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        release.set()
        result = await execution
    finally:
        release.set()
        if not execution.done():
            await execution

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert finished == [2]
    assert [commit.result.is_error for commit in commits.tool_commits] == [True, False]


@pytest.mark.asyncio
async def test_execute_stop_policy_never_starts_later_safe_call(tmp_path: Path) -> None:
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def read_value(index: int) -> str:
        if index == 1:
            first_entered.set()
            await release_first.wait()
            raise RuntimeError("读取失败")
        second_entered.set()
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls)])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation(
        options=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
    )
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    try:
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        next_turn = asyncio.Event()
        asyncio.get_running_loop().call_soon(next_turn.set)
        await next_turn.wait()
        assert not second_entered.is_set()
        release_first.set()
        result = await execution
    finally:
        release_first.set()
        if not execution.done():
            await execution

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert [commit.tool_call.tool_call_id for commit in commits.tool_commits] == ["read-1"]


@pytest.mark.asyncio
async def test_execute_parallel_window_keeps_sync_callable_results_ordered(
    tmp_path: Path,
) -> None:
    executed: list[int] = []

    def read_value(index: int) -> str:
        executed.append(index)
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert executed == [1, 2]
    assert [commit.result.model_content for commit in commits.tool_commits] == [
        "value-1",
        "value-2",
    ]


@pytest.mark.asyncio
async def test_execute_parallel_file_reads_share_one_read_state(tmp_path: Path) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("first", encoding="utf-8")
    second_path.write_text("second", encoding="utf-8")
    registry = ToolRegistry()
    register_file_tools(registry=registry)

    class ReadStateObserver(ToolMiddleware):
        def __init__(self) -> None:
            self.identities: list[int] = []

        async def before_call(
            self,
            tool: BaseTool,
            params: dict[str, Any],
            context: ToolExecutionContext,
        ) -> None:
            del params
            if tool.definition.group == "file":
                assert context.read_state is not None
                self.identities.append(id(context.read_state))

    observer = ReadStateObserver()
    calls = [
        ToolUseBlock(id="read-1", name="read_file", input={"file_path": "first.txt"}),
        ToolUseBlock(id="read-2", name="read_file", input={"file_path": "second.txt"}),
        ToolUseBlock(
            id="write-1",
            name="write_file",
            input={"file_path": "first.txt", "content": "updated"},
        ),
    ]
    provider = FakeProvider([_tool_batch_response(calls), _text_response("完成")])
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=registry,
        middleware=[observer],
        permission_policy=DefaultPermissionPolicy(write_mode="allow"),
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert first_path.read_text(encoding="utf-8") == "updated"
    assert len(set(observer.identities)) == 1
    committed_call_ids = [commit.tool_call.tool_call_id for commit in commits.tool_commits]
    assert committed_call_ids == ["read-1", "read-2", "write-1"]
    expected_paths = {str(first_path.resolve()), str(second_path.resolve())}
    committed_read_states = [
        ReadFileState.model_validate(commit.cursor_after.read_state)
        for commit in commits.tool_commits[:2]
    ]
    assert all(set(state.files) == expected_paths for state in committed_read_states)
    read_state = ReadFileState.model_validate(result.cursor.read_state)
    assert set(read_state.files) == expected_paths


@pytest.mark.asyncio
async def test_execute_control_hole_commits_only_known_result_prefix(
    tmp_path: Path,
) -> None:
    """第二条 control 异常后不得提交已完成的第三条结果。"""
    all_entered = asyncio.Event()
    release = asyncio.Event()
    entered: set[int] = set()

    async def read_value(index: int) -> str:
        entered.add(index)
        if entered == {1, 2, 3}:
            all_entered.set()
        await all_entered.wait()
        if index == 1:
            return "value-1"
        if index == 2:
            await release.wait()
            raise IrisCancellationRequestedError("测试 control hole")
        return "value-3"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2, 3)
    ]
    provider = FakeProvider([_tool_batch_response(calls)])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    await asyncio.wait_for(all_entered.wait(), timeout=1)
    release.set()
    result = await asyncio.wait_for(execution, timeout=1)

    assert result.outcome is RuntimeActivationOutcome.OUTCOME_UNKNOWN
    assert [commit.tool_call.tool_call_id for commit in commits.tool_commits] == ["read-1"]
    assert result.cursor.next_tool_index == 1
    assert set(commits.claims) == {"read-1", "read-2", "read-3"}


@pytest.mark.asyncio
async def test_execute_infrastructure_failure_drains_all_window_children(
    tmp_path: Path,
) -> None:
    class ControlledInfrastructureExit(BaseException):
        pass

    entered: list[int] = []
    done: set[int] = set()
    all_entered = asyncio.Event()
    release_second_failure = asyncio.Event()
    keep_first_pending = asyncio.Event()
    keep_third_pending = asyncio.Event()

    async def read_value(index: int) -> str:
        entered.append(index)
        if len(entered) == 3:
            all_entered.set()
        try:
            if index == 1:
                try:
                    await keep_first_pending.wait()
                except asyncio.CancelledError:
                    raise ControlledInfrastructureExit("infra-1") from None
            if index == 2:
                await release_second_failure.wait()
                raise ControlledInfrastructureExit("infra-2")
            await keep_third_pending.wait()
            return "unreachable"
        finally:
            done.add(index)

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2, 3)
    ]
    provider = FakeProvider([_tool_batch_response(calls)])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    await asyncio.wait_for(all_entered.wait(), timeout=1)
    release_second_failure.set()
    with pytest.raises(ControlledInfrastructureExit, match="infra-1"):
        await asyncio.wait_for(execution, timeout=1)

    assert done == {1, 2, 3}


@pytest.mark.asyncio
async def test_execute_propagates_unexpected_child_self_cancellation(
    tmp_path: Path,
) -> None:
    entered: list[int] = []
    both_entered = asyncio.Event()
    keep_sibling_pending = asyncio.Event()
    sibling_done = asyncio.Event()

    async def read_value(index: int) -> str:
        assert f"read-{index}" in commits.claims
        entered.append(index)
        if len(entered) == 2:
            both_entered.set()
        if index == 2:
            await both_entered.wait()
            raise asyncio.CancelledError("child self cancellation")
        try:
            await keep_sibling_pending.wait()
            return "unreachable"
        finally:
            sibling_done.set()

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    calls = [
        ToolUseBlock(id=f"read-{index}", name="read_value", input={"index": index})
        for index in (1, 2)
    ]
    provider = FakeProvider([_tool_batch_response(calls)])
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    try:
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        await asyncio.wait_for(sibling_done.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError, match="child self cancellation"):
            await execution
    finally:
        if not execution.done():
            execution.cancel()
            try:
                await execution
            except asyncio.CancelledError:
                pass


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
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.SUSPENDED
    assert result.suspension is not None
    assert result.cursor.position == "tool_batch"
    assert commits.model_commits == []
    assert len(commits.suspensions) == 1
    assert "suspend" in commits.events
    assert "claim_tool_call" not in commits.events
    assert steering.events == []


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
    assert effects == []
    assert commits.claims == {}
    assert commits.tool_commits == []
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
    projection = _approved_projection(waiting).model_copy(update={"tool_call_id": "other-call"})
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
    provider = FakeProvider([_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))])
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
async def test_execute_awaits_memory_query_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "runtime-memory.db", use_fts=False)
    service = MemoryService(store, io_execution_mode=MemoryIOExecutionMode.THREAD)
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    service.remember(
        MemoryWriteInput(
            scope=scope,
            text="用户喜欢简洁回答",
            reason="test seed",
        )
    )
    loop_thread = threading.get_ident()
    search_threads: list[int] = []
    original_search = store.search

    def search(query: MemoryQuery) -> list[MemorySearchResult]:
        search_threads.append(threading.get_ident())
        return original_search(query)

    monkeypatch.setattr(store, "search", search)
    provider = FakeProvider([_text_response("完成")])
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        memory_service=service,
    )
    activation = start_activation(
        options=RuntimeExecutionOptions(
            memory_query=MemoryQuery(
                scope=scope,
                text="简洁",
            ).model_dump(mode="json"),
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
    assert search_threads and all(thread_id != loop_thread for thread_id in search_threads)
    assert any("用户喜欢简洁回答" in message.text for message in provider.requests[0].messages)


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
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="submission-1", message=Msg.user("新方向"))]
    )

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "TOOL_NOT_ALLOWED"
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].claim is None
    assert steering.events == []


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
    provider = FakeProvider([_tool_response(ToolUseBlock(id="effect-1", name="effect", input={}))])
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
    provider = FakeProvider([_tool_response(ToolUseBlock(id="write-1", name="write", input={}))])
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
    provider = FakeProvider([_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))])
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
async def test_thread_callable_timeout_after_claim_discards_late_result(
    tmp_path: Path,
) -> None:
    """tool timeout 只停止等待 thread worker，claim 必须以 unknown 收口。"""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_effect() -> str:
        started.set()
        try:
            release.wait(timeout=2)
            return "late-effect-complete"
        finally:
            finished.set()

    registry = ToolRegistry()
    registry.register_function(
        blocking_effect,
        description="线程阻塞工具",
        execution_mode=CallableExecutionMode.THREAD,
    )
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="thread-1", name="blocking_effect", input={}))]
    )
    runtime = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    activation = start_activation(options=RuntimeExecutionOptions(tool_timeout_seconds=0.02))
    commits = FakeRuntimeCommitPort(activation)
    execution = asyncio.create_task(
        runtime.execute(
            activation,
            commits=commits,
            cancellation=MutableCancellationSignal(),
        )
    )

    try:
        assert await asyncio.to_thread(started.wait, 1)
        result = await asyncio.wait_for(execution, timeout=1)
        assert result.outcome is RuntimeActivationOutcome.OUTCOME_UNKNOWN
        assert result.error is not None and result.error.code == "TOOL_OUTCOME_UNKNOWN"
        assert len(commits.claims) == 1
        assert commits.tool_commits == []
        events_before_release = list(commits.events)

        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        for _ in range(3):
            await asyncio.sleep(0)

        assert commits.events == events_before_release
        assert commits.tool_commits == []
    finally:
        release.set()
        if not execution.done():
            await execution


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
    provider = FakeProvider([_tool_response(ToolUseBlock(id="effect-1", name="effect", input={}))])
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
    provider = FakeProvider([_tool_response(ToolUseBlock(id="echo-1", name="echo", input={}))])
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
