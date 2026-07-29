from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from iris.agents import AgentConfig
from iris.context import ContextBuilder, ContextBuildInput
from iris.exceptions import IrisProviderError, IrisRunConflictError, IrisRunPersistenceError
from iris.hitl import (
    HumanInteraction,
    HumanInteractionService,
    InMemoryInteractionStore,
    InteractionStore,
    PermissionPrompt,
    QuestionPrompt,
)
from iris.lifecycle import CheckpointResumability, RuntimeExecutionOptions, SessionSnapshot
from iris.memory import MemoryContextBuilder, MemoryService
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.runtime import (
    AgentRuntime,
    ModelStepReservation,
    RuntimeActivationInput,
    RuntimeApprovedToolCall,
    RuntimeCursor,
    RuntimeEnvironment,
    RuntimeMessageAssembler,
    RuntimeModelStepCommit,
    RuntimeProvider,
    RuntimeSuspension,
    RuntimeSuspensionResult,
    RuntimeToolCall,
    RuntimeToolResultCommit,
    ToolBridge,
    ToolCallClaim,
)
from iris.session import InMemorySessionStore, SessionStore
from iris.tools import (
    DefaultPermissionPolicy,
    PermissionPolicy,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryView,
    ToolResult,
)


class MutableCancellationSignal:
    """测试用 activation-scope 取消信号。"""

    def __init__(self, *, requested: bool = False) -> None:
        self.requested = requested

    def raise_if_requested(self) -> None:
        """在 signal 已置位时抛出协作式取消异常。"""
        if self.requested:
            from iris.tools import CancellationRequestedError

            raise CancellationRequestedError("测试 activation 已取消")


class FakeRuntimeCommitPort:
    """严格记录 required commit 顺序的内存测试端口。"""

    def __init__(
        self,
        activation: RuntimeActivationInput,
        *,
        messages: Sequence[object] = (),
        max_model_steps: int = 20,
        remaining_deadline_seconds: float | None = None,
        cancellation_requested: bool = False,
        fail_at: str | None = None,
    ) -> None:
        from iris.message import Msg

        self.activation = activation
        self.cursor = activation.cursor
        self.messages = [Msg.model_validate(message) for message in messages]
        self.max_model_steps = max_model_steps
        self.deadline = remaining_deadline_seconds
        self.cancel_requested = cancellation_requested
        self.fail_at = fail_at
        self.events: list[str] = []
        self.model_commits: list[RuntimeModelStepCommit] = []
        self.tool_commits: list[RuntimeToolResultCommit] = []
        self.suspensions: list[RuntimeSuspension] = []
        self.claims: dict[str, tuple[RuntimeToolCall, ToolCallClaim]] = {}
        self._reserved = 0
        self._revision = 0
        self._outstanding_reservation: RuntimeCursor | None = None
        self._prepared_calls: dict[str, RuntimeToolCall] = {}
        self._interaction_kinds: dict[str, str] = {}

    def load_session(self) -> SessionSnapshot:
        """返回当前 revisioned history。"""
        self._record("load_session")
        return SessionSnapshot(
            session_id=self.activation.session_id,
            revision=self._revision,
            messages=list(self.messages),
        )

    def reserve_model_step(self, cursor: RuntimeCursor) -> ModelStepReservation:
        """校验 cursor 并模拟预算预留。"""
        self._record("reserve_model_step")
        self._require_cursor(cursor)
        if self._outstanding_reservation is not None:
            raise IrisRunConflictError("fake port 不允许重叠 model reservation")
        granted = self._reserved < self.max_model_steps
        if granted:
            self._reserved += 1
            self._outstanding_reservation = cursor
        return ModelStepReservation(
            granted=granted,
            step_index=cursor.step_index,
            cursor=cursor,
            remaining_deadline_seconds=self.deadline,
        )

    def commit_model_step(self, commit: RuntimeModelStepCommit) -> RuntimeCursor:
        """提交模型消息与 cursor。"""
        self._record("commit_model_step")
        self._require_cursor(commit.cursor_before)
        self._consume_reservation(commit.cursor_before)
        self._require_model_transition(commit)
        self._remember_prepared_calls(commit.prepared_tool_calls)
        self.messages.extend(commit.message_delta)
        self.cursor = commit.cursor_after
        self._revision += 1
        self.model_commits.append(commit)
        return self.cursor

    def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
        """对 exact tool subject 提交幂等 claim。"""
        self._record("claim_tool_call")
        self._require_current_tool_call(call)
        existing = self.claims.get(call.tool_call_id)
        if existing is not None:
            if existing[0] != call:
                raise IrisRunConflictError("fake port tool claim identity 已变化")
            return existing[1]
        claim = ToolCallClaim(
            run_id=call.run_id,
            activation_id=call.activation_id,
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            fingerprint=call.fingerprint,
            tool_version=call.tool_version + 1,
        )
        self.claims[call.tool_call_id] = (call, claim)
        return claim

    def commit_tool_result(self, commit: RuntimeToolResultCommit) -> RuntimeCursor:
        """提交工具结果、消息与 cursor。"""
        self._record("commit_tool_result")
        self._require_current_tool_call(commit.tool_call)
        if commit.claim is not None:
            stored = self.claims.get(commit.tool_call.tool_call_id)
            if stored is None or stored != (commit.tool_call, commit.claim):
                raise IrisRunConflictError("fake port tool result claim 不匹配")
        elif not self._claimless_result_allowed(commit):
            raise IrisRunConflictError("fake port 普通 effect result 缺少 durable claim")
        self._require_single_step_cursor(commit)
        self.messages.extend(commit.message_delta)
        self.cursor = commit.cursor_after
        self._revision += 1
        self.tool_commits.append(commit)
        return self.cursor

    def suspend(self, suspension: RuntimeSuspension) -> RuntimeSuspensionResult:
        """原子提交模型消息、cursor 与 pending interaction。"""
        self._record("suspend")
        self._require_cursor(suspension.cursor_before)
        if suspension.cursor_before.position == "before_model":
            self._consume_reservation(suspension.cursor_before)
        elif self._outstanding_reservation is not None:
            raise IrisRunConflictError("fake port tool-batch suspend 遇到未消费 reservation")
        self._require_suspension_transition(suspension)
        self._remember_prepared_calls(suspension.prepared_tool_calls)
        prompt = suspension.interaction_request.prompt
        if isinstance(prompt, QuestionPrompt):
            interaction_kind = "question"
        elif isinstance(prompt, PermissionPrompt):
            interaction_kind = "permission"
        else:
            raise IrisRunConflictError("fake port 收到未知 interaction prompt")
        self._interaction_kinds[
            suspension.interaction_request.tool_call.tool_call_id
        ] = interaction_kind
        self.messages.extend(suspension.message_delta)
        self.cursor = suspension.cursor
        self._revision += 1
        self.suspensions.append(suspension)
        interaction = HumanInteraction(
            session_id=self.activation.session_id,
            run_id=self.activation.run_id,
            step_index=suspension.cursor.step_index,
            request=suspension.interaction_request,
            checkpoint={
                "checkpoint_version": 1,
                "resumability": CheckpointResumability.SAFE.value,
                "engine_cursor": suspension.cursor.model_dump(mode="json"),
            },
            expires_at=suspension.expires_at,
        )
        return RuntimeSuspensionResult(cursor=self.cursor, interaction=interaction)

    def cancellation_requested(self) -> bool:
        """返回 durable cancellation request。"""
        self._record("cancellation_requested")
        return self.cancel_requested

    def remaining_deadline_seconds(self) -> float | None:
        """返回测试配置的剩余 deadline。"""
        self._record("remaining_deadline_seconds")
        return self.deadline

    def _require_cursor(self, cursor: RuntimeCursor) -> None:
        if cursor != self.cursor:
            raise IrisRunConflictError("fake port cursor 已过期")

    def _consume_reservation(self, cursor: RuntimeCursor) -> None:
        if self._outstanding_reservation != cursor:
            raise IrisRunConflictError("fake port model commit/suspend 缺少对应 reservation")
        self._outstanding_reservation = None

    def _remember_prepared_calls(self, calls: Sequence[RuntimeToolCall]) -> None:
        self._prepared_calls = {call.tool_call_id: call for call in calls}

    def _require_model_transition(self, commit: RuntimeModelStepCommit) -> None:
        before = commit.cursor_before
        after = commit.cursor_after
        assistant = commit.assistant_message
        if before.position != "before_model":
            raise IrisRunConflictError("fake port model commit 必须从 before_model 开始")
        if not commit.message_delta or commit.message_delta[-1] != assistant:
            raise IrisRunConflictError("fake port model commit 缺少精确 assistant delta")
        calls = tuple(assistant.tool_calls)
        if calls:
            valid = (
                after.position == "tool_batch"
                and after.step_index == before.step_index
                and after.next_tool_index == 0
                and after.tool_calls == calls
                and not after.tool_results
                and after.assistant_message == assistant
                and self._prepared_match_calls(commit.prepared_tool_calls, calls)
            )
        else:
            valid = (
                after.position == "outcome_ready"
                and after.step_index == before.step_index
                and not after.tool_calls
                and not after.tool_results
                and after.assistant_message == assistant
                and not commit.prepared_tool_calls
            )
        if not valid:
            raise IrisRunConflictError("fake port model commit cursor/message/prepared 转换无效")

    def _require_suspension_transition(self, suspension: RuntimeSuspension) -> None:
        before = suspension.cursor_before
        cursor = suspension.cursor
        assistant = suspension.assistant_message
        calls = tuple(assistant.tool_calls)
        if before.position == "before_model":
            valid = (
                bool(suspension.message_delta)
                and suspension.message_delta[-1] == assistant
                and cursor.position == "tool_batch"
                and cursor.step_index == before.step_index
                and cursor.next_tool_index == 0
                and cursor.tool_calls == calls
                and cursor.assistant_message == assistant
                and self._prepared_match_calls(suspension.prepared_tool_calls, calls)
            )
        else:
            remaining = before.tool_calls[before.next_tool_index :]
            valid = (
                not suspension.message_delta
                and cursor == before
                and assistant == before.assistant_message
                and self._prepared_match_calls(
                    suspension.prepared_tool_calls,
                    remaining,
                    start=before.next_tool_index + 1,
                )
                and bool(suspension.prepared_tool_calls)
                and suspension.interaction_request.tool_call.tool_call_id
                == suspension.prepared_tool_calls[0].tool_call_id
            )
        request_call = suspension.interaction_request.tool_call
        prepared_gate = next(
            (
                call
                for call in suspension.prepared_tool_calls
                if call.tool_call_id == request_call.tool_call_id
            ),
            None,
        )
        if (
            not valid
            or prepared_gate is None
            or prepared_gate.tool_name != request_call.tool_name
            or prepared_gate.arguments != request_call.arguments
            or prepared_gate.fingerprint != request_call.fingerprint
        ):
            raise IrisRunConflictError("fake port suspension cursor/message/gate 转换无效")

    @staticmethod
    def _prepared_match_calls(
        prepared: Sequence[RuntimeToolCall],
        calls: Sequence[ToolUseBlock],
        *,
        start: int = 1,
    ) -> bool:
        return len(prepared) == len(calls) and all(
            fact.ordinal == ordinal
            and fact.tool_call_id == call.id
            and fact.tool_name == call.name
            for ordinal, fact, call in zip(
                range(start, start + len(calls)),
                prepared,
                calls,
                strict=True,
            )
        )

    def _claimless_result_allowed(self, commit: RuntimeToolResultCommit) -> bool:
        result = commit.result
        kind = self._interaction_kinds.get(commit.tool_call.tool_call_id)
        if kind == "question":
            return not result.is_error
        if kind == "permission":
            return (
                result.is_error
                and result.error is not None
                and result.error.code == "USER_REJECTED"
            )
        return (
            result.is_error
            and result.error is not None
            and result.error.code
            in {
                "CIRCUIT_OPEN",
                "NOT_FOUND",
                "PERMISSION_ERROR",
                "TOOL_NOT_ALLOWED",
                "VALIDATION_ERROR",
            }
        )

    def _require_current_tool_call(self, call: RuntimeToolCall) -> None:
        if (
            call.run_id != self.activation.run_id
            or call.activation_id != self.activation.activation_id
        ):
            raise IrisRunConflictError("fake port 收到跨 run/activation tool fact")
        if self.cursor.position != "tool_batch":
            raise IrisRunConflictError("fake port 当前 cursor 不是 tool batch")
        expected = self.cursor.tool_calls[self.cursor.next_tool_index]
        if (
            call.step_index != self.cursor.step_index
            or call.ordinal != self.cursor.next_tool_index + 1
            or call.tool_call_id != expected.id
            or call.tool_name != expected.name
        ):
            raise IrisRunConflictError("fake port tool fact 跳过或改变当前 cursor 调用")
        prepared = self._prepared_calls.get(call.tool_call_id)
        if prepared is not None and (
            call.step_index != prepared.step_index
            or call.ordinal != prepared.ordinal
            or call.tool_call_id != prepared.tool_call_id
            or call.tool_name != prepared.tool_name
            or call.arguments != prepared.arguments
            or call.fingerprint != prepared.fingerprint
        ):
            raise IrisRunConflictError("fake port tool fact 与 prepared subject 不匹配")

    def _require_single_step_cursor(self, commit: RuntimeToolResultCommit) -> None:
        before = self.cursor
        after = commit.cursor_after
        expected_results = (*before.tool_results, commit.result)
        is_last = before.next_tool_index + 1 == len(before.tool_calls)
        if is_last:
            valid = (
                after.position == "before_model"
                and after.step_index == before.step_index + 1
                and not after.tool_calls
                and not after.tool_results
            )
        else:
            valid = (
                after.position == "tool_batch"
                and after.step_index == before.step_index
                and after.next_tool_index == before.next_tool_index + 1
                and after.tool_calls == before.tool_calls
                and after.tool_results == expected_results
                and after.assistant_message == before.assistant_message
            )
        if not valid:
            raise IrisRunConflictError("fake port tool result 必须精确推进一个 cursor 调用")

    def _record(self, operation: str) -> None:
        self.events.append(operation)
        if self.fail_at == operation:
            raise IrisRunPersistenceError("模拟 required commit 失败", operation=operation)


def start_activation(
    *,
    input: str = "当前问题",
    run_id: str = "run-1",
    activation_id: str = "activation-1",
    session_id: str = "session-1",
    options: RuntimeExecutionOptions | None = None,
) -> RuntimeActivationInput:
    """构造从 step 0 开始的 start activation。"""
    return RuntimeActivationInput(
        run_id=run_id,
        activation_id=activation_id,
        session_id=session_id,
        kind="start",
        input=input,
        cursor=RuntimeCursor(position="before_model", step_index=0),
        options=options or RuntimeExecutionOptions(),
    )


def resume_activation(
    cursor: RuntimeCursor,
    *,
    run_id: str = "run-1",
    activation_id: str = "activation-2",
    session_id: str = "session-1",
    options: RuntimeExecutionOptions | None = None,
    kind: str = "resume",
    interaction_projection: ToolResult | RuntimeApprovedToolCall | None = None,
) -> RuntimeActivationInput:
    """构造绑定 durable cursor 的 resume/recover activation。"""
    return RuntimeActivationInput(
        run_id=run_id,
        activation_id=activation_id,
        session_id=session_id,
        kind=kind,
        input=None,
        cursor=cursor,
        options=options or RuntimeExecutionOptions(),
        interaction_projection=interaction_projection,
    )


class FakeProvider:
    """测试用 provider，只记录请求并按顺序返回预设响应。"""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self._requests: list[LLMRequest] = []

    @property
    def requests(self) -> list[LLMRequest]:
        """返回已捕获的请求快照。"""
        return list(self._requests)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求并返回下一条预设响应。"""
        self._requests.append(request)
        if not self._responses:
            raise IrisProviderError("FakeProvider 响应已耗尽", provider="fake")
        return self._responses.pop(0)


def build_runtime(
    *,
    agent_config: AgentConfig,
    context_input: ContextBuildInput,
    provider: RuntimeProvider,
    session_store: SessionStore | None = None,
    context_builder: ContextBuilder | None = None,
    assembler: RuntimeMessageAssembler | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_view: ToolRegistryView | None = None,
    tool_executor: ToolExecutor | None = None,
    workspace_root: Path | None = None,
    permission_policy: PermissionPolicy | None = None,
    interaction_store: InteractionStore | None = None,
    interaction_service: HumanInteractionService | None = None,
    memory_service: MemoryService | None = None,
    memory_context_builder: MemoryContextBuilder | None = None,
) -> AgentRuntime:
    """为测试构造包含一致依赖图的 runtime。"""
    resolved_session_store = session_store or InMemorySessionStore()
    registry = tool_registry or (tool_view.registry if tool_view is not None else ToolRegistry())
    resolved_tool_view = tool_view or registry.view()
    resolved_policy = permission_policy or DefaultPermissionPolicy()
    resolved_tool_executor = tool_executor or ToolExecutor(
        registry,
        permission_policy=resolved_policy,
    )
    resolved_interaction_service = interaction_service or HumanInteractionService(
        interaction_store or InMemoryInteractionStore()
    )
    environment = RuntimeEnvironment(
        agent_config=agent_config,
        context_input=context_input,
        provider=provider,
        session_store=resolved_session_store,
        context_builder=context_builder or ContextBuilder(),
        assembler=assembler or RuntimeMessageAssembler(),
        tool_bridge=ToolBridge(
            tool_view=resolved_tool_view,
            tool_executor=resolved_tool_executor,
        ),
        interaction_service=resolved_interaction_service,
        workspace_root=workspace_root or Path.cwd(),
        memory_service=memory_service,
        memory_context_builder=memory_context_builder or MemoryContextBuilder(),
    )
    return AgentRuntime(environment)
