"""Lifecycle store 的 command 与同步协议。

所有 mutation 只接收一个显式 command，使 CAS、activation fence 与历史 revision
成为可验证输入。

Example:
    store.create_run(command)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ..hitl.models import HumanInteraction, HumanInteractionResponse
from ..message.message import Msg
from ..tools.base import ToolResult
from .history import ForkPointCursor, ForkPointPage, RunHistorySnapshot
from .models import (
    ActivationKind,
    AgentRunOptions,
    AgentRunRequest,
    RecoveryDisposition,
    RunCheckpoint,
    RunControlSnapshot,
    RunErrorInfo,
    RunEvent,
    RunRecord,
    RunResult,
    RunStopReason,
    RunToolCallRecord,
    RunUsage,
    SessionCompaction,
    SessionSnapshot,
    SubagentRunLink,
    TokenUsage,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ForkSession:
    """从 terminal 顶层 run 的消息截点原子创建独立 session。"""

    source_run_id: str
    target_session_id: str
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateRun:
    """原子创建 logical run、session lane 与初始 activation。"""

    request: AgentRunRequest
    options: AgentRunOptions
    agent_id: str
    environment_fingerprint: str
    start_activation_id: str
    initial_checkpoint: RunCheckpoint
    now: datetime

    def __post_init__(self) -> None:
        """校验初始 checkpoint 与 run identity 的跨字段约束。"""
        if self.request.run_id is None:
            raise ValueError("CreateRun request 必须包含最终 run_id")
        if self.initial_checkpoint.run_id != self.request.run_id:
            raise ValueError("initial checkpoint 必须绑定 request run_id")
        if self.initial_checkpoint.activation_id != self.start_activation_id:
            raise ValueError("initial checkpoint 必须绑定 start activation")
        if self.initial_checkpoint.sequence != 1:
            raise ValueError("initial checkpoint sequence 必须为 1")
        if self.initial_checkpoint.environment_fingerprint != self.environment_fingerprint:
            raise ValueError("initial checkpoint environment fingerprint 不匹配")


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmitChildRun:
    """同事务创建普通 child run 并绑定 exact parent prepared tool。"""

    parent_run_id: str
    expected_parent_run_revision: int
    parent_activation_id: str
    parent_tool_call_id: str
    expected_parent_tool_version: int
    child_create: CreateRun


@dataclass(frozen=True, slots=True, kw_only=True)
class RebindSubagentProxy:
    """保持 parent 工具未提交，首次绑定或替换已回答的 proxy。"""

    parent_run_id: str
    expected_parent_run_revision: int
    parent_activation_id: str | None
    parent_tool_call_id: str
    expected_parent_tool_version: int
    pending_proxy: HumanInteraction
    replaced_proxy_interaction_id: str | None
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class FinalizeSubagentResult:
    """Child terminal 后原子提交 parent 结果并按需创建 RESUME activation。"""

    parent_run_id: str
    expected_parent_run_revision: int
    parent_activation_id: str | None
    expected_parent_session_revision: int
    parent_tool_call_id: str
    expected_parent_tool_version: int
    proxy_interaction_id: str | None
    resume_activation_id: str | None
    result: ToolResult
    message_delta: list[Msg]
    checkpoint: RunCheckpoint
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeWaitingRun:
    """为 resolved waiting run 创建新的 activation fence。"""

    run_id: str
    expected_run_revision: int
    new_activation_id: str
    kind: ActivationKind
    expected_checkpoint_sequence: int
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class ReserveModelStep:
    """在 provider effect 前 durable 预留一个模型步。"""

    run_id: str
    expected_run_revision: int
    activation_id: str
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordCompactionUsage:
    """保存一份已返回的摘要响应用量，不推进主步骤或历史。"""

    run_id: str
    expected_run_revision: int
    activation_id: str
    usage: TokenUsage
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class CommitCompaction:
    """原子替换摘要投影并保存同一执行位置的 checkpoint。"""

    run_id: str
    expected_run_revision: int
    activation_id: str
    expected_session_revision: int
    compaction: SessionCompaction
    checkpoint: RunCheckpoint
    before_input_tokens: int
    after_input_tokens: int
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class CommitModelStep:
    """原子提交模型响应、history、tool intents 与 checkpoint。"""

    run_id: str
    expected_run_revision: int
    activation_id: str
    expected_session_revision: int
    message_delta: list[Msg] = field(default_factory=list)
    usage: RunUsage
    prepared_tool_calls: list[RunToolCallRecord] = field(default_factory=list)
    checkpoint: RunCheckpoint
    assistant_message: Msg | None = None
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimToolCall:
    """在工具副作用前 durable claim 一次 prepared 调用。"""

    run_id: str
    expected_run_revision: int
    activation_id: str
    tool_call_id: str
    fingerprint: str
    expected_tool_version: int
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class CommitToolResult:
    """原子提交一次精确工具结果、history 与 checkpoint。"""

    run_id: str
    expected_run_revision: int
    activation_id: str
    expected_session_revision: int
    tool_call_id: str
    expected_tool_version: int
    result: ToolResult
    message_delta: list[Msg] = field(default_factory=list)
    checkpoint: RunCheckpoint
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class SuspendRun:
    """原子提交当前模型步并将 active run 转为 waiting。"""

    run_id: str
    expected_run_revision: int
    activation_id: str
    expected_session_revision: int
    message_delta: list[Msg] = field(default_factory=list)
    prepared_tool_calls: list[RunToolCallRecord] = field(default_factory=list)
    checkpoint: RunCheckpoint
    pending_interaction: HumanInteraction
    assistant_message: Msg | None = None
    usage: RunUsage
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolveInteraction:
    """以 response/version/fingerprint CAS 解决 pending interaction。"""

    run_id: str
    expected_run_revision: int
    interaction_id: str
    expected_interaction_version: int
    response: HumanInteractionResponse
    expected_fingerprint: str
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestCancellation:
    """记录首次 cancellation request，可显式结算 waiting run。"""

    run_id: str
    expected_run_revision: int
    activation_id: str | None = None
    reason: str
    settle_waiting: bool = False
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class FinishRun:
    """将 active 或 waiting run 原子结算为 terminal。"""

    run_id: str
    expected_run_revision: int
    activation_id: str | None = None
    stop_reason: RunStopReason
    assistant_message: Msg | None = None
    error: RunErrorInfo | None = None
    interaction_close_reason: str | None = None
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class RecoverActiveRun:
    """根据 durable effect facts 放弃并恢复或终止旧 activation。"""

    run_id: str
    expected_run_revision: int
    expected_activation_id: str
    expected_checkpoint_sequence: int
    recovery_disposition: RecoveryDisposition
    new_activation_id: str | None = None
    now: datetime

    def __post_init__(self) -> None:
        """校验 recovery disposition 与新 activation 的组合。"""
        if (self.recovery_disposition is RecoveryDisposition.RESUME) != (
            self.new_activation_id is not None
        ):
            raise ValueError("resume recovery 必须且只能包含 new_activation_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCommit:
    """一次 mutation 的事实回执；session 仅返回发生变化时的 revision。"""

    run: RunRecord
    session_revision: int | None = None
    checkpoint: RunCheckpoint | None = None
    interaction: HumanInteraction | None = None
    events: tuple[RunEvent, ...] = ()
    result: RunResult | None = None


class LifecycleStore(Protocol):
    """Logical run aggregate 的同步 durable boundary。"""

    def create_run(self, command: CreateRun) -> RunCommit: ...

    def admit_child_run(self, command: AdmitChildRun) -> SubagentRunLink: ...

    def rebind_subagent_proxy(self, command: RebindSubagentProxy) -> RunCommit: ...

    def finalize_subagent_result(self, command: FinalizeSubagentResult) -> RunCommit: ...

    def load_subagent_link(
        self, parent_run_id: str, parent_tool_call_id: str
    ) -> SubagentRunLink | None: ...

    def resume_waiting_run(self, command: ResumeWaitingRun) -> RunCommit: ...

    def reserve_model_step(self, command: ReserveModelStep) -> RunCommit: ...

    def record_compaction_usage(self, command: RecordCompactionUsage) -> RunCommit: ...

    def commit_compaction(self, command: CommitCompaction) -> RunCommit: ...

    def commit_model_step(self, command: CommitModelStep) -> RunCommit: ...

    def claim_tool_call(self, command: ClaimToolCall) -> RunCommit: ...

    def commit_tool_result(self, command: CommitToolResult) -> RunCommit: ...

    def suspend_run(self, command: SuspendRun) -> RunCommit: ...

    def resolve_interaction(self, command: ResolveInteraction) -> RunCommit: ...

    def request_cancellation(self, command: RequestCancellation) -> RunCommit: ...

    def finish_run(self, command: FinishRun) -> RunCommit: ...

    def recover_active_run(self, command: RecoverActiveRun) -> RunCommit: ...

    def load_run(self, run_id: str) -> RunRecord | None: ...

    def load_run_control(self, run_id: str) -> RunControlSnapshot | None: ...

    def load_session(self, session_id: str) -> SessionSnapshot: ...

    def list_fork_points(
        self,
        session_id: str,
        *,
        after: ForkPointCursor | None = None,
        limit: int = 50,
    ) -> ForkPointPage:
        """按创建时间与 run ID 升序分页返回 terminal 顶层 run。"""
        ...

    def load_session_at_run(self, source_run_id: str) -> RunHistorySnapshot:
        """读取指定合格 run 末尾的已提交历史，不包含后续轮次。"""
        ...

    def fork_session(self, command: ForkSession) -> SessionSnapshot:
        """同事务复制来源历史和 lineage，不创建运行或占用 lane。"""
        ...

    def load_session_lane(self, session_id: str) -> str | None: ...

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None: ...

    def load_checkpoint(self, run_id: str) -> RunCheckpoint | None: ...

    def load_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
    ) -> RunToolCallRecord | None: ...

    def list_tool_calls(
        self, run_id: str, *, step_index: int | None = None
    ) -> list[RunToolCallRecord]:
        """按 step/ordinal 返回工具事实，可限定为一个模型步。"""
        ...

    def load_result(self, run_id: str) -> RunResult | None: ...

    def list_events(
        self,
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]: ...


__all__ = [
    "ResumeWaitingRun",
    "ClaimToolCall",
    "CommitModelStep",
    "CommitCompaction",
    "RecordCompactionUsage",
    "CommitToolResult",
    "CreateRun",
    "FinishRun",
    "LifecycleStore",
    "RecoverActiveRun",
    "RequestCancellation",
    "ReserveModelStep",
    "ResolveInteraction",
    "RunCommit",
    "SuspendRun",
]
