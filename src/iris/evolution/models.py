"""项目经验的持久材料、处理范围与一次维护结果。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvolutionSource(BaseModel):
    """真实经历的 lifecycle 身份，不复制运行资格。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    lifecycle_source_id: str
    run_id: str
    session_id: str


class EvolutionSession(BaseModel):
    """宿主显式请求的可选会话归属，不伪造 Run。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    lifecycle_source_id: str
    session_id: str


class RevisionTarget(BaseModel):
    """候选目标身份；开放范围由 revision 边界拥有。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["prompt", "config"]
    name: str = Field(pattern=r"\S")


class RevisionEvidence(BaseModel):
    """问题引用的已有记录与必要原文片段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str = Field(pattern=r"\S")
    quote: str = Field(pattern=r"\S")


class ExperienceOrigin(BaseModel):
    """A 从真实材料提炼的问题来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["experience"] = "experience"
    sources: tuple[EvolutionSource, ...] = Field(min_length=1)


class HostOrigin(BaseModel):
    """宿主明确请求，可无历史失败或 Run 归属。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["host"] = "host"
    session: EvolutionSession | None = None


class RevisionRequest(BaseModel):
    """宿主显式提交的有限目标修订请求。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    description: str = Field(pattern=r"\S")
    targets: tuple[RevisionTarget, ...] = Field(min_length=1)
    session: EvolutionSession | None = None


class RevisionItem(BaseModel):
    """B 的有界工作项；经历问题和宿主请求共享处理队列。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(default_factory=lambda: uuid4().hex)
    description: str = Field(pattern=r"\S")
    targets: tuple[RevisionTarget, ...] = Field(min_length=1)
    evidence: tuple[RevisionEvidence, ...] = ()
    origin: Annotated[ExperienceOrigin | HostOrigin, Field(discriminator="kind")]


class EvolutionRecord(BaseModel):
    """已提交消息中的原文片段与稳定引用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str
    message_ordinal: int = Field(ge=0)
    block_index: int = Field(ge=0)
    role: str
    text: str
    occurred_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionRange(BaseModel):
    """一个来源的消息半开区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: EvolutionSource
    start_message_count: int = Field(ge=0)
    end_message_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> EvolutionRange:
        if self.end_message_count < self.start_message_count:
            raise ValueError("消息区间结束位置不能早于开始位置")
        return self


class EvolutionMaterial(EvolutionRange):
    """一次读取的完整消息范围，允许过滤后没有正文。"""

    records: tuple[EvolutionRecord, ...] = ()


class EvolutionCaptureBlock(EvolutionMaterial):
    """独立完整发布的捕获块，允许跨进程重复区间。"""

    initial_message_count: int = Field(ge=0)
    terminal_message_count: int | None = Field(default=None, ge=0)
    outcome: str | None = None

    @model_validator(mode="after")
    def _capture_bounds(self) -> EvolutionCaptureBlock:
        if self.initial_message_count > self.start_message_count:
            raise ValueError("捕获不能包含 Run 之前的历史")
        if (
            self.terminal_message_count is not None
            and self.end_message_count != self.terminal_message_count
        ):
            raise ValueError("封源必须捕获到终态消息位置")
        if any(
            not self.start_message_count <= record.message_ordinal < self.end_message_count
            for record in self.records
        ):
            raise ValueError("记录必须位于捕获块的消息区间内")
        return self


@dataclass(frozen=True, slots=True)
class EvolutionSourceState:
    """捕获块和已确认进度的来源投影。"""

    source: EvolutionSource
    initial_message_count: int
    captured_until: int
    consumed_until: int
    terminal_message_count: int | None = None
    outcome: str | None = None


@dataclass(frozen=True, slots=True)
class PendingMaterials:
    """已按来源过滤的有界完整消息批次。"""

    items: tuple[EvolutionMaterial, ...]
    has_more: bool


@dataclass(frozen=True, slots=True)
class EvolutionMaintenanceScope:
    """宿主传入的合格来源范围与提交前资格检查。"""

    allowed_sources: frozenset[tuple[str, str]]
    check: Callable[[tuple[EvolutionSource, ...]], Awaitable[bool]]
    allowed_sessions: frozenset[tuple[str, str]]
    check_session: Callable[[EvolutionSession | None], Awaitable[bool]]
    requested_revision_id: str | None = None
    experience_only: bool = False


class EvolutionResult(BaseModel):
    """单次 A 或 B 的短结果，不保存模型调用轨迹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: Literal["experience", "revision"] = "experience"
    status: Literal["updated", "no_change", "empty", "failed", "cancelled", "conflict"]
    reason: str = ""
    consumed_ranges: tuple[EvolutionRange, ...] = ()
    usage: dict[str, int] = Field(default_factory=dict)
    has_more: bool = False
    effect: str = ""
    revision_id: str | None = None
    targets: tuple[RevisionTarget, ...] = ()
