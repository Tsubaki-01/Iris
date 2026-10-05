"""项目经验的持久材料、处理范围与一次维护结果。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvolutionSource(BaseModel):
    """真实经历的 lifecycle 身份，不复制运行资格。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    lifecycle_source_id: str
    run_id: str
    session_id: str


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


class EvolutionSourceState(BaseModel):
    """捕获块和已确认进度的来源投影。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: EvolutionSource
    initial_message_count: int = Field(ge=0)
    captured_until: int = Field(ge=0)
    consumed_until: int = Field(ge=0)
    terminal_message_count: int | None = Field(default=None, ge=0)
    outcome: str | None = None

    @model_validator(mode="after")
    def _state_bounds(self) -> EvolutionSourceState:
        if not self.initial_message_count <= self.consumed_until <= self.captured_until:
            raise ValueError("消费位置必须位于已捕获的 Run 区间内")
        if (
            self.terminal_message_count is not None
            and self.captured_until != self.terminal_message_count
        ):
            raise ValueError("封源必须已完整捕获")
        return self


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


class EvolutionResult(BaseModel):
    """单次 A 的短结果与实际处理区间，不保存模型调用轨迹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["updated", "no_change", "empty", "failed", "cancelled", "conflict"]
    reason: str = ""
    consumed_ranges: tuple[EvolutionRange, ...] = ()
    usage: dict[str, int] = Field(default_factory=dict)
    has_more: bool = False
    effect: str = ""
