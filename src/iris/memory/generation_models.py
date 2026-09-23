"""记忆生成阶段的配置、持久结果和内部批次契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    MemoryCategory,
    MemoryEpisode,
    MemoryEvent,
    MemoryEvidenceRef,
    MemoryItem,
    MemoryItemKind,
    MemoryObservation,
    _new_id,
    _now_iso,
)


class MemoryGenerationConfig(BaseModel):
    """自动维护开关、空闲等待和各生成阶段的独立预算。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    idle_seconds: float = Field(default=300, ge=0)
    flush_input_budget_tokens: int = Field(default=32000, gt=0)
    flush_output_budget_tokens: int = Field(default=4000, gt=0)
    dream_input_budget_tokens: int = Field(default=32000, gt=0)
    dream_output_budget_tokens: int = Field(default=4000, gt=0)


class MemoryConsumedRange(BaseModel):
    """一次成功 flush 实际消费的稳定原文半开区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    episode_id: str
    record_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class GenerationResult(BaseModel):
    """一次阶段执行的实际结果、成本和错误；不混入主 run usage。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(default_factory=_new_id)
    namespace: str
    stage: Literal["capture", "flush", "dream", "overview"]
    status: Literal["completed", "empty", "failed", "cancelled", "conflict", "blocked"]
    usage: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float = 0
    error: str | None = None
    input_ids: tuple[str, ...] = ()
    consumed_ranges: tuple[MemoryConsumedRange, ...] = ()
    counts: dict[str, int] = Field(default_factory=dict)
    item_revision: int | None = None
    has_more: bool = False
    created_at: str = Field(default_factory=_now_iso)


class MemoryCaptureSource(BaseModel):
    """一个 lifecycle run 的持久捕获水位。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    lifecycle_source_id: str
    run_id: str
    session_id: str
    namespace: str
    initial_message_count: int = Field(ge=0)
    captured_until: int = Field(ge=0)
    terminal_message_count: int | None = Field(default=None, ge=0)
    outcome: str | None = None
    registered_revision: int = 0


@dataclass(frozen=True, slots=True)
class EpisodeCursor:
    """指向不可变记录序列内下一个待处理字符。"""

    record_index: int = 0
    text_offset: int = 0


@dataclass(frozen=True, slots=True)
class EpisodeProgress:
    """不可变经历与可变 flush 处理位置的读取投影。"""

    episode: MemoryEpisode
    cursor: EpisodeCursor
    source_outcome: str | None = None


@dataclass(frozen=True, slots=True)
class EpisodeSlice:
    """一次 flush 精确消费的原文半开区间。"""

    episode_id: str
    record_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class FlushCommit:
    """同一事务提交的观察、原文进度与阶段成本。"""

    namespace: str
    slices: tuple[EpisodeSlice, ...]
    observations: tuple[MemoryObservation, ...]
    result: GenerationResult


@dataclass(frozen=True, slots=True)
class ObservationState:
    """观察的历史处理结果，区别于当前 Item 支持关系。"""

    observation: MemoryObservation
    status: str
    item_id: str | None = None
    reason: str = ""
    blocked_budget: int | None = None
    dependency_item_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DreamChange:
    """显式记忆写入的待整理事件。"""

    event_id: str
    item_id: str
    status: str = "pending"
    reason: str = ""
    blocked_budget: int | None = None
    dependency_item_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DreamSnapshot:
    """一次数据库读快照中的固定输入及全部必要比较材料。"""

    namespace: str
    item_revision: int
    observations: tuple[MemoryObservation, ...]
    changes: tuple[DreamChange, ...]
    items: tuple[MemoryItem, ...]
    events: tuple[MemoryEvent, ...]


class DreamOperation(BaseModel):
    """模型提出并由程序绑定新 ID 的知识修订操作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["add", "update", "merge", "delete", "support"]
    target_id: str | None = None
    new_id: str | None = None
    text: str | None = None
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    evidence: tuple[MemoryEvidenceRef, ...] = ()
    merge_ids: tuple[str, ...] = ()
    reason: str = Field(pattern=r"\S")


class ObservationResolution(BaseModel):
    """一个观察本次归入的主条目，或无需保留的原因。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    observation_id: str
    item_id: str | None = None
    reason: str = Field(pattern=r"\S")


class DreamPlan(BaseModel):
    """完整批次的最终条目操作和观察去向。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operations: tuple[DreamOperation, ...] = ()
    resolutions: tuple[ObservationResolution, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationState:
    """当前 namespace 的持久积压、受阻数量和最近执行结果。"""

    namespace: str
    pending_episodes: int
    pending_observations: int
    blocked_observations: int
    pending_changes: int
    blocked_changes: int
    item_revision: int
    projection_revision: int | None
    latest_results: tuple[GenerationResult, ...]
    overview_revision: int | None = None
